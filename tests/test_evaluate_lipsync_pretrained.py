"""Tests for src/evaluate_lipsync_pretrained.py."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


PAIR_FIELDS = [
    "pair_id", "split", "source_video_id", "lip_feature_path",
    "audio_sample_id", "audio_path", "audio_feature_path", "audio_provider",
    "audio_label", "sync_label", "sync_label_binary", "negative_type",
    "source_folder", "voice_id_or_name",
]


def _row(pair_id, sync, provider="original", neg_type="", vid="v1", aid="a1"):
    r = {k: "" for k in PAIR_FIELDS}
    r.update({
        "pair_id": pair_id, "split": "val", "source_video_id": vid,
        "audio_sample_id": aid, "audio_provider": provider,
        "sync_label_binary": str(sync), "negative_type": neg_type,
        "source_folder": "real",
    })
    return r


def _write_manifest(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PAIR_FIELDS); w.writeheader()
        for r in rows: w.writerow(r)


def _write_emb(d, key, dim=512, n_windows=20):
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{key}.npy", np.random.rand(n_windows, dim).astype(np.float16))


def test_compute_metrics_computes_full_schema():
    from src.evaluate_lipsync_pretrained import compute_metrics

    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3])
    labels = np.array([0, 1, 1, 0, 1, 1])
    providers = ["original", "elevenlabs", "google_tts", "original", "original", "openai_tts"]
    negatives = ["", "mismatched_generated", "mismatched_generated", "", "mismatched_original", "generated_same_transcript"]

    m = compute_metrics(scores, labels, providers, negatives)
    assert set(m.keys()) >= {
        "roc_auc", "eer", "threshold_used", "f1",
        "confusion", "sync_accuracy",
        "per_negative_type_recall", "per_provider_recall",
    }
    assert 0.0 <= m["roc_auc"] <= 1.0


def test_compute_metrics_sanity_guardrail_provider_equals_negative_type():
    from src.evaluate_lipsync_pretrained import compute_metrics

    scores = np.array([0.9, 0.9, 0.1, 0.9])
    labels = np.array([1, 1, 0, 1])
    providers = ["original", "elevenlabs", "original", "original"]
    negatives = ["mismatched_original", "mismatched_generated", "", "mismatched_original"]

    m = compute_metrics(scores, labels, providers, negatives)
    orig_provider = m["per_provider_recall"]["original"]
    orig_neg = m["per_negative_type_recall"]["mismatched_original"]
    assert abs(orig_provider - orig_neg) < 1e-9


def test_evaluate_backend_refuses_partial_by_default(tmp_path, monkeypatch):
    from src.evaluate_lipsync_pretrained import evaluate_backend, EvaluationRefusedError
    from src.models.lipsync_pretrained_head import LipSyncPretrainedHead

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("p1", 1), _row("p2", 0)])
    vdir, adir = tmp_path / "v", tmp_path / "a"
    _write_emb(vdir, "v1"); _write_emb(adir, "a1")

    head = LipSyncPretrainedHead(sync_feature_dim=7, embed_dim=512)
    ckpt = tmp_path / "best.pt"
    torch.save({"state_dict": head.state_dict(),
                "config": {"embed_dim": 512, "hidden": 128, "dropout": 0.3}}, ckpt)

    fake_fail = tmp_path / "fail.csv"
    fake_fail.write_text("sample_id,stage,error_type,error_message,timestamp\nv1,face_detect,X,y,z\n")

    monkeypatch.setattr("src.evaluate_lipsync_pretrained.resolve_backend",
                        lambda b: (vdir, adir, fake_fail, 512))
    with pytest.raises(EvaluationRefusedError):
        evaluate_backend(
            checkpoint=ckpt, backend="syncnet", manifest=manifest,
            split="val", allow_partial=False, out=tmp_path / "syncnet_val.txt",
        )


def test_evaluate_backend_allows_partial_with_flag(tmp_path, monkeypatch):
    """Partial evaluation: one row excluded via fail_csv, two rows evaluated (both classes).

    Uses three distinct rows (p1 excluded, p2 and p3 evaluated) so that
    n_evaluated=2 with both label classes present, satisfying the fail-fast
    guards added in the plan revision while still exercising the partial-ok path.
    """
    from src.evaluate_lipsync_pretrained import evaluate_backend
    from src.models.lipsync_pretrained_head import LipSyncPretrainedHead

    manifest = tmp_path / "m.csv"
    # p1: sync=1, audio=a1 (failed) → excluded
    # p2: sync=0, video=v2, audio=a2 → evaluated
    # p3: sync=1, video=v3, audio=a3 → evaluated
    _write_manifest(manifest, [
        _row("p1", 1, provider="original", neg_type="mismatched_original", vid="v1", aid="a1"),
        _row("p2", 0, provider="original", vid="v2", aid="a2"),
        _row("p3", 1, provider="original", neg_type="mismatched_original", vid="v3", aid="a3"),
    ])
    vdir, adir = tmp_path / "v", tmp_path / "a"
    # v1 and a1 exist in filesystem, but a1 is in the failures CSV
    _write_emb(vdir, "v1"); _write_emb(adir, "a1")
    _write_emb(vdir, "v2"); _write_emb(adir, "a2")
    _write_emb(vdir, "v3"); _write_emb(adir, "a3")

    head = LipSyncPretrainedHead(sync_feature_dim=7, embed_dim=512)
    ckpt = tmp_path / "best.pt"
    torch.save({"state_dict": head.state_dict(),
                "config": {"embed_dim": 512, "hidden": 128, "dropout": 0.3}}, ckpt)

    # a1 is in the failures CSV → p1 is excluded; p2 and p3 pass through
    fail_csv = tmp_path / "fail.csv"
    fail_csv.write_text("sample_id,stage,error_type,error_message,timestamp\na1,audio_decode,X,y,z\n")

    monkeypatch.setattr("src.evaluate_lipsync_pretrained.resolve_backend",
                        lambda b: (vdir, adir, fail_csv, 512))
    out = tmp_path / "syncnet_val.txt"
    rc = evaluate_backend(
        checkpoint=ckpt, backend="syncnet", manifest=manifest,
        split="val", allow_partial=True, out=out,
    )
    assert rc == 0
    text = out.read_text()
    assert "partial_evaluation=true" in text
    assert "n_manifest=3" in text
    assert "n_excluded=1" in text


def test_evaluate_backend_refuses_test_split(tmp_path):
    from src.evaluate_lipsync_pretrained import evaluate_backend

    manifest = tmp_path / "m.csv"
    with pytest.raises(ValueError, match=r"test split"):
        evaluate_backend(
            checkpoint=Path("/nonexistent"), backend="syncnet",
            manifest=manifest, split="test", allow_partial=False,
            out=tmp_path / "x",
        )


def test_evaluate_backend_refuses_zero_evaluated_even_with_allow_partial(tmp_path, monkeypatch):
    from src.evaluate_lipsync_pretrained import evaluate_backend, EvaluationRefusedError
    from src.models.lipsync_pretrained_head import LipSyncPretrainedHead

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("p1", 1)])
    vdir, adir = tmp_path / "v", tmp_path / "a"
    # No embeddings written → row is excluded (files don't exist)

    head = LipSyncPretrainedHead(sync_feature_dim=7, embed_dim=512)
    ckpt = tmp_path / "best.pt"
    torch.save({"state_dict": head.state_dict(),
                "config": {"embed_dim": 512, "hidden": 128, "dropout": 0.3}}, ckpt)

    monkeypatch.setattr("src.evaluate_lipsync_pretrained.resolve_backend",
                        lambda b: (vdir, adir, tmp_path / "no.csv", 512))
    with pytest.raises(EvaluationRefusedError, match=r"n_evaluated=0"):
        evaluate_backend(
            checkpoint=ckpt, backend="syncnet", manifest=manifest,
            split="val", allow_partial=True, out=tmp_path / "syncnet_val.txt",
        )


def test_evaluate_backend_refuses_single_class_labels_even_with_allow_partial(tmp_path, monkeypatch):
    from src.evaluate_lipsync_pretrained import evaluate_backend, EvaluationRefusedError
    from src.models.lipsync_pretrained_head import LipSyncPretrainedHead

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("p1", 1), _row("p2", 1)])
    vdir, adir = tmp_path / "v", tmp_path / "a"
    _write_emb(vdir, "v1"); _write_emb(adir, "a1")

    head = LipSyncPretrainedHead(sync_feature_dim=7, embed_dim=512)
    ckpt = tmp_path / "best.pt"
    torch.save({"state_dict": head.state_dict(),
                "config": {"embed_dim": 512, "hidden": 128, "dropout": 0.3}}, ckpt)

    monkeypatch.setattr("src.evaluate_lipsync_pretrained.resolve_backend",
                        lambda b: (vdir, adir, tmp_path / "no.csv", 512))
    with pytest.raises(EvaluationRefusedError, match=r"missing a label class"):
        evaluate_backend(
            checkpoint=ckpt, backend="syncnet", manifest=manifest,
            split="val", allow_partial=True, out=tmp_path / "syncnet_val.txt",
        )


def test_emit_comparison_writes_three_row_table(tmp_path):
    from src.evaluate_lipsync_pretrained import emit_comparison

    wavlm = tmp_path / "wavlm.txt"
    wavlm.write_text(
        "split=val n_manifest=3168 n_evaluated=3168 n_excluded=0 excluded_by_reason={} "
        "partial_evaluation=false positive_class=async_inconsistent_pair "
        "roc_auc=0.8409 eer=0.2527 threshold_used=0.6726 f1=0.8261\n"
        "confusion={'tn': 465, 'fp': 157, 'fn': 644, 'tp': 1902}\n"
        "sync_accuracy=0.7476\n"
        "per_negative_type_recall={'generated_same_transcript': 1.0, 'mismatched_generated': 0.999, 'mismatched_original': 0.4831}\n"
        "per_provider_recall={'elevenlabs': 1.0, 'google_tts': 1.0, 'openai_tts': 0.998, 'original': 0.4831}\n"
    )
    syncnet = tmp_path / "syncnet.txt"; syncnet.write_text(wavlm.read_text())
    avhubert = tmp_path / "avhubert.txt"; avhubert.write_text("N/A — Task 0 blocked: import error\n")

    out = tmp_path / "cmp.md"
    rc = emit_comparison(
        wavlm_val=wavlm, syncnet_val=syncnet, avhubert_val=avhubert, out=out,
    )
    assert rc == 0
    text = out.read_text()
    assert "WavLM" in text and "SyncNet" in text and "AV-HuBERT" in text
    assert "0.4831" in text
    assert "blocked" in text.lower()
