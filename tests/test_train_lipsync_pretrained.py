"""Tests for src/train_lipsync_pretrained.py."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch


PAIR_FIELDS = [
    "pair_id", "split", "source_video_id", "lip_feature_path",
    "audio_sample_id", "audio_path", "audio_feature_path", "audio_provider",
    "audio_label", "sync_label", "sync_label_binary", "negative_type",
    "source_folder", "voice_id_or_name",
]


def _row(pair_id, split, sync):
    r = {k: "" for k in PAIR_FIELDS}
    r.update({
        "pair_id": pair_id, "split": split,
        "source_video_id": "v1", "audio_sample_id": "a1",
        "audio_provider": "original", "sync_label_binary": str(sync),
        "source_folder": "real", "negative_type": "" if sync == 0 else "mismatched_original",
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


def test_resolve_backend_maps_syncnet_and_avhubert():
    from src.train_lipsync_pretrained import resolve_backend
    from src import common

    v, a, f, d = resolve_backend("syncnet")
    assert v == common.FEAT_SYNCNET_VISUAL_DIR
    assert a == common.FEAT_SYNCNET_AUDIO_DIR
    assert f == common.SYNCNET_FAILURES_CSV
    assert d == 512

    v, a, f, d = resolve_backend("avhubert")
    assert v == common.FEAT_AVHUBERT_VISUAL_DIR
    assert a == common.FEAT_AVHUBERT_AUDIO_DIR
    assert f == common.AVHUBERT_FAILURES_CSV
    assert d == 768


def test_resolve_backend_raises_on_unknown():
    from src.train_lipsync_pretrained import resolve_backend
    with pytest.raises(ValueError, match=r"unknown backend"):
        resolve_backend("nope")


def test_train_smoke_writes_checkpoint(tmp_path, monkeypatch):
    from src.train_lipsync_pretrained import PretrainedTrainConfig, train

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [
        _row("p1", "train", 0),
        _row("p2", "train", 1),
        _row("p3", "val", 0),
        _row("p4", "val", 1),
    ])
    vdir, adir = tmp_path / "v", tmp_path / "a"
    _write_emb(vdir, "v1"); _write_emb(adir, "a1")
    out_ckpt = tmp_path / "best.pt"
    runs_dir = tmp_path / "runs"

    cfg = PretrainedTrainConfig(
        backend="syncnet",
        manifest=manifest,
        visual_dir=vdir,
        audio_dir=adir,
        failures_csv=None,
        run_name="smoke",
        runs_dir=runs_dir,
        out=out_ckpt,
        embed_dim=512,
        epochs=2,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        dropout=0.0,
        patience=2,
        hidden=64,
        device="cpu",
        seed=42,
    )
    train(cfg)
    assert out_ckpt.exists()
    assert (runs_dir / "smoke" / "metrics.csv").exists()


def test_main_splits_check_only_resolves_backend_paths(tmp_path):
    """The trainer never opens the test split (it hardcodes train/val in
    ``train()``), and the dataset layer already refuses ``split='test'`` — that
    guarantee is covered by ``tests/data/test_lipsync_pretrained_dataset.py::
    test_dataset_refuses_test_split``. This test only asserts that
    ``--splits-check-only`` short-circuits after CLI arg + backend-path
    resolution without instantiating any dataset."""
    from src.train_lipsync_pretrained import main

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("p1", "train", 0)])
    rc = main([
        "--backend", "syncnet",
        "--manifest", str(manifest),
        "--visual-dir", str(tmp_path / "v"),
        "--audio-dir", str(tmp_path / "a"),
        "--run-name", "smoke",
        "--runs-dir", str(tmp_path / "runs"),
        "--out", str(tmp_path / "b.pt"),
        "--splits-check-only",
    ])
    assert rc == 0
