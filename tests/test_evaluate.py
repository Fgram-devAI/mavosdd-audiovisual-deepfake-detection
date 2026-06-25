"""Tests for src/evaluate.py metric functions, evaluate_checkpoint, and CLI."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from src import evaluate
from src.models.late_fusion import LateFusionClassifier


# ---------- shared synthetic manifest + .npy fixture helpers ----------

_AUDIO_CSV_FIELDS = [
    "sample_id", "source_video_id", "split", "media_type", "source_folder",
    "provider", "voice_id_or_name", "audio_path", "video_path",
    "audio_feature_path", "lip_feature_path",
    "audio_label", "audio_label_binary",
    "video_label", "video_label_binary",
    "pair_label", "pair_label_binary",
]


def _row(sample_id: str, *, split: str, provider: str, label: int) -> dict:
    blank = {k: "" for k in _AUDIO_CSV_FIELDS}
    blank.update({
        "sample_id": sample_id,
        "source_video_id": sample_id,
        "split": split,
        "media_type": "audio",
        "source_folder": "real" if label == 0 else "tts",
        "provider": provider,
        "audio_path": f"data/audio_wav/{sample_id}.wav",
        "video_path": f"data/raw/{sample_id}.mp4",
        "audio_label": "bonafide" if label == 0 else "spoof",
        "audio_label_binary": str(label),
        "video_label": "real",
        "video_label_binary": "0",
        "pair_label": "matched_bonafide",
        "pair_label_binary": str(label),
    })
    return blank


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_AUDIO_CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_npy(dir_path: Path, sample_id: str, t: int = 16, dim: int = 768) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    arr = np.random.RandomState(abs(hash(sample_id)) % (2**32)).randn(t, dim).astype(np.float32)
    np.save(dir_path / f"{sample_id}.npy", arr)


def _build_eval_fixture(tmp_path: Path, n_per_split: int = 4) -> tuple[Path, Path, Path]:
    """Return (manifest, audio_dir, ckpt_path) with val + test splits populated."""
    manifest = tmp_path / "manifest.csv"
    audio_dir = tmp_path / "feat_codec"
    ckpt_path = tmp_path / "ckpt.pt"

    rows = []
    for i in range(n_per_split):
        for split in ("val", "test"):
            sid = f"{split}_{i}"
            provider = "elevenlabs" if i % 2 == 0 else "google_tts"
            label = i % 2
            rows.append(_row(sid, split=split, provider=provider, label=label))
            _write_npy(audio_dir, sid)
    _write_manifest(manifest, rows)

    model = LateFusionClassifier("audio", emb=128, p=0.3)
    ckpt = {
        "state_dict": model.state_dict(),
        "modality": "audio",
        "backend": "wav2vec2",
        "audio_dir": str(audio_dir),
        "model_hparams": {"modality": "audio", "emb": 128, "dropout": 0.3},
        "norm_stats": {
            "audio_mean": np.zeros(768, dtype=np.float32),
            "audio_std": np.ones(768, dtype=np.float32),
            "eps": 1e-6,
        },
        "val_metrics": {},
        "seed": 42,
        "manifest": str(manifest),
    }
    torch.save(ckpt, ckpt_path)
    return manifest, audio_dir, ckpt_path


class TestRocAuc:
    def test_perfect_separation_is_one(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([0.1, 0.2, 0.8, 0.9])
        assert evaluate.roc_auc(y, score) == pytest.approx(1.0)

    def test_inverted_separation_is_zero(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([0.9, 0.8, 0.2, 0.1])
        assert evaluate.roc_auc(y, score) == pytest.approx(0.0)


class TestEqualErrorRate:
    def test_separable_scores_have_zero_eer(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        score = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
        eer, _ = evaluate.equal_error_rate(y, score)
        assert eer == pytest.approx(0.0, abs=1e-6)

    def test_tied_scores_have_eer_near_half(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        score = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        eer, _ = evaluate.equal_error_rate(y, score)
        assert eer == pytest.approx(0.5, abs=0.5)


class TestF1AtThreshold:
    def test_f1_prec_rec_at_known_threshold(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([0.1, 0.6, 0.4, 0.9])
        f1, prec, rec = evaluate.f1_at_threshold(y, score, threshold=0.5)
        # preds = [0,1,0,1]; tp=1, fp=1, fn=1
        # prec=0.5, rec=0.5, f1=0.5
        assert prec == pytest.approx(0.5)
        assert rec == pytest.approx(0.5)
        assert f1 == pytest.approx(0.5)


class TestConfusion:
    def test_confusion_counts(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        pred = np.array([0, 1, 1, 0, 0, 1])
        cm = evaluate.confusion(y, pred)
        # tn=2, fp=1, fn=1, tp=2
        assert cm == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}


class TestPerProviderRecall:
    def test_per_provider_recall_three_providers(self):
        y = np.array([1, 1, 1, 1, 1, 1])
        pred = np.array([1, 0, 1, 1, 1, 0])
        providers = np.array(["a", "a", "b", "b", "c", "c"])
        result = evaluate.per_provider_recall(y, pred, providers)
        assert result == {"a": pytest.approx(0.5), "b": pytest.approx(1.0), "c": pytest.approx(0.5)}

    def test_skips_provider_with_no_positives(self):
        y = np.array([0, 0, 1, 1])
        pred = np.array([0, 0, 1, 1])
        providers = np.array(["a", "a", "b", "b"])
        result = evaluate.per_provider_recall(y, pred, providers)
        assert "a" not in result
        assert result["b"] == pytest.approx(1.0)


class TestMetricBattery:
    def test_battery_assembles_dict(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        providers = np.array(["x", "x", "x", "y", "y", "y"])
        out = evaluate.metric_battery(y, score, providers)
        assert "roc_auc" in out and out["roc_auc"] == pytest.approx(1.0)
        assert "eer" in out and "eer_threshold" in out
        assert "f1" in out and "precision" in out and "recall" in out
        assert "confusion" in out
        assert "per_provider_recall" in out
        assert "n" in out and out["n"] == 6


# ---------- Task 4: evaluate_checkpoint + test gate + CLI ----------

class TestEvaluateCheckpointGate:
    def test_evaluate_refuses_test_without_flag(self, tmp_path):
        _, _, ckpt = _build_eval_fixture(tmp_path)
        with pytest.raises(SystemExit):
            evaluate.evaluate_checkpoint(ckpt, split="test")

    def test_evaluate_allows_test_with_flag(self, tmp_path):
        _, _, ckpt = _build_eval_fixture(tmp_path)
        out = evaluate.evaluate_checkpoint(ckpt, split="test", allow_test=True, device="cpu")
        assert "roc_auc" in out
        assert "n" in out and out["n"] > 0

    def test_evaluate_val_works_without_flag(self, tmp_path):
        _, _, ckpt = _build_eval_fixture(tmp_path)
        out = evaluate.evaluate_checkpoint(ckpt, split="val", device="cpu")
        assert "roc_auc" in out


class TestEvaluateCLI:
    def test_cli_test_split_without_flag_exits_nonzero(self, tmp_path):
        _, _, ckpt = _build_eval_fixture(tmp_path)
        rc = evaluate.main([
            "--checkpoint", str(ckpt),
            "--split", "test",
            "--device", "cpu",
        ])
        assert rc != 0

    def test_cli_val_split_returns_zero(self, tmp_path):
        _, _, ckpt = _build_eval_fixture(tmp_path)
        rc = evaluate.main([
            "--checkpoint", str(ckpt),
            "--split", "val",
            "--device", "cpu",
        ])
        assert rc == 0


# ---------- Task: visual + fusion checkpoint paths ----------

def _write_npz(dir_path: Path, source_video_id: str, *, t: int = 20, dim: int = 84) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    feats = np.random.RandomState(abs(hash(source_video_id)) % (2**32)) \
        .randn(t, dim).astype(np.float32)
    mask = np.ones(t, dtype=np.float32)
    np.savez(dir_path / f"{source_video_id}.npz", feats=feats, mask=mask)


def _build_visual_eval_fixture(tmp_path: Path, n_per_split: int = 4) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "visual.csv"
    lips_dir = tmp_path / "lips"
    ckpt_path = tmp_path / "ckpt_visual.pt"

    rows = []
    for i in range(n_per_split):
        for split in ("val", "test"):
            sid = f"{split}_{i}"
            provider = "elevenlabs" if i % 2 == 0 else "google_tts"
            label = i % 2
            rows.append(_row(sid, split=split, provider=provider, label=label))
            _write_npz(lips_dir, sid)
    _write_manifest(manifest, rows)

    model = LateFusionClassifier("visual", emb=128, p=0.3)
    ckpt = {
        "state_dict": model.state_dict(),
        "modality": "visual",
        "backend": None,
        "audio_dir": None,
        "model_hparams": {"modality": "visual", "emb": 128, "dropout": 0.3},
        "norm_stats": {
            "lips_mean": np.zeros(84, dtype=np.float32),
            "lips_std": np.ones(84, dtype=np.float32),
            "eps": 1e-6,
        },
        "val_metrics": {},
        "seed": 42,
        "manifest": str(manifest),
    }
    torch.save(ckpt, ckpt_path)
    return manifest, lips_dir, ckpt_path


def _build_fusion_eval_fixture(tmp_path: Path, n_per_split: int = 4) -> tuple[Path, Path, Path, Path]:
    manifest = tmp_path / "fusion.csv"
    audio_dir = tmp_path / "feat_codec"
    lips_dir = tmp_path / "lips"
    ckpt_path = tmp_path / "ckpt_fusion.pt"

    rows = []
    for i in range(n_per_split):
        for split in ("val", "test"):
            sid = f"{split}_{i}"
            provider = "elevenlabs" if i % 2 == 0 else "google_tts"
            label = i % 2
            rows.append(_row(sid, split=split, provider=provider, label=label))
            _write_npy(audio_dir, sid)
            _write_npz(lips_dir, sid)
    _write_manifest(manifest, rows)

    model = LateFusionClassifier("fusion", emb=128, p=0.3)
    ckpt = {
        "state_dict": model.state_dict(),
        "modality": "fusion",
        "backend": "wav2vec2",
        "audio_dir": str(audio_dir),
        "model_hparams": {"modality": "fusion", "emb": 128, "dropout": 0.3},
        "norm_stats": {
            "audio_mean": np.zeros(768, dtype=np.float32),
            "audio_std": np.ones(768, dtype=np.float32),
            "lips_mean": np.zeros(84, dtype=np.float32),
            "lips_std": np.ones(84, dtype=np.float32),
            "eps": 1e-6,
        },
        "val_metrics": {},
        "seed": 42,
        "manifest": str(manifest),
    }
    torch.save(ckpt, ckpt_path)
    return manifest, audio_dir, lips_dir, ckpt_path


class TestEvaluateCheckpointVisual:
    def test_visual_refuses_test_without_flag(self, tmp_path):
        _, _, ckpt = _build_visual_eval_fixture(tmp_path)
        with pytest.raises(SystemExit):
            evaluate.evaluate_checkpoint(ckpt, split="test")

    def test_visual_allows_test_with_flag(self, tmp_path):
        _, lips_dir, ckpt = _build_visual_eval_fixture(tmp_path)
        out = evaluate.evaluate_checkpoint(
            ckpt, split="test", allow_test=True, device="cpu", lips_dir=lips_dir,
        )
        assert "roc_auc" in out and out["n"] > 0

    def test_visual_val_works_without_flag(self, tmp_path):
        _, lips_dir, ckpt = _build_visual_eval_fixture(tmp_path)
        out = evaluate.evaluate_checkpoint(
            ckpt, split="val", device="cpu", lips_dir=lips_dir,
        )
        assert "roc_auc" in out


class TestEvaluateCheckpointFusion:
    def test_fusion_refuses_test_without_flag(self, tmp_path):
        _, _, _, ckpt = _build_fusion_eval_fixture(tmp_path)
        with pytest.raises(SystemExit):
            evaluate.evaluate_checkpoint(ckpt, split="test")

    def test_fusion_allows_test_with_flag(self, tmp_path):
        _, _, lips_dir, ckpt = _build_fusion_eval_fixture(tmp_path)
        out = evaluate.evaluate_checkpoint(
            ckpt, split="test", allow_test=True, device="cpu", lips_dir=lips_dir,
        )
        assert "roc_auc" in out and out["n"] > 0
