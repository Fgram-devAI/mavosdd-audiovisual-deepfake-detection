"""Tests for src/evaluate_lipsync.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def trained_ckpt(tmp_path):
    from tests.test_train_lipsync import (
        _row, _write_manifest,
    )
    from src.train_lipsync import LipSyncRunConfig, run_lipsync_training

    rng = np.random.default_rng(42)
    audio_dir = tmp_path / "audio"
    lips_dir = tmp_path / "lips"
    for name in ["A", "B", "C", "D", "gen_A", "gen_B", "gen_C", "gen_D"]:
        audio_dir.mkdir(parents=True, exist_ok=True)
        np.save(audio_dir / f"{name}.npy",
                rng.standard_normal((80, 768)).astype(np.float32))
    for vid in ["A", "B", "C", "D"]:
        lips_dir.mkdir(parents=True, exist_ok=True)
        np.savez(lips_dir / f"{vid}.npz",
                 feats=rng.standard_normal((20, 84)).astype(np.float32),
                 mask=np.ones(20, dtype=np.float32))
    manifest = tmp_path / "pairs.csv"
    rows = [
        _row("pos__A", "A", "A", "train", "0", "", "original", "bonafide"),
        _row("pos__B", "B", "B", "train", "0", "", "original", "bonafide"),
        _row("neg__A", "gen_A", "A", "train", "1",
             "generated_same_transcript", "elevenlabs", "spoof"),
        _row("neg__B", "gen_B", "B", "train", "1", "mismatched_original",
             "original", "bonafide"),
        _row("pos__C", "C", "C", "val", "0", "", "original", "bonafide"),
        _row("neg__C", "gen_C", "C", "val", "1",
             "generated_same_transcript", "elevenlabs", "spoof"),
        _row("pos__D", "D", "D", "val", "0", "", "original", "bonafide"),
        _row("neg__D", "gen_D", "D", "val", "1", "mismatched_original",
             "original", "bonafide"),
    ]
    _write_manifest(manifest, rows)

    cfg = LipSyncRunConfig(
        backend="wav2vec2", manifest=manifest,
        audio_dir=audio_dir, lips_dir=lips_dir,
        run_name="eval_smoke",
        runs_dir=tmp_path / "runs",
        checkpoint_path=tmp_path / "ckpt.pt",
        epochs=1, batch_size=2, lr=1e-3,
        weight_decay=1e-2, dropout=0.0,
        patience=5, device="cpu", seed=42,
    )
    run_lipsync_training(cfg)
    return {"ckpt": tmp_path / "ckpt.pt", "manifest": manifest,
            "audio_dir": audio_dir, "lips_dir": lips_dir}


def test_evaluate_returns_breakdown_keys(trained_ckpt):
    from src.evaluate_lipsync import evaluate_lipsync_checkpoint

    result = evaluate_lipsync_checkpoint(
        trained_ckpt["ckpt"], split="val", device="cpu",
        manifest=trained_ckpt["manifest"],
        audio_dir=trained_ckpt["audio_dir"],
        lips_dir=trained_ckpt["lips_dir"],
    )

    assert "roc_auc" in result
    assert "per_negative_type_recall" in result
    assert "per_provider_recall" in result
    assert result["positive_class"] == "async_inconsistent_pair"
    assert "positive_sync_accuracy" in result
    assert "threshold_used" in result


def test_evaluate_refuses_test_split(trained_ckpt):
    from src.evaluate_lipsync import evaluate_lipsync_checkpoint

    with pytest.raises(SystemExit, match="Refusing to evaluate on test split"):
        evaluate_lipsync_checkpoint(
            trained_ckpt["ckpt"], split="test", device="cpu",
            manifest=trained_ckpt["manifest"],
            audio_dir=trained_ckpt["audio_dir"],
            lips_dir=trained_ckpt["lips_dir"],
        )
