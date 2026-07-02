"""Smoke test for src/train_lipsync.py — trains one epoch on a tiny fixture."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch


def _write_manifest(path: Path, rows: list[dict]) -> None:
    from src.data.build_lipsync_pairs import LIPSYNC_MANIFEST_SCHEMA
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(LIPSYNC_MANIFEST_SCHEMA))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in LIPSYNC_MANIFEST_SCHEMA})


def _row(pair_id: str, sid: str, vid: str, split: str, sync_bin: str,
         negative_type: str, provider: str, label: str) -> dict:
    return {
        "pair_id": pair_id, "split": split, "source_video_id": vid,
        "lip_feature_path": f"data/features/lips/{vid}.npz",
        "audio_sample_id": sid,
        "audio_path": f"data/audio_wav/real/{sid}.wav",
        "audio_feature_path": f"data/features/audio_wav2vec2_codec/{sid}.npy",
        "audio_provider": provider, "audio_label": label,
        "sync_label": "sync" if sync_bin == "0" else "async",
        "sync_label_binary": sync_bin, "negative_type": negative_type,
        "source_folder": "real", "voice_id_or_name": "",
    }


@pytest.fixture
def tiny_fixture(tmp_path):
    rng = np.random.default_rng(42)
    audio_dir = tmp_path / "audio"
    lips_dir = tmp_path / "lips"
    for name in ["A", "B", "C", "D", "gen_A", "gen_B", "gen_C", "gen_D"]:
        (audio_dir).mkdir(parents=True, exist_ok=True)
        np.save(audio_dir / f"{name}.npy",
                rng.standard_normal((80, 768)).astype(np.float32))
    for vid in ["A", "B", "C", "D"]:
        (lips_dir).mkdir(parents=True, exist_ok=True)
        np.savez(lips_dir / f"{vid}.npz",
                 feats=rng.standard_normal((20, 84)).astype(np.float32),
                 mask=np.ones(20, dtype=np.float32))

    manifest = tmp_path / "pairs.csv"
    rows = [
        _row("pos__A", "A", "A", "train", "0", "", "original", "bonafide"),
        _row("pos__B", "B", "B", "train", "0", "", "original", "bonafide"),
        _row("neg__A", "gen_A", "A", "train", "1",
             "generated_same_transcript", "elevenlabs", "spoof"),
        _row("neg__B", "gen_B", "B", "train", "1",
             "generated_same_transcript", "google_tts", "spoof"),
        _row("pos__C", "C", "C", "val", "0", "", "original", "bonafide"),
        _row("neg__C", "gen_C", "C", "val", "1",
             "generated_same_transcript", "elevenlabs", "spoof"),
        _row("pos__D", "D", "D", "val", "0", "", "original", "bonafide"),
        _row("neg__D", "gen_D", "D", "val", "1", "mismatched_original",
             "original", "bonafide"),
    ]
    _write_manifest(manifest, rows)
    return {"manifest": manifest, "audio_dir": audio_dir, "lips_dir": lips_dir}


def test_run_lipsync_training_writes_checkpoint_and_metrics_csv(tmp_path, tiny_fixture):
    from src.train_lipsync import LipSyncRunConfig, run_lipsync_training

    cfg = LipSyncRunConfig(
        backend="wav2vec2",
        manifest=tiny_fixture["manifest"],
        audio_dir=tiny_fixture["audio_dir"],
        lips_dir=tiny_fixture["lips_dir"],
        run_name="smoke",
        runs_dir=tmp_path / "runs",
        checkpoint_path=tmp_path / "ckpt.pt",
        epochs=2,
        batch_size=2,
        lr=1e-3,
        weight_decay=1e-2,
        dropout=0.0,
        patience=5,
        device="cpu",
        seed=42,
    )

    result = run_lipsync_training(cfg)

    assert (tmp_path / "ckpt.pt").exists()
    assert (tmp_path / "runs" / "smoke" / "metrics.csv").exists()
    assert "best_epoch" in result and result["best_epoch"] >= 1
    assert "roc_auc" in result["best_val_metrics"]
    ckpt = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)
    for key in ("state_dict", "model_hparams", "backend", "audio_dir", "lips_dir",
                "manifest", "val_metrics", "seed"):
        assert key in ckpt, f"missing checkpoint key: {key}"
