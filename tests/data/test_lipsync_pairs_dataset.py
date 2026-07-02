"""Tests for src/data/lipsync_pairs.py::LipSyncPairDataset."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch


def _write_pair_manifest(path: Path, rows: list[dict]) -> None:
    from src.data.build_lipsync_pairs import LIPSYNC_MANIFEST_SCHEMA

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(LIPSYNC_MANIFEST_SCHEMA))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in LIPSYNC_MANIFEST_SCHEMA})


def _write_audio(path: Path, t_frames: int = 199) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.random.default_rng(0).standard_normal((t_frames, 768)).astype(np.float32))


def _write_lips(path: Path, t_frames: int = 20, feat_dim: int = 84) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        feats=np.random.default_rng(0).standard_normal((t_frames, feat_dim)).astype(np.float32),
        mask=np.ones(t_frames, dtype=np.float32),
    )


def _row(pair_id: str, *, audio_sample_id: str, vid: str, split: str = "train",
         sync_bin: str = "0", negative_type: str = "", provider: str = "original",
         label: str = "bonafide") -> dict:
    return {
        "pair_id": pair_id, "split": split, "source_video_id": vid,
        "lip_feature_path": f"data/features/lips/{vid}.npz",
        "audio_sample_id": audio_sample_id,
        "audio_path": f"data/audio_wav/real/{audio_sample_id}.wav",
        "audio_feature_path": f"data/features/audio_wav2vec2_codec/{audio_sample_id}.npy",
        "audio_provider": provider, "audio_label": label,
        "sync_label": "sync" if sync_bin == "0" else "async",
        "sync_label_binary": sync_bin, "negative_type": negative_type,
        "source_folder": "real", "voice_id_or_name": "",
    }


def test_lipsync_dataset_returns_expected_shapes_and_label(tmp_path):
    from src.data.lipsync_pairs import LipSyncPairDataset

    audio_dir = tmp_path / "audio"
    lips_dir = tmp_path / "lips"
    _write_audio(audio_dir / "A.npy")
    _write_lips(lips_dir / "A.npz")

    manifest = tmp_path / "pairs.csv"
    _write_pair_manifest(manifest, [_row("pos__A", audio_sample_id="A", vid="A")])

    ds = LipSyncPairDataset(
        manifest_path=manifest, split="train", backend="wav2vec2",
        audio_dir=audio_dir, lips_dir=lips_dir,
    )
    assert len(ds) == 1
    item = ds[0]
    assert item["audio"].dtype == torch.float32
    assert item["audio"].ndim == 2 and item["audio"].shape[1] == 768
    assert tuple(item["lips"].shape) == (20, 84)
    assert tuple(item["lips_mask"].shape) == (20,)
    assert item["label"].dtype == torch.long
    assert int(item["label"]) == 0
    assert item["pair_id"] == "pos__A"
    assert item["negative_type"] == ""


def test_lipsync_dataset_filters_by_split(tmp_path):
    from src.data.lipsync_pairs import LipSyncPairDataset

    audio_dir = tmp_path / "audio"; lips_dir = tmp_path / "lips"
    for vid in ("A", "B"):
        _write_audio(audio_dir / f"{vid}.npy")
        _write_lips(lips_dir / f"{vid}.npz")

    manifest = tmp_path / "pairs.csv"
    _write_pair_manifest(manifest, [
        _row("pos__A", audio_sample_id="A", vid="A", split="train"),
        _row("pos__B", audio_sample_id="B", vid="B", split="val"),
    ])

    train = LipSyncPairDataset(manifest_path=manifest, split="train",
                               backend="wav2vec2", audio_dir=audio_dir, lips_dir=lips_dir)
    val = LipSyncPairDataset(manifest_path=manifest, split="val",
                             backend="wav2vec2", audio_dir=audio_dir, lips_dir=lips_dir)
    assert len(train) == 1 and len(val) == 1
    assert train[0]["source_video_id"] == "A"
    assert val[0]["source_video_id"] == "B"


def test_lipsync_dataset_raises_when_audio_feature_missing(tmp_path):
    from src.data.feature_store import FeatureStoreValidationError
    from src.data.lipsync_pairs import LipSyncPairDataset

    audio_dir = tmp_path / "audio"; lips_dir = tmp_path / "lips"
    _write_lips(lips_dir / "A.npz")  # no audio .npy on disk

    manifest = tmp_path / "pairs.csv"
    _write_pair_manifest(manifest, [_row("pos__A", audio_sample_id="A", vid="A")])

    ds = LipSyncPairDataset(manifest_path=manifest, split="train",
                            backend="wav2vec2", audio_dir=audio_dir, lips_dir=lips_dir)
    with pytest.raises(FeatureStoreValidationError, match="missing audio feature"):
        ds[0]


def test_lipsync_collate_stacks_tensors_and_carries_metadata(tmp_path):
    from src.data.lipsync_pairs import LipSyncPairDataset, lipsync_collate

    audio_dir = tmp_path / "audio"; lips_dir = tmp_path / "lips"
    for vid in ("A", "B"):
        _write_audio(audio_dir / f"{vid}.npy")
        _write_lips(lips_dir / f"{vid}.npz")

    manifest = tmp_path / "pairs.csv"
    _write_pair_manifest(manifest, [
        _row("pos__A", audio_sample_id="A", vid="A"),
        _row("neg__gst__A_0", audio_sample_id="B", vid="A", sync_bin="1",
             negative_type="generated_same_transcript", provider="elevenlabs",
             label="spoof"),
    ])
    ds = LipSyncPairDataset(manifest_path=manifest, split="train",
                            backend="wav2vec2", audio_dir=audio_dir, lips_dir=lips_dir)
    batch = lipsync_collate([ds[0], ds[1]])

    assert tuple(batch["audio"].shape) == (2, 199, 768)
    assert tuple(batch["lips"].shape) == (2, 20, 84)
    assert tuple(batch["lips_mask"].shape) == (2, 20)
    assert batch["label"].tolist() == [0, 1]
    assert [m["negative_type"] for m in batch["metadata"]] == ["", "generated_same_transcript"]
    assert [m["audio_provider"] for m in batch["metadata"]] == ["original", "elevenlabs"]
