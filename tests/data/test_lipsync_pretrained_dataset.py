"""Tests for src/data/lipsync_pretrained_dataset.py."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest


PAIR_FIELDS = [
    "pair_id", "split", "source_video_id", "lip_feature_path",
    "audio_sample_id", "audio_path", "audio_feature_path", "audio_provider",
    "audio_label", "sync_label", "sync_label_binary", "negative_type",
    "source_folder", "voice_id_or_name",
]


def _row(pair_id, *, split="train", vid="v1", aid="a1", provider="original",
         neg_type="", sync_label_binary="0"):
    r = {k: "" for k in PAIR_FIELDS}
    r.update({
        "pair_id": pair_id, "split": split, "source_video_id": vid,
        "audio_sample_id": aid, "audio_provider": provider,
        "sync_label_binary": sync_label_binary, "negative_type": neg_type,
        "source_folder": "real",
    })
    return r


def _write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PAIR_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_emb(dirpath: Path, key: str, shape=(20, 512)):
    dirpath.mkdir(parents=True, exist_ok=True)
    np.save(dirpath / f"{key}.npy", np.random.rand(*shape).astype(np.float16))


def test_compute_sync_features_returns_fixed_length_vector():
    from src.data.lipsync_pretrained_dataset import compute_sync_features, SYNC_FEATURE_DIM

    v = np.random.rand(20, 512).astype(np.float32)
    a = np.random.rand(20, 512).astype(np.float32)
    feats = compute_sync_features(v, a)
    assert feats.shape == (SYNC_FEATURE_DIM,)
    assert feats.dtype == np.float32


def test_compute_sync_features_zero_offset_dominates_for_aligned_streams():
    from src.data.lipsync_pretrained_dataset import compute_sync_features, SYNC_FEATURE_NAMES

    rng = np.random.default_rng(0)
    v = rng.standard_normal((20, 128)).astype(np.float32)
    a = v + 0.01 * rng.standard_normal((20, 128)).astype(np.float32)
    feats = compute_sync_features(v, a)
    d = dict(zip(SYNC_FEATURE_NAMES, feats.tolist()))
    assert d["mean_cos_sim_zero_offset"] > 0.9
    assert d["zero_minus_best_offset"] > -0.05


def test_dataset_returns_sync_features_and_pooled_vectors(tmp_path):
    from src.data.lipsync_pretrained_dataset import (
        LipSyncPretrainedDataset, SYNC_FEATURE_DIM,
    )

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("p1", vid="v1", aid="a1", sync_label_binary="1")])
    vdir, adir = tmp_path / "v", tmp_path / "a"
    _write_emb(vdir, "v1"); _write_emb(adir, "a1")

    ds = LipSyncPretrainedDataset(
        manifest=manifest, split="train", backend="syncnet",
        visual_dir=vdir, audio_dir=adir,
    )
    item = ds[0]
    assert item["sync_features"].shape == (SYNC_FEATURE_DIM,)
    assert item["pooled_visual"].shape == (512,)
    assert item["pooled_audio"].shape == (512,)
    assert float(item["sync_label"]) == 1.0
    assert item["pair_id"] == "p1"


def test_dataset_excludes_rows_missing_visual_or_audio(tmp_path):
    from src.data.lipsync_pretrained_dataset import LipSyncPretrainedDataset

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [
        _row("p1", vid="v1", aid="a1"),
        _row("p2", vid="v_missing", aid="a1"),
        _row("p3", vid="v1", aid="a_missing"),
    ])
    vdir, adir = tmp_path / "v", tmp_path / "a"
    _write_emb(vdir, "v1"); _write_emb(adir, "a1")

    ds = LipSyncPretrainedDataset(
        manifest=manifest, split="train", backend="syncnet",
        visual_dir=vdir, audio_dir=adir,
    )
    assert len(ds) == 1
    assert ds.excluded_pair_ids == {"p2", "p3"}


def test_dataset_excludes_rows_in_failures_csv(tmp_path):
    from src.data.lipsync_pretrained_dataset import LipSyncPretrainedDataset

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("p1", vid="v1", aid="a1")])
    vdir, adir = tmp_path / "v", tmp_path / "a"
    _write_emb(vdir, "v1"); _write_emb(adir, "a1")

    fail = tmp_path / "fail.csv"
    fail.write_text("sample_id,stage,error_type,error_message,timestamp\nv1,face_detect,X,y,2026\n")

    ds = LipSyncPretrainedDataset(
        manifest=manifest, split="train", backend="syncnet",
        visual_dir=vdir, audio_dir=adir, failures_csv=fail,
    )
    assert len(ds) == 0
    assert ds.excluded_pair_ids == {"p1"}


def test_dataset_refuses_test_split(tmp_path):
    from src.data.lipsync_pretrained_dataset import LipSyncPretrainedDataset

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("p1", split="test")])
    vdir, adir = tmp_path / "v", tmp_path / "a"

    with pytest.raises(ValueError, match=r"test split"):
        LipSyncPretrainedDataset(
            manifest=manifest, split="test", backend="syncnet",
            visual_dir=vdir, audio_dir=adir,
        )


def test_make_dataloader_yields_batched_tensors(tmp_path):
    from src.data.lipsync_pretrained_dataset import (
        LipSyncPretrainedDataset, SYNC_FEATURE_DIM, make_dataloader,
    )

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [
        _row(f"p{i}", vid="v1", aid="a1", sync_label_binary=str(i % 2)) for i in range(4)
    ])
    vdir, adir = tmp_path / "v", tmp_path / "a"
    _write_emb(vdir, "v1"); _write_emb(adir, "a1")

    ds = LipSyncPretrainedDataset(
        manifest=manifest, split="train", backend="syncnet",
        visual_dir=vdir, audio_dir=adir,
    )
    dl = make_dataloader(ds, batch_size=2, shuffle=False, num_workers=0, seed=42)
    batch = next(iter(dl))
    assert batch["sync_features"].shape == (2, SYNC_FEATURE_DIM)
    assert batch["pooled_visual"].shape == (2, 512)
    assert batch["pooled_audio"].shape == (2, 512)
    assert batch["sync_label"].shape == (2,)
