"""Tests for src/data/feature_store.py."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------- Task 1: skeleton ----------

def test_feature_store_validation_error_is_exception():
    from src.data.feature_store import FeatureStoreValidationError

    assert issubclass(FeatureStoreValidationError, Exception)


def test_supported_backends_and_audio_dim_constants():
    from src.data.feature_store import AUDIO_FEATURE_DIM, SUPPORTED_BACKENDS

    assert AUDIO_FEATURE_DIM == 768
    assert SUPPORTED_BACKENDS == ("wav2vec2", "wavlm", "hubert")


def test_resolve_audio_backend_dir_maps_each_backend():
    from src import common
    from src.data.feature_store import resolve_audio_backend_dir

    assert resolve_audio_backend_dir("wav2vec2") == common.FEAT_AUDIO_WAV2VEC2_DIR
    assert resolve_audio_backend_dir("wavlm") == common.FEAT_AUDIO_WAVLM_DIR
    assert resolve_audio_backend_dir("hubert") == common.FEAT_AUDIO_HUBERT_DIR


def test_resolve_audio_backend_dir_unknown_raises_validation_error():
    from src.data.feature_store import (
        FeatureStoreValidationError,
        resolve_audio_backend_dir,
    )

    with pytest.raises(FeatureStoreValidationError, match=r"unknown.*backend"):
        resolve_audio_backend_dir("not-a-backend")


# ---------- shared manifest helpers ----------

_AUDIO_CSV_FIELDS = [
    "sample_id", "source_video_id", "split", "media_type", "source_folder",
    "provider", "voice_id_or_name", "audio_path", "video_path",
    "audio_feature_path", "lip_feature_path",
    "audio_label", "audio_label_binary",
    "video_label", "video_label_binary",
    "pair_label", "pair_label_binary",
]


def _audio_row(
    sample_id: str,
    *,
    split: str = "train",
    provider: str = "original",
    source_folder: str = "real",
    audio_label_binary: str = "0",
    pair_label_binary: str = "0",
    source_video_id: str | None = None,
) -> dict:
    blank = {k: "" for k in _AUDIO_CSV_FIELDS}
    blank.update({
        "sample_id": sample_id,
        "source_video_id": source_video_id or sample_id,
        "split": split,
        "media_type": "audio",
        "source_folder": source_folder,
        "provider": provider,
        "audio_path": f"data/audio_wav/{source_folder}/{sample_id}.wav",
        "video_path": f"data/raw/{source_folder}/{sample_id}.mp4",
        "audio_label": "bonafide" if audio_label_binary == "0" else "spoof",
        "audio_label_binary": audio_label_binary,
        "video_label": "real",
        "video_label_binary": "0",
        "pair_label": "matched_bonafide",
        "pair_label_binary": pair_label_binary,
    })
    return blank


def _write_manifest(path: Path, rows: list[dict], fields: list[str] = _AUDIO_CSV_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_audio_npy(dir_path: Path, sample_id: str, t: int = 199, dim: int = 768,
                     dtype=np.float32) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    arr = np.random.randn(t, dim).astype(dtype)
    out = dir_path / f"{sample_id}.npy"
    np.save(out, arr)
    return out


# ---------- Task 2: AudioFeatureDataset ----------

def test_audio_dataset_filters_split_and_returns_label_and_metadata(tmp_path):
    from src.data.feature_store import AudioFeatureDataset

    manifest = tmp_path / "audio.csv"
    audio_dir = tmp_path / "feat"
    _write_manifest(manifest, [
        _audio_row("a", split="train", audio_label_binary="0"),
        _audio_row("b", split="val",   audio_label_binary="1"),
        _audio_row("c", split="train", audio_label_binary="1"),
    ])
    for sid in ("a", "b", "c"):
        _write_audio_npy(audio_dir, sid)

    ds = AudioFeatureDataset(manifest, split="train", backend="wav2vec2", audio_dir=audio_dir)

    assert len(ds) == 2
    item = ds[0]
    assert item["sample_id"] == "a"
    assert item["split"] == "train"
    assert item["provider"] == "original"
    assert item["source_folder"] == "real"
    assert item["source_video_id"] == "a"
    assert isinstance(item["audio"], torch.Tensor)
    assert item["audio"].dtype == torch.float32
    assert item["audio"].shape == (199, 768)
    assert isinstance(item["label"], torch.Tensor)
    assert item["label"].dtype == torch.long
    assert int(item["label"].item()) == 0
    assert int(ds[1]["label"].item()) == 1


def test_audio_dataset_defaults_to_backend_root_when_audio_dir_omitted(tmp_path, monkeypatch):
    from src import common
    from src.data.feature_store import AudioFeatureDataset

    backend_root = tmp_path / "wav2vec2_root"
    monkeypatch.setattr(common, "FEAT_AUDIO_WAV2VEC2_DIR", backend_root, raising=True)
    import src.data.feature_store as fs
    monkeypatch.setitem(fs.AUDIO_BACKEND_DIRS, "wav2vec2", backend_root)

    manifest = tmp_path / "audio.csv"
    _write_manifest(manifest, [_audio_row("a", split="train")])
    _write_audio_npy(backend_root, "a")

    ds = AudioFeatureDataset(manifest, split="train", backend="wav2vec2")

    assert ds[0]["audio"].shape == (199, 768)


def test_audio_dataset_missing_file_raises_validation_error(tmp_path):
    from src.data.feature_store import AudioFeatureDataset, FeatureStoreValidationError

    manifest = tmp_path / "audio.csv"
    audio_dir = tmp_path / "feat"
    _write_manifest(manifest, [_audio_row("ghost", split="train")])

    ds = AudioFeatureDataset(manifest, split="train", backend="wav2vec2", audio_dir=audio_dir)

    with pytest.raises(FeatureStoreValidationError, match=r"missing audio feature"):
        _ = ds[0]
