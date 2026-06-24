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


# ---------- Task 3: VisualFeatureDataset ----------

def _write_lip_npz(dir_path: Path, source_video_id: str, t: int = 20, dim: int = 84,
                   *, mask: np.ndarray | None = None,
                   feats: np.ndarray | None = None,
                   key_override: str | None = None,
                   omit_mask: bool = False) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    feats = feats if feats is not None else np.random.randn(t, dim).astype(np.float32)
    mask = mask if mask is not None else np.ones(t, dtype=np.float32)
    out = dir_path / f"{source_video_id}.npz"
    if key_override is not None:
        np.savez(out, **{key_override: feats, "mask": mask})
    elif omit_mask:
        np.savez(out, feats=feats)
    else:
        np.savez(out, feats=feats, mask=mask)
    return out


def test_visual_dataset_filters_split_and_returns_pair_label(tmp_path):
    from src.data.feature_store import VisualFeatureDataset

    manifest = tmp_path / "visual.csv"
    lips_dir = tmp_path / "lips"
    _write_manifest(manifest, [
        _audio_row("a", split="train", pair_label_binary="0", source_video_id="vid_a"),
        _audio_row("b", split="val",   pair_label_binary="1", source_video_id="vid_b"),
        _audio_row("c", split="train", pair_label_binary="1", source_video_id="vid_c"),
    ])
    for vid in ("vid_a", "vid_b", "vid_c"):
        _write_lip_npz(lips_dir, vid)

    ds = VisualFeatureDataset(manifest, split="train", lips_dir=lips_dir)

    assert len(ds) == 2
    item = ds[0]
    assert item["sample_id"] == "a"
    assert item["source_video_id"] == "vid_a"
    assert item["lips"].dtype == torch.float32
    assert item["lips"].shape == (20, 84)
    assert item["lips_mask"].shape == (20,)
    assert int(item["label"].item()) == 0
    assert int(ds[1]["label"].item()) == 1


def test_visual_dataset_missing_npz_raises(tmp_path):
    from src.data.feature_store import VisualFeatureDataset, FeatureStoreValidationError

    manifest = tmp_path / "visual.csv"
    lips_dir = tmp_path / "lips"
    _write_manifest(manifest, [_audio_row("a", split="train", source_video_id="vid_a")])

    ds = VisualFeatureDataset(manifest, split="train", lips_dir=lips_dir)
    with pytest.raises(FeatureStoreValidationError, match=r"missing lip feature"):
        _ = ds[0]


def test_visual_dataset_rejects_wrong_npz_key(tmp_path):
    from src.data.feature_store import VisualFeatureDataset, FeatureStoreValidationError

    manifest = tmp_path / "visual.csv"
    lips_dir = tmp_path / "lips"
    _write_manifest(manifest, [_audio_row("a", split="train", source_video_id="vid_a")])
    _write_lip_npz(lips_dir, "vid_a", key_override="wrong_key")

    ds = VisualFeatureDataset(manifest, split="train", lips_dir=lips_dir)
    with pytest.raises(FeatureStoreValidationError, match=r"missing 'feats' key"):
        _ = ds[0]


def test_visual_dataset_rejects_nan_inf(tmp_path):
    from src.data.feature_store import VisualFeatureDataset, FeatureStoreValidationError

    manifest = tmp_path / "visual.csv"
    lips_dir = tmp_path / "lips"
    _write_manifest(manifest, [_audio_row("a", split="train", source_video_id="vid_a")])

    bad = np.random.randn(20, 84).astype(np.float32)
    bad[3, 4] = np.nan
    _write_lip_npz(lips_dir, "vid_a", feats=bad)

    ds = VisualFeatureDataset(manifest, split="train", lips_dir=lips_dir)
    with pytest.raises(FeatureStoreValidationError, match=r"NaN|Inf"):
        _ = ds[0]


def test_visual_dataset_rejects_missing_mask(tmp_path):
    from src.data.feature_store import VisualFeatureDataset, FeatureStoreValidationError

    manifest = tmp_path / "visual.csv"
    lips_dir = tmp_path / "lips"
    _write_manifest(manifest, [_audio_row("a", split="train", source_video_id="vid_a")])
    _write_lip_npz(lips_dir, "vid_a", omit_mask=True)

    ds = VisualFeatureDataset(manifest, split="train", lips_dir=lips_dir)
    with pytest.raises(FeatureStoreValidationError, match=r"missing 'mask' key"):
        _ = ds[0]


def test_visual_dataset_rejects_mask_shape_mismatch(tmp_path):
    from src.data.feature_store import VisualFeatureDataset, FeatureStoreValidationError

    manifest = tmp_path / "visual.csv"
    lips_dir = tmp_path / "lips"
    _write_manifest(manifest, [_audio_row("a", split="train", source_video_id="vid_a")])
    feats = np.random.randn(20, 84).astype(np.float32)
    mask_wrong = np.ones(19, dtype=np.float32)  # length != feats.shape[0]
    _write_lip_npz(lips_dir, "vid_a", feats=feats, mask=mask_wrong)

    ds = VisualFeatureDataset(manifest, split="train", lips_dir=lips_dir)
    with pytest.raises(FeatureStoreValidationError, match=r"mask.*shape"):
        _ = ds[0]


def test_visual_dataset_rejects_mask_not_1d(tmp_path):
    from src.data.feature_store import VisualFeatureDataset, FeatureStoreValidationError

    manifest = tmp_path / "visual.csv"
    lips_dir = tmp_path / "lips"
    _write_manifest(manifest, [_audio_row("a", split="train", source_video_id="vid_a")])
    feats = np.random.randn(20, 84).astype(np.float32)
    mask_2d = np.ones((20, 1), dtype=np.float32)
    _write_lip_npz(lips_dir, "vid_a", feats=feats, mask=mask_2d)

    ds = VisualFeatureDataset(manifest, split="train", lips_dir=lips_dir)
    with pytest.raises(FeatureStoreValidationError, match=r"mask.*1-D|mask.*rank"):
        _ = ds[0]


# ---------- Task 4: FusionFeatureDataset ----------

def test_fusion_dataset_returns_audio_and_lips_for_same_row(tmp_path):
    from src.data.feature_store import FusionFeatureDataset

    manifest = tmp_path / "fusion.csv"
    audio_dir = tmp_path / "audio_feat"
    lips_dir = tmp_path / "lips"
    _write_manifest(manifest, [
        _audio_row("a", split="train", pair_label_binary="0", source_video_id="vid_a"),
        _audio_row("b", split="val",   pair_label_binary="1", source_video_id="vid_b"),
    ])
    for sid in ("a", "b"):
        _write_audio_npy(audio_dir, sid)
    for vid in ("vid_a", "vid_b"):
        _write_lip_npz(lips_dir, vid)

    ds = FusionFeatureDataset(
        manifest, split="train", backend="wav2vec2",
        audio_dir=audio_dir, lips_dir=lips_dir,
    )

    assert len(ds) == 1
    item = ds[0]
    assert item["sample_id"] == "a"
    assert item["source_video_id"] == "vid_a"
    assert item["audio"].shape == (199, 768)
    assert item["lips"].shape == (20, 84)
    assert item["lips_mask"].shape == (20,)
    assert int(item["label"].item()) == 0


def test_fusion_dataset_backend_selection_resolves_default_audio_root(tmp_path, monkeypatch):
    """All three audio backends route to their own root when audio_dir is omitted."""
    from src import common
    import src.data.feature_store as fs
    from src.data.feature_store import FusionFeatureDataset

    manifest = tmp_path / "fusion.csv"
    lips_dir = tmp_path / "lips"
    _write_manifest(manifest, [_audio_row("a", split="train", source_video_id="vid_a")])
    _write_lip_npz(lips_dir, "vid_a")

    roots = {
        "wav2vec2": tmp_path / "w2v",
        "wavlm":    tmp_path / "wlm",
        "hubert":   tmp_path / "hub",
    }
    for backend, root in roots.items():
        _write_audio_npy(root, "a")

    monkeypatch.setattr(common, "FEAT_AUDIO_WAV2VEC2_DIR", roots["wav2vec2"], raising=True)
    monkeypatch.setattr(common, "FEAT_AUDIO_WAVLM_DIR",    roots["wavlm"],    raising=True)
    monkeypatch.setattr(common, "FEAT_AUDIO_HUBERT_DIR",   roots["hubert"],   raising=True)
    monkeypatch.setattr(fs, "AUDIO_BACKEND_DIRS", dict(roots))

    for backend in ("wav2vec2", "wavlm", "hubert"):
        ds = FusionFeatureDataset(manifest, split="train", backend=backend, lips_dir=lips_dir)
        assert ds[0]["audio"].shape == (199, 768)
