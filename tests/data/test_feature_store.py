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


# ---------- Task 5: validate_feature_store + CLI ----------

def test_validate_audio_view_ok(tmp_path):
    from src.data.feature_store import validate_feature_store

    manifest = tmp_path / "audio.csv"
    audio_dir = tmp_path / "feat"
    _write_manifest(manifest, [
        _audio_row("a", split="train", audio_label_binary="0"),
        _audio_row("b", split="val",   audio_label_binary="1"),
    ])
    _write_audio_npy(audio_dir, "a")
    _write_audio_npy(audio_dir, "b")

    report = validate_feature_store(
        "audio", manifest, backend="wav2vec2", audio_dir=audio_dir,
    )
    assert report.view == "audio"
    assert report.backend == "wav2vec2"
    assert report.manifest_rows == 2
    assert report.split_counts == {"train": 1, "val": 1}
    assert report.label_counts == {"0": 1, "1": 1}
    assert report.missing == []
    assert report.bad_shape == []
    assert report.path_mismatches == []


def test_validate_reports_path_mismatch_as_warning_not_failure(tmp_path):
    """audio_feature_path that disagrees with the reconstructed path is warned, not failed."""
    from src.data import feature_store as fs

    manifest = tmp_path / "audio.csv"
    audio_dir = tmp_path / "feat"
    row = _audio_row("a", split="train")
    row["audio_feature_path"] = "data/features/audio/somewhere_else.npy"  # ≠ reconstructed
    _write_manifest(manifest, [row])
    _write_audio_npy(audio_dir, "a")

    report = fs.validate_feature_store("audio", manifest, backend="wav2vec2", audio_dir=audio_dir)
    assert report.missing == []
    assert report.bad_shape == []
    assert len(report.path_mismatches) == 1
    assert "a" in report.path_mismatches[0]
    assert "somewhere_else" in report.path_mismatches[0]

    # CLI exit code must NOT trip on path_mismatches alone.
    rc = fs.main([
        "--validate", "--view", "audio", "--backend", "wav2vec2",
        "--manifest", str(manifest), "--audio-dir", str(audio_dir),
    ])
    assert rc == 0


def test_validate_audio_view_reports_missing_file(tmp_path):
    from src.data.feature_store import validate_feature_store

    manifest = tmp_path / "audio.csv"
    audio_dir = tmp_path / "feat"
    _write_manifest(manifest, [_audio_row("ghost", split="train")])

    report = validate_feature_store("audio", manifest, backend="wav2vec2", audio_dir=audio_dir)
    assert len(report.missing) == 1
    assert "ghost" in report.missing[0]


def test_validate_audio_view_reports_bad_shape(tmp_path):
    from src.data.feature_store import validate_feature_store

    manifest = tmp_path / "audio.csv"
    audio_dir = tmp_path / "feat"
    _write_manifest(manifest, [_audio_row("a", split="train")])
    audio_dir.mkdir()
    np.save(audio_dir / "a.npy", np.zeros((199, 512), dtype=np.float32))

    report = validate_feature_store("audio", manifest, backend="wav2vec2", audio_dir=audio_dir)
    assert report.missing == []
    assert len(report.bad_shape) == 1
    assert "a" in report.bad_shape[0]


def test_validate_visual_view_reports_missing_and_bad_key(tmp_path):
    from src.data.feature_store import validate_feature_store

    manifest = tmp_path / "visual.csv"
    lips_dir = tmp_path / "lips"
    _write_manifest(manifest, [
        _audio_row("a", split="train", source_video_id="vid_a"),
        _audio_row("b", split="train", source_video_id="vid_b"),
    ])
    _write_lip_npz(lips_dir, "vid_b", key_override="wrong_key")

    report = validate_feature_store("visual", manifest, lips_dir=lips_dir)
    assert len(report.missing) == 1
    assert "vid_a" in report.missing[0]
    assert len(report.bad_shape) == 1
    assert "vid_b" in report.bad_shape[0]


def test_validate_fusion_view_requires_backend(tmp_path):
    from src.data.feature_store import validate_feature_store, FeatureStoreValidationError

    manifest = tmp_path / "fusion.csv"
    _write_manifest(manifest, [_audio_row("a", split="train")])

    with pytest.raises(FeatureStoreValidationError, match=r"backend.*required"):
        validate_feature_store("fusion", manifest)


def test_validate_fusion_view_ok(tmp_path):
    from src.data.feature_store import validate_feature_store

    manifest = tmp_path / "fusion.csv"
    audio_dir = tmp_path / "audio_feat"
    lips_dir = tmp_path / "lips"
    _write_manifest(manifest, [_audio_row("a", split="train", source_video_id="vid_a")])
    _write_audio_npy(audio_dir, "a")
    _write_lip_npz(lips_dir, "vid_a")

    report = validate_feature_store(
        "fusion", manifest, backend="wav2vec2",
        audio_dir=audio_dir, lips_dir=lips_dir,
    )
    assert report.missing == []
    assert report.bad_shape == []


def test_cli_validate_audio_returns_nonzero_on_missing(tmp_path):
    from src.data import feature_store as fs

    manifest = tmp_path / "audio.csv"
    audio_dir = tmp_path / "feat"
    _write_manifest(manifest, [_audio_row("ghost", split="train")])

    rc = fs.main([
        "--validate", "--view", "audio", "--backend", "wav2vec2",
        "--manifest", str(manifest), "--audio-dir", str(audio_dir),
    ])
    assert rc == 1


def test_cli_validate_audio_returns_zero_on_ok(tmp_path):
    from src.data import feature_store as fs

    manifest = tmp_path / "audio.csv"
    audio_dir = tmp_path / "feat"
    _write_manifest(manifest, [_audio_row("a", split="train")])
    _write_audio_npy(audio_dir, "a")

    rc = fs.main([
        "--validate", "--view", "audio", "--backend", "wav2vec2",
        "--manifest", str(manifest), "--audio-dir", str(audio_dir),
    ])
    assert rc == 0


# ---------- Task 6: normalization ----------

def test_fit_normalization_stats_audio_only(tmp_path):
    from src.data.feature_store import (
        AudioFeatureDataset, fit_normalization_stats, NormalizationStats,
    )

    manifest = tmp_path / "audio.csv"
    audio_dir = tmp_path / "feat"
    _write_manifest(manifest, [
        _audio_row("a", split="train"),
        _audio_row("b", split="train"),
        _audio_row("c", split="val"),  # must not affect fit
    ])
    np.random.seed(0)
    _write_audio_npy(audio_dir, "a")
    _write_audio_npy(audio_dir, "b")
    _write_audio_npy(audio_dir, "c")

    train_ds = AudioFeatureDataset(manifest, split="train", backend="wav2vec2", audio_dir=audio_dir)
    stats = fit_normalization_stats(train_ds, modalities=("audio",))

    assert isinstance(stats, NormalizationStats)
    assert stats.audio_mean.shape == (768,)
    assert stats.audio_std.shape == (768,)
    assert stats.lips_mean is None
    assert stats.lips_std is None


def test_fit_normalization_does_not_read_val_or_test_rows(tmp_path):
    """Constructing a train-split dataset must skip val/test files entirely."""
    from src.data.feature_store import AudioFeatureDataset, fit_normalization_stats

    manifest = tmp_path / "audio.csv"
    audio_dir = tmp_path / "feat"
    _write_manifest(manifest, [
        _audio_row("a", split="train"),
        _audio_row("b", split="val"),
        _audio_row("c", split="test"),
    ])
    _write_audio_npy(audio_dir, "a")
    # b.npy and c.npy intentionally NOT written; if fit reads them, np.load will fail.

    train_ds = AudioFeatureDataset(manifest, split="train", backend="wav2vec2", audio_dir=audio_dir)
    stats = fit_normalization_stats(train_ds, modalities=("audio",))
    assert stats.audio_mean is not None


def test_dataset_applies_normalization_when_provided(tmp_path):
    from src.data.feature_store import AudioFeatureDataset, NormalizationStats

    manifest = tmp_path / "audio.csv"
    audio_dir = tmp_path / "feat"
    _write_manifest(manifest, [_audio_row("a", split="train")])
    audio_dir.mkdir(parents=True, exist_ok=True)
    np.save(audio_dir / "a.npy", np.ones((199, 768), dtype=np.float32) * 5.0)

    stats = NormalizationStats(
        audio_mean=np.full(768, 5.0, dtype=np.float32),
        audio_std=np.full(768, 1.0, dtype=np.float32),
        lips_mean=None, lips_std=None,
    )

    ds = AudioFeatureDataset(
        manifest, split="train", backend="wav2vec2",
        audio_dir=audio_dir, normalization=stats,
    )
    audio = ds[0]["audio"]
    assert torch.allclose(audio, torch.zeros_like(audio), atol=1e-6)


def test_fit_normalization_refuses_non_train_dataset(tmp_path):
    """Calling fit on a val dataset must raise — no silent val/test leakage."""
    from src.data.feature_store import (
        AudioFeatureDataset, fit_normalization_stats, FeatureStoreValidationError,
    )

    manifest = tmp_path / "audio.csv"
    audio_dir = tmp_path / "feat"
    _write_manifest(manifest, [_audio_row("a", split="val")])
    _write_audio_npy(audio_dir, "a")

    val_ds = AudioFeatureDataset(manifest, split="val", backend="wav2vec2", audio_dir=audio_dir)
    assert val_ds.split == "val"

    with pytest.raises(FeatureStoreValidationError, match=r"train.*split|split.*train"):
        fit_normalization_stats(val_ds, modalities=("audio",))


def test_fit_normalization_refuses_dataset_without_split_attr(tmp_path):
    from src.data.feature_store import fit_normalization_stats, FeatureStoreValidationError

    class Bogus:
        def __len__(self): return 0
        def __getitem__(self, i): raise IndexError(i)

    with pytest.raises(FeatureStoreValidationError, match=r"split"):
        fit_normalization_stats(Bogus(), modalities=("audio",))


def test_fit_normalization_lips_only(tmp_path):
    from src.data.feature_store import (
        VisualFeatureDataset, fit_normalization_stats,
    )

    manifest = tmp_path / "visual.csv"
    lips_dir = tmp_path / "lips"
    _write_manifest(manifest, [
        _audio_row("a", split="train", source_video_id="vid_a"),
        _audio_row("b", split="train", source_video_id="vid_b"),
    ])
    _write_lip_npz(lips_dir, "vid_a")
    _write_lip_npz(lips_dir, "vid_b")

    train_ds = VisualFeatureDataset(manifest, split="train", lips_dir=lips_dir)
    stats = fit_normalization_stats(train_ds, modalities=("lips",))

    assert stats.audio_mean is None
    assert stats.audio_std is None
    assert stats.lips_mean.shape == (84,)
    assert stats.lips_std.shape == (84,)


# ---------- Task 7: collate + make_dataloader ----------

def test_feature_collate_stacks_audio_lips_label_and_keeps_metadata(tmp_path):
    from src.data.feature_store import FusionFeatureDataset, feature_collate

    manifest = tmp_path / "fusion.csv"
    audio_dir = tmp_path / "audio_feat"
    lips_dir = tmp_path / "lips"
    _write_manifest(manifest, [
        _audio_row("a", split="train", source_video_id="vid_a"),
        _audio_row("b", split="train", source_video_id="vid_b"),
    ])
    for sid in ("a", "b"):
        _write_audio_npy(audio_dir, sid)
    for vid in ("vid_a", "vid_b"):
        _write_lip_npz(lips_dir, vid)

    ds = FusionFeatureDataset(
        manifest, split="train", backend="wav2vec2",
        audio_dir=audio_dir, lips_dir=lips_dir,
    )
    batch = feature_collate([ds[0], ds[1]])

    assert batch["audio"].shape == (2, 199, 768)
    assert batch["lips"].shape == (2, 20, 84)
    assert batch["lips_mask"].shape == (2, 20)
    assert batch["label"].shape == (2,)
    assert batch["label"].dtype == torch.long
    assert isinstance(batch["metadata"], list)
    assert batch["metadata"][0]["sample_id"] == "a"
    assert batch["metadata"][1]["sample_id"] == "b"


def test_feature_collate_raises_on_shape_mismatch(tmp_path):
    from src.data.feature_store import feature_collate, FeatureStoreValidationError

    item_a = {
        "audio": torch.zeros(199, 768),
        "label": torch.tensor(0, dtype=torch.long),
        "sample_id": "a", "source_video_id": "a", "split": "train",
        "provider": "x", "source_folder": "y",
    }
    item_b = {
        "audio": torch.zeros(200, 768),  # mismatched time
        "label": torch.tensor(1, dtype=torch.long),
        "sample_id": "b", "source_video_id": "b", "split": "train",
        "provider": "x", "source_folder": "y",
    }
    with pytest.raises(FeatureStoreValidationError, match=r"audio.*shape mismatch"):
        feature_collate([item_a, item_b])


def test_make_dataloader_iterates_one_batch(tmp_path):
    from src.data.feature_store import AudioFeatureDataset, make_dataloader

    manifest = tmp_path / "audio.csv"
    audio_dir = tmp_path / "feat"
    _write_manifest(manifest, [_audio_row("a", split="train"), _audio_row("b", split="train")])
    _write_audio_npy(audio_dir, "a")
    _write_audio_npy(audio_dir, "b")

    ds = AudioFeatureDataset(manifest, split="train", backend="wav2vec2", audio_dir=audio_dir)
    loader = make_dataloader(ds, batch_size=2, shuffle=False, num_workers=0)

    batches = list(loader)
    assert len(batches) == 1
    assert batches[0]["audio"].shape == (2, 199, 768)
    assert batches[0]["label"].shape == (2,)
    assert len(batches[0]["metadata"]) == 2


# ---------- Task 8: leakage + backend coverage ----------

def test_datasets_never_open_raw_video_or_wav(monkeypatch, tmp_path):
    """If dataset code touches a raw media file, this test fails loudly."""
    import builtins
    from src.data.feature_store import (
        AudioFeatureDataset, VisualFeatureDataset, FusionFeatureDataset,
    )

    manifest = tmp_path / "fusion.csv"
    audio_dir = tmp_path / "audio_feat"
    lips_dir = tmp_path / "lips"
    _write_manifest(manifest, [_audio_row("a", split="train", source_video_id="vid_a")])
    _write_audio_npy(audio_dir, "a")
    _write_lip_npz(lips_dir, "vid_a")

    forbidden = (".mp4", ".wav", ".mp3", ".m4a", ".mov", ".webm")
    real_open = builtins.open

    def guarded_open(path, *args, **kwargs):
        s = str(path).lower()
        if any(s.endswith(ext) for ext in forbidden):
            raise AssertionError(f"dataset code opened raw media file: {path}")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)

    for ds in (
        AudioFeatureDataset(manifest, split="train", backend="wav2vec2", audio_dir=audio_dir),
        VisualFeatureDataset(manifest, split="train", lips_dir=lips_dir),
        FusionFeatureDataset(manifest, split="train", backend="wav2vec2",
                             audio_dir=audio_dir, lips_dir=lips_dir),
    ):
        _ = ds[0]


def test_validate_each_audio_backend(tmp_path):
    """All three audio backends complete a clean validate with no errors."""
    from src.data.feature_store import validate_feature_store

    manifest = tmp_path / "audio.csv"
    _write_manifest(manifest, [_audio_row("a", split="train")])

    for backend in ("wav2vec2", "wavlm", "hubert"):
        audio_dir = tmp_path / f"audio_{backend}"
        _write_audio_npy(audio_dir, "a")
        report = validate_feature_store(
            "audio", manifest, backend=backend, audio_dir=audio_dir,
        )
        assert report.missing == []
        assert report.bad_shape == []


def test_validate_fusion_each_backend(tmp_path):
    """Fusion validation works against each audio backend independently."""
    from src.data.feature_store import validate_feature_store

    manifest = tmp_path / "fusion.csv"
    lips_dir = tmp_path / "lips"
    _write_manifest(manifest, [_audio_row("a", split="train", source_video_id="vid_a")])
    _write_lip_npz(lips_dir, "vid_a")

    for backend in ("wav2vec2", "wavlm", "hubert"):
        audio_dir = tmp_path / f"audio_{backend}"
        _write_audio_npy(audio_dir, "a")
        report = validate_feature_store(
            "fusion", manifest, backend=backend,
            audio_dir=audio_dir, lips_dir=lips_dir,
        )
        assert report.missing == []
        assert report.bad_shape == []
