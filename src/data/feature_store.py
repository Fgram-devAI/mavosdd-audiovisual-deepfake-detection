"""Split-safe PyTorch datasets and validation over cached feature stores.

This module is the single bridge between derived manifests (under
``data/derived/``) and training code. It never reads raw audio or video.

Public API:
    FeatureStoreValidationError
    AudioFeatureDataset
    VisualFeatureDataset
    FusionFeatureDataset
    fit_normalization_stats
    feature_collate
    make_dataloader
    validate_feature_store
"""
from __future__ import annotations

from pathlib import Path

from src import common


class FeatureStoreValidationError(Exception):
    """Raised when a manifest, cached feature file, or shape contract is invalid."""


AUDIO_FEATURE_DIM: int = 768
SUPPORTED_BACKENDS: tuple[str, ...] = ("wav2vec2", "wavlm", "hubert")

AUDIO_BACKEND_DIRS: dict[str, Path] = {
    "wav2vec2": common.FEAT_AUDIO_WAV2VEC2_DIR,
    "wavlm": common.FEAT_AUDIO_WAVLM_DIR,
    "hubert": common.FEAT_AUDIO_HUBERT_DIR,
}


def resolve_audio_backend_dir(backend: str) -> Path:
    """Return the default audio feature root for a backend short name."""
    try:
        return AUDIO_BACKEND_DIRS[backend]
    except KeyError as exc:
        raise FeatureStoreValidationError(
            f"unknown audio backend: {backend!r}. "
            f"Supported: {SUPPORTED_BACKENDS}"
        ) from exc


import csv as _csv

import numpy as np
import torch
from torch.utils.data import Dataset


_METADATA_KEYS: tuple[str, ...] = (
    "sample_id", "source_video_id", "split", "provider", "source_folder",
)


def _read_manifest_rows(manifest_path: Path | str) -> list[dict]:
    with Path(manifest_path).open(newline="") as f:
        return list(_csv.DictReader(f))


def _filter_split(rows: list[dict], split: str) -> list[dict]:
    return [r for r in rows if r.get("split") == split]


def _row_metadata(row: dict) -> dict:
    return {k: row.get(k, "") for k in _METADATA_KEYS}


def _label_long(row: dict, column: str) -> torch.Tensor:
    raw = row.get(column, "")
    if raw == "" or raw is None:
        raise FeatureStoreValidationError(
            f"manifest row {row.get('sample_id', '?')!r} missing label column {column!r}"
        )
    return torch.tensor(int(raw), dtype=torch.long)


def _load_audio_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FeatureStoreValidationError(f"missing audio feature: {path}")
    arr = np.load(path)
    if arr.ndim != 2:
        raise FeatureStoreValidationError(
            f"audio feature {path} must be rank-2, got shape {arr.shape}"
        )
    if arr.shape[1] != AUDIO_FEATURE_DIM:
        raise FeatureStoreValidationError(
            f"audio feature {path} feature dim {arr.shape[1]} != {AUDIO_FEATURE_DIM}"
        )
    if arr.shape[0] <= 0:
        raise FeatureStoreValidationError(
            f"audio feature {path} time dim must be positive, got {arr.shape[0]}"
        )
    return arr.astype(np.float32, copy=False)


class AudioFeatureDataset(Dataset):
    """Audio-only dataset over a backend-specific feature store."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        backend: str,
        audio_dir: Path | None = None,
    ) -> None:
        self._rows = _filter_split(_read_manifest_rows(manifest_path), split)
        self._audio_dir = Path(audio_dir) if audio_dir is not None else resolve_audio_backend_dir(backend)
        self.backend = backend
        self.split = split

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        audio_path = self._audio_dir / f"{row['sample_id']}.npy"
        arr = _load_audio_array(audio_path)
        item = {
            "audio": torch.from_numpy(arr),
            "label": _label_long(row, "audio_label_binary"),
        }
        item.update(_row_metadata(row))
        return item


def _load_lip_array(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FeatureStoreValidationError(f"missing lip feature: {path}")
    with np.load(path) as npz:
        if "feats" not in npz.files:
            raise FeatureStoreValidationError(
                f"lip feature {path} missing 'feats' key (found {list(npz.files)!r})"
            )
        if "mask" not in npz.files:
            raise FeatureStoreValidationError(
                f"lip feature {path} missing 'mask' key (found {list(npz.files)!r})"
            )
        feats = np.asarray(npz["feats"])
        mask = np.asarray(npz["mask"])
    if not np.issubdtype(feats.dtype, np.floating) and not np.issubdtype(feats.dtype, np.integer):
        raise FeatureStoreValidationError(
            f"lip feature {path} 'feats' must be numeric, got dtype {feats.dtype}"
        )
    if feats.ndim != 2:
        raise FeatureStoreValidationError(
            f"lip feature {path} 'feats' must be rank-2, got shape {feats.shape}"
        )
    if feats.shape[0] <= 0 or feats.shape[1] <= 0:
        raise FeatureStoreValidationError(
            f"lip feature {path} 'feats' has non-positive dim: shape {feats.shape}"
        )
    if mask.ndim != 1:
        raise FeatureStoreValidationError(
            f"lip feature {path} 'mask' must be 1-D (rank-1), got shape {mask.shape}"
        )
    if mask.shape[0] != feats.shape[0]:
        raise FeatureStoreValidationError(
            f"lip feature {path} 'mask' shape {mask.shape} does not match "
            f"feats time dim {feats.shape[0]}"
        )
    if not np.isfinite(feats).all():
        raise FeatureStoreValidationError(
            f"lip feature {path} 'feats' contains NaN or Inf"
        )
    return feats.astype(np.float32, copy=False), mask.astype(np.float32, copy=False)


class VisualFeatureDataset(Dataset):
    """Lip-only dataset over data/features/lips/{source_video_id}.npz."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        lips_dir: Path | None = None,
    ) -> None:
        self._rows = _filter_split(_read_manifest_rows(manifest_path), split)
        self._lips_dir = Path(lips_dir) if lips_dir is not None else common.FEAT_LIPS_DIR
        self.split = split

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        lip_path = self._lips_dir / f"{row['source_video_id']}.npz"
        feats, mask = _load_lip_array(lip_path)
        item = {
            "lips": torch.from_numpy(feats),
            "lips_mask": torch.from_numpy(mask),
            "label": _label_long(row, "pair_label_binary"),
        }
        item.update(_row_metadata(row))
        return item
