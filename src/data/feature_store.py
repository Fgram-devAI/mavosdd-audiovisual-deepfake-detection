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
