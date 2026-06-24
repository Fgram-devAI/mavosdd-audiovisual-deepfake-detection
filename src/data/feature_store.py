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
