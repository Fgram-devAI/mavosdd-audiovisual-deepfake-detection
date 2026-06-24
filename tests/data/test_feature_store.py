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
