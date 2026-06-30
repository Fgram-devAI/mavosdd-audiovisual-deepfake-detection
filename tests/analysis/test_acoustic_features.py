"""Unit tests for src/analysis/acoustic_features.py (core features only)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest


def test_compute_features_returns_full_core_schema(make_tone_wav):
    from src.analysis.acoustic_features import CORE_FEATURE_COLUMNS, compute_features

    feats = compute_features(make_tone_wav(duration_s=1.0))
    assert set(feats.keys()) == set(CORE_FEATURE_COLUMNS)
    for value in feats.values():
        assert isinstance(value, float)
        assert math.isfinite(value), "no NaN/Inf on valid input"


def test_silence_features(make_silent_wav):
    from src.analysis.acoustic_features import compute_features

    feats = compute_features(make_silent_wav(duration_s=1.0))
    assert feats["rms"] == pytest.approx(0.0, abs=1e-6)
    assert feats["peak_amplitude"] == pytest.approx(0.0, abs=1e-6)
    assert feats["silence_ratio"] == pytest.approx(1.0, abs=1e-3)
    assert feats["leading_silence_s"] == pytest.approx(1.0, abs=0.05)
    assert feats["trailing_silence_s"] == pytest.approx(1.0, abs=0.05)
    assert feats["speech_activity_ratio"] == pytest.approx(0.0, abs=1e-3)


def test_tone_spectral_centroid_locates_frequency(make_tone_wav):
    from src.analysis.acoustic_features import compute_features

    feats = compute_features(make_tone_wav(freq_hz=1000.0, duration_s=1.0))
    assert abs(feats["spectral_centroid_mean"] - 1000.0) < 100.0
    assert feats["silence_ratio"] < 0.1
    assert feats["speech_activity_ratio"] > 0.9


def test_half_silence_half_tone_locates_leading_silence(make_half_silence_half_tone_wav):
    from src.analysis.acoustic_features import compute_features

    feats = compute_features(make_half_silence_half_tone_wav(total_s=2.0))
    assert feats["leading_silence_s"] == pytest.approx(1.0, abs=0.1)
    assert feats["trailing_silence_s"] < 0.1


def test_white_noise_has_high_zcr(make_white_noise_wav, make_tone_wav):
    from src.analysis.acoustic_features import compute_features

    noise = compute_features(make_white_noise_wav(duration_s=1.0))
    tone = compute_features(make_tone_wav(freq_hz=200.0, duration_s=1.0))
    assert noise["zcr_mean"] > tone["zcr_mean"]


def test_unreadable_audio_raises(make_corrupt_wav):
    from src.analysis.acoustic_features import AcousticFeatureError, compute_features

    with pytest.raises(AcousticFeatureError):
        compute_features(make_corrupt_wav())


def test_mp3_path_works(make_tiny_mp3):
    from src.analysis.acoustic_features import CORE_FEATURE_COLUMNS, compute_features

    feats = compute_features(make_tiny_mp3(duration_s=0.5))
    assert set(feats.keys()) == set(CORE_FEATURE_COLUMNS)


def test_feature_columns_helper_matches_core():
    from src.analysis.acoustic_features import CORE_FEATURE_COLUMNS, feature_columns

    assert feature_columns(with_f0=False) == CORE_FEATURE_COLUMNS
