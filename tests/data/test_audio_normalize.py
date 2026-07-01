"""Unit tests for src.data.audio_normalize pure transforms."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from src.data import audio_normalize as an


def test_resample_mono_16k_stereo_44100_to_mono_16000(make_stereo_wav):
    # Read a 44100 stereo fixture via soundfile to skip librosa's own mono/resample.
    p = make_stereo_wav()  # 16000 Hz stereo, 1 s
    wave, sr = sf.read(str(p), always_2d=True)
    assert wave.ndim == 2 and wave.shape[1] == 2 and sr == 16000
    out = an.resample_mono_16k(wave, sr=16000, target_sr=16000)
    assert out.ndim == 1
    assert out.dtype == np.float32
    assert abs(out.shape[0] - 16000) <= 1


def test_resample_mono_16k_44100_to_16000(make_44100_wav):
    p = make_44100_wav()
    wave, sr = sf.read(str(p), always_2d=False)
    assert sr == 44100
    out = an.resample_mono_16k(wave, sr=44100, target_sr=16000)
    assert out.ndim == 1
    assert out.dtype == np.float32
    # 1 s of 44100 → ~16000 samples at 16 kHz.
    assert 15900 <= out.shape[0] <= 16100


def test_resample_mono_16k_is_deterministic(make_44100_wav):
    p = make_44100_wav()
    wave, sr = sf.read(str(p))
    a = an.resample_mono_16k(wave, sr=sr, target_sr=16000)
    b = an.resample_mono_16k(wave, sr=sr, target_sr=16000)
    np.testing.assert_array_equal(a, b)


def test_resample_mono_16k_raises_on_empty():
    with pytest.raises(an.ResampleError):
        an.resample_mono_16k(np.zeros(0, dtype=np.float32), sr=16000)
