"""Tests for src/features/audio_io.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def _write_sine_wav(path: Path, seconds: float, sr: int = 16000, freq: float = 440.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sr * seconds)) / sr
    wave = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), wave, sr)


def test_load_audio_window_returns_exact_length_when_input_is_long_enough(tmp_path):
    from src.features.audio_io import load_audio_window

    src = tmp_path / "long.wav"
    _write_sine_wav(src, seconds=6.0)

    out = load_audio_window(src, sr=16000, seconds=4.0)

    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.shape == (64000,)


def test_load_audio_window_center_crops_longer_audio(tmp_path):
    from src.features.audio_io import load_audio_window

    src = tmp_path / "long.wav"
    # Build a known waveform: ramp from 0..N-1 so we can verify the slice indices.
    sr = 16000
    n = int(sr * 6.0)
    wave = np.arange(n, dtype=np.float32) / n
    sf.write(str(src), wave, sr)

    out = load_audio_window(src, sr=sr, seconds=4.0)

    expected_start = (n - 64000) // 2
    assert np.allclose(out[0], wave[expected_start], atol=1e-4)
    assert np.allclose(out[-1], wave[expected_start + 64000 - 1], atol=1e-4)


def test_load_audio_window_right_pads_with_zeros_when_too_short(tmp_path):
    from src.features.audio_io import load_audio_window

    src = tmp_path / "short.wav"
    _write_sine_wav(src, seconds=1.0)

    out = load_audio_window(src, sr=16000, seconds=4.0)

    assert out.shape == (64000,)
    assert out.dtype == np.float32
    # Last 3 seconds (48000 samples) must be exactly zero padding.
    assert np.array_equal(out[16000:], np.zeros(48000, dtype=np.float32))


def test_load_audio_window_is_deterministic(tmp_path):
    from src.features.audio_io import load_audio_window

    src = tmp_path / "again.wav"
    _write_sine_wav(src, seconds=5.0)

    a = load_audio_window(src)
    b = load_audio_window(src)

    assert np.array_equal(a, b)


def test_load_audio_window_resamples_to_target_sr(tmp_path):
    from src.features.audio_io import load_audio_window

    src = tmp_path / "44k.wav"
    _write_sine_wav(src, seconds=4.0, sr=44100)

    out = load_audio_window(src, sr=16000, seconds=4.0)

    assert out.shape == (64000,)
    assert out.dtype == np.float32
