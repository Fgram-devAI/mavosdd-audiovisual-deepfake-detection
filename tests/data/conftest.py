"""Synthetic WAV fixtures for the audio-normalization test suite.

Scoped under tests/data/. Does not override anything from tests/conftest.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import soundfile as sf


def _write_wav(path: Path, wave: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wave.astype(np.float32), sr)


@pytest.fixture
def make_silent_wav_norm(tmp_path: Path) -> Callable[..., Path]:
    def _make(name: str = "silent.wav", duration_s: float = 1.0, sr: int = 16000) -> Path:
        out = tmp_path / name
        wave = np.zeros(int(sr * duration_s), dtype=np.float32)
        _write_wav(out, wave, sr)
        return out
    return _make


@pytest.fixture
def make_tone_wav_norm(tmp_path: Path) -> Callable[..., Path]:
    def _make(
        name: str = "tone.wav",
        freq_hz: float = 1000.0,
        duration_s: float = 1.0,
        amplitude: float = 0.3,
        sr: int = 16000,
    ) -> Path:
        out = tmp_path / name
        t = np.arange(int(sr * duration_s)) / sr
        wave = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
        _write_wav(out, wave, sr)
        return out
    return _make


@pytest.fixture
def make_hf_tone_wav(tmp_path: Path) -> Callable[..., Path]:
    def _make(
        name: str = "hf_tone.wav",
        freq_hz: float = 7500.0,
        duration_s: float = 1.0,
        amplitude: float = 0.3,
        sr: int = 16000,
    ) -> Path:
        out = tmp_path / name
        t = np.arange(int(sr * duration_s)) / sr
        wave = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
        _write_wav(out, wave, sr)
        return out
    return _make


@pytest.fixture
def make_padded_speechlike_wav(tmp_path: Path) -> Callable[..., Path]:
    def _make(name: str = "padded.wav", sr: int = 16000) -> Path:
        out = tmp_path / name
        pad = np.zeros(int(sr * 0.5), dtype=np.float32)
        t = np.arange(int(sr * 1.0)) / sr
        # 1 s of 1 kHz carrier amplitude-modulated by 4 Hz — mimics voiced energy.
        carrier = np.sin(2 * np.pi * 1000.0 * t)
        envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
        body = (0.4 * carrier * envelope).astype(np.float32)
        wave = np.concatenate([pad, body, pad])
        _write_wav(out, wave, sr)
        return out
    return _make


@pytest.fixture
def make_hot_wav(tmp_path: Path) -> Callable[..., Path]:
    def _make(
        name: str = "hot.wav",
        freq_hz: float = 1000.0,
        amplitude: float = 0.95,
        duration_s: float = 1.0,
        sr: int = 16000,
    ) -> Path:
        out = tmp_path / name
        t = np.arange(int(sr * duration_s)) / sr
        wave = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
        _write_wav(out, wave, sr)
        return out
    return _make


@pytest.fixture
def make_stereo_wav(tmp_path: Path) -> Callable[..., Path]:
    def _make(name: str = "stereo.wav", sr: int = 16000) -> Path:
        out = tmp_path / name
        t = np.arange(sr) / sr  # 1 second
        left = 0.3 * np.sin(2 * np.pi * 1000.0 * t)
        right = 0.3 * np.sin(2 * np.pi * 500.0 * t)
        wave = np.stack([left, right], axis=1).astype(np.float32)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), wave, sr)
        return out
    return _make


@pytest.fixture
def make_44100_wav(tmp_path: Path) -> Callable[..., Path]:
    def _make(name: str = "44k.wav") -> Path:
        out = tmp_path / name
        sr = 44100
        t = np.arange(sr) / sr
        wave = (0.3 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
        _write_wav(out, wave, sr)
        return out
    return _make


@pytest.fixture
def make_broken_wav(tmp_path: Path) -> Callable[..., Path]:
    def _make(name: str = "broken.wav") -> Path:
        out = tmp_path / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"NOT A WAV" + b"\x00" * 32)
        return out
    return _make


@pytest.fixture
def make_short_wav(tmp_path: Path) -> Callable[..., Path]:
    def _make(name: str = "short.wav", duration_s: float = 0.1, sr: int = 16000) -> Path:
        out = tmp_path / name
        t = np.arange(int(sr * duration_s)) / sr
        wave = (0.3 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
        _write_wav(out, wave, sr)
        return out
    return _make
