"""Shared synthetic-audio fixtures for the acoustic-probe test suite."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(scope="session")
def ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH; MP3 fixture tests require it")


def _write_wav(path: Path, wave: np.ndarray, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wave.astype(np.float32), sr)


@pytest.fixture
def make_silent_wav(tmp_path: Path) -> Callable[..., Path]:
    def _make(name: str = "silent.wav", duration_s: float = 1.0, sr: int = 16000) -> Path:
        out = tmp_path / name
        wave = np.zeros(int(sr * duration_s), dtype=np.float32)
        _write_wav(out, wave, sr)
        return out
    return _make


@pytest.fixture
def make_tone_wav(tmp_path: Path) -> Callable[..., Path]:
    def _make(
        name: str = "tone.wav",
        freq_hz: float = 1000.0,
        duration_s: float = 1.0,
        amplitude: float = 0.5,
        sr: int = 16000,
    ) -> Path:
        out = tmp_path / name
        t = np.arange(int(sr * duration_s)) / sr
        wave = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
        _write_wav(out, wave, sr)
        return out
    return _make


@pytest.fixture
def make_white_noise_wav(tmp_path: Path) -> Callable[..., Path]:
    def _make(
        name: str = "noise.wav",
        duration_s: float = 1.0,
        amplitude: float = 0.05,
        seed: int = 42,
        sr: int = 16000,
    ) -> Path:
        out = tmp_path / name
        rng = np.random.default_rng(seed)
        wave = (amplitude * rng.standard_normal(int(sr * duration_s))).astype(np.float32)
        _write_wav(out, wave, sr)
        return out
    return _make


@pytest.fixture
def make_half_silence_half_tone_wav(tmp_path: Path) -> Callable[..., Path]:
    def _make(
        name: str = "half_silence_half_tone.wav",
        total_s: float = 2.0,
        freq_hz: float = 1000.0,
        amplitude: float = 0.5,
        sr: int = 16000,
    ) -> Path:
        out = tmp_path / name
        n_total = int(sr * total_s)
        n_half = n_total // 2
        t = np.arange(n_half) / sr
        tone = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
        wave = np.concatenate([np.zeros(n_half, dtype=np.float32), tone])
        _write_wav(out, wave, sr)
        return out
    return _make


@pytest.fixture
def make_corrupt_wav(tmp_path: Path) -> Callable[..., Path]:
    def _make(name: str = "corrupt.wav") -> Path:
        out = tmp_path / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"NOT_A_WAV_FILE_AT_ALL" * 4)
        return out
    return _make


@pytest.fixture
def make_tiny_mp3(ffmpeg_available, tmp_path: Path) -> Callable[..., Path]:
    def _make(
        name: str = "tone.mp3",
        freq_hz: float = 1000.0,
        duration_s: float = 1.0,
    ) -> Path:
        out = tmp_path / name
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi",
                "-i", f"sine=frequency={freq_hz}:duration={duration_s}",
                "-ar", "16000", "-ac", "1",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        return out
    return _make
