"""Deterministic audio loading for frozen-encoder feature extraction.

The single public entry point, ``load_audio_window``, returns a fixed-length
mono float32 numpy array at the requested sample rate. Longer inputs are
center-cropped; shorter inputs are right-padded with zeros. Behavior is
deterministic so cached .npy features remain reproducible.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def load_audio_window(
    path: str | Path,
    sr: int = 16000,
    seconds: float = 4.0,
) -> np.ndarray:
    """Load ``path`` as mono ``sr`` Hz audio, return a length-``int(sr*seconds)`` float32 window."""
    n_samples = int(sr * seconds)
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(str(audio_path))
    wave, _ = librosa.load(str(path), sr=sr, mono=True)
    wave = np.asarray(wave, dtype=np.float32)

    if wave.shape[0] >= n_samples:
        start = (wave.shape[0] - n_samples) // 2
        wave = wave[start : start + n_samples]
    else:
        pad = n_samples - wave.shape[0]
        wave = np.pad(wave, (0, pad))

    return wave.astype(np.float32, copy=False)
