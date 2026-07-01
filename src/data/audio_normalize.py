"""Deterministic audio-normalization primitives.

Pure numpy transforms with no I/O and no argparse. Each function raises a
typed subclass of ``AudioNormalizeError`` on failure so the CLI orchestrator
can catch per-stage and log a stable ``stage`` string.
"""
from __future__ import annotations

import numpy as np


class AudioNormalizeError(Exception):
    """Base for all audio-normalization transform failures."""


class DecodeError(AudioNormalizeError):
    pass


class ResampleError(AudioNormalizeError):
    pass


class TrimError(AudioNormalizeError):
    pass


class LowpassError(AudioNormalizeError):
    pass


class LoudnessError(AudioNormalizeError):
    pass


class PeakSafetyError(AudioNormalizeError):
    pass


class WriteError(AudioNormalizeError):
    pass


def resample_mono_16k(
    wave: np.ndarray,
    sr: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """Downmix to mono and resample to ``target_sr`` (default 16 kHz).

    Deterministic: uses ``librosa.resample(..., res_type="soxr_hq")``.
    """
    import librosa

    if wave.size == 0:
        raise ResampleError("empty_waveform")
    arr = np.asarray(wave)
    if arr.ndim == 2:
        # (frames, channels) -> mono by mean across channels
        arr = arr.mean(axis=1)
    arr = arr.astype(np.float32, copy=False)

    if sr == target_sr:
        return arr

    try:
        out = librosa.resample(arr, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
    except Exception as exc:  # noqa: BLE001 — narrow to typed error at boundary
        raise ResampleError(f"resample_failed: {exc}") from exc
    return out.astype(np.float32, copy=False)


def trim_silence(
    wave: np.ndarray,
    *,
    sr: int = 16000,
    top_db: float = 30.0,
    min_trimmed_s: float = 0.5,
) -> tuple[np.ndarray, bool]:
    """Symmetric leading + trailing silence trim via ``librosa.effects.trim``.

    Deterministic. If the trimmed length would fall below ``min_trimmed_s``,
    return the un-trimmed input and ``fallback_used=True`` so the CLI can
    audit-log the fallback without failing the row.
    """
    import librosa

    arr = np.asarray(wave, dtype=np.float32)
    if arr.size == 0:
        raise TrimError("empty_waveform")
    try:
        trimmed, _ = librosa.effects.trim(
            arr,
            top_db=top_db,
            frame_length=400,
            hop_length=160,
        )
    except Exception as exc:  # noqa: BLE001
        raise TrimError(f"trim_failed: {exc}") from exc

    if trimmed.shape[0] < int(min_trimmed_s * sr):
        return arr, True
    return trimmed.astype(np.float32, copy=False), False
