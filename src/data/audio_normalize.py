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


def lowpass(
    wave: np.ndarray,
    *,
    sr: int = 16000,
    cutoff_hz: float = 7000.0,
    order: int = 8,
) -> tuple[np.ndarray, bool]:
    """Zero-phase Butterworth low-pass at ``cutoff_hz`` (default 7 kHz).

    Uses ``scipy.signal.butter`` with ``output="sos"`` and ``sosfiltfilt``.
    Cutoff must be strictly below Nyquist (``sr / 2``); otherwise
    ``LowpassError`` is raised. Waveforms shorter than ``sosfiltfilt``'s
    minimum padding requirement are returned unchanged with ``skipped=True``.
    """
    import scipy.signal as sps

    arr = np.asarray(wave, dtype=np.float32)
    if arr.size == 0:
        raise LowpassError("empty_waveform")

    nyquist = sr / 2.0
    if cutoff_hz >= nyquist:
        raise LowpassError(f"cutoff_at_or_above_nyquist: cutoff={cutoff_hz} sr={sr}")

    try:
        sos = sps.butter(order, cutoff_hz, btype="low", fs=sr, output="sos")
    except Exception as exc:  # noqa: BLE001
        raise LowpassError(f"butter_failed: {exc}") from exc

    # sosfiltfilt's internal padlen check (as of SciPy 1.15) requires
    # arr.shape[0] > 3 * (2 * n_sections + 1). For an 8th-order filter that
    # is 4 SOS sections → padlen = 27, so waveforms of length <= 27 must
    # skip filtering entirely. We ALSO catch ValueError from sosfiltfilt
    # in case a future SciPy tightens the requirement further.
    n_sections = sos.shape[0]
    padlen = 3 * (2 * n_sections + 1)
    if arr.shape[0] <= padlen:
        return arr, True

    try:
        out = sps.sosfiltfilt(sos, arr)
    except ValueError as exc:
        if "padlen" in str(exc) or "length of the input" in str(exc):
            return arr, True
        raise LowpassError(f"sosfiltfilt_failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise LowpassError(f"sosfiltfilt_failed: {exc}") from exc
    return out.astype(np.float32, copy=False), False


def loudness_normalize(
    wave: np.ndarray,
    *,
    sr: int = 16000,
    target_lufs: float = -23.0,
    silence_floor_lufs: float = -70.0,
) -> tuple[np.ndarray, bool]:
    """EBU R128 loudness normalize via ``pyloudnorm``.

    Measures integrated LUFS with a 400 ms block; if the result is ``-inf``
    or below ``silence_floor_lufs``, returns the input unchanged with
    ``skipped_silence=True`` so the CLI can audit-log the silence skip
    without failing the row.
    """
    import math

    import pyloudnorm as pyln

    arr = np.asarray(wave, dtype=np.float32)
    if arr.size == 0:
        raise LoudnessError("empty_waveform")

    try:
        meter = pyln.Meter(rate=sr, block_size=0.400)
        measured = float(meter.integrated_loudness(arr.astype(np.float64)))
    except Exception as exc:  # noqa: BLE001
        raise LoudnessError(f"measure_failed: {exc}") from exc

    if not math.isfinite(measured) or measured <= silence_floor_lufs:
        return arr, True

    gain_db = target_lufs - measured
    gain = float(10.0 ** (gain_db / 20.0))
    out = (arr * gain).astype(np.float32, copy=False)
    return out, False


def peak_safety(wave: np.ndarray, *, ceiling: float = 0.99) -> np.ndarray:
    """Scale the entire buffer so that ``max(|wave|) <= ceiling``.

    Deterministic. Never soft-clips. If the peak is already within the
    ceiling, returns the input unchanged.
    """
    arr = np.asarray(wave, dtype=np.float32)
    if arr.size == 0:
        raise PeakSafetyError("empty_waveform")
    peak = float(np.abs(arr).max())
    if peak <= ceiling or peak == 0.0:
        return arr
    scale = float(ceiling) / peak
    return (arr * scale).astype(np.float32, copy=False)
