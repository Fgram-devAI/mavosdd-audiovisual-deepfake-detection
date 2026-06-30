"""Pure handcrafted acoustic feature extraction for the confound probe.

This module is deliberately small and side-effect free: one function takes an
audio path, returns a flat dict of scalar floats. The CLI in
``src/analysis/acoustic_probe.py`` is the only consumer that does any I/O
orchestration around it.

Frame parameters are pinned to ``n_fft=400, hop_length=160, win_length=400`` at
``sr=16000`` so the feature values are reproducible regardless of any future
librosa default drift.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_SR = 16000
_N_FFT = 400
_HOP = 160
_WIN = 400
_SILENCE_DBFS = -40.0  # frame-energy threshold for silence/activity ratios

CORE_FEATURE_COLUMNS: list[str] = [
    "duration_s",
    "rms",
    "peak_amplitude",
    "crest_factor",
    "silence_ratio",
    "leading_silence_s",
    "trailing_silence_s",
    "speech_activity_ratio",
    "noise_floor_db",
    "zcr_mean",
    "zcr_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "spectral_flatness_mean",
    "spectral_flatness_std",
    "bandwidth_ceiling_hz",
    "hf_energy_ratio",
]

F0_FEATURE_COLUMNS: list[str] = [
    "f0_mean",
    "f0_std",
    "f0_range",
    "voiced_frame_ratio",
]


class AcousticFeatureError(Exception):
    """Raised when audio cannot be loaded or yields NaN/Inf features."""


def feature_columns(*, with_f0: bool) -> list[str]:
    """Return the ordered feature schema for the given flag set."""
    return list(CORE_FEATURE_COLUMNS) + (list(F0_FEATURE_COLUMNS) if with_f0 else [])


def _load_mono(path: Path, sr: int) -> np.ndarray:
    import librosa

    try:
        wave, _ = librosa.load(str(path), sr=sr, mono=True)
    except Exception as exc:  # librosa wraps soundfile / audioread errors
        raise AcousticFeatureError(f"unreadable_audio: {exc}") from exc
    if wave.size == 0:
        raise AcousticFeatureError("zero_length_audio")
    return wave.astype(np.float32, copy=False)


def _frame_rms_db(wave: np.ndarray) -> np.ndarray:
    import librosa

    rms = librosa.feature.rms(y=wave, frame_length=_N_FFT, hop_length=_HOP, center=True)[0]
    rms = np.maximum(rms, 1e-10)
    return 20.0 * np.log10(rms)


def _silence_mask(frame_db: np.ndarray) -> np.ndarray:
    return frame_db < _SILENCE_DBFS


def _leading_trailing_silence_s(frame_db: np.ndarray) -> tuple[float, float]:
    mask = _silence_mask(frame_db)
    if mask.all():
        total_s = len(mask) * _HOP / _SR
        return float(total_s), float(total_s)
    first_active = int(np.argmin(mask))  # first False
    last_active = len(mask) - 1 - int(np.argmin(mask[::-1]))
    leading_s = first_active * _HOP / _SR
    trailing_s = (len(mask) - 1 - last_active) * _HOP / _SR
    return float(leading_s), float(trailing_s)


def _bandwidth_ceiling_hz(spec_mag: np.ndarray, sr: int) -> float:
    # Highest FFT bin whose median energy across frames exceeds 1% of the loudest bin.
    bin_energy = np.median(spec_mag, axis=1)
    if bin_energy.max() <= 0:
        return 0.0
    threshold = 0.01 * bin_energy.max()
    active_bins = np.where(bin_energy > threshold)[0]
    if active_bins.size == 0:
        return 0.0
    top_bin = int(active_bins.max())
    return float(top_bin * sr / _N_FFT)


def _hf_energy_ratio(spec_mag: np.ndarray, sr: int) -> float:
    freqs = np.linspace(0, sr / 2, spec_mag.shape[0])
    energy = (spec_mag ** 2).sum(axis=1)
    total = energy.sum()
    if total <= 0:
        return 0.0
    hf = energy[freqs >= 4000.0].sum()
    return float(hf / total)


def compute_features(
    audio_path: Path,
    *,
    sr: int = 16000,
    with_f0: bool = False,
) -> dict[str, float]:
    """Compute the handcrafted acoustic feature dict for one audio file.

    Args:
        audio_path: WAV or MP3 (anything librosa can decode at ``sr``).
        sr: Sample rate to resample to. Default 16 kHz.
        with_f0: If True, also compute the four F0 columns (slow path).

    Returns:
        Dict whose keys exactly equal ``feature_columns(with_f0=with_f0)``.

    Raises:
        AcousticFeatureError on any decode failure, zero-length input, or
        NaN/Inf in the computed features.
    """
    import librosa

    if sr != _SR:
        # We hard-code frame parameters for sr=16000; refuse other rates rather
        # than silently producing features at a different time/frequency grid.
        raise AcousticFeatureError(f"unsupported_sr: {sr}")

    wave = _load_mono(audio_path, sr)
    duration_s = float(wave.shape[0] / sr)

    abs_wave = np.abs(wave)
    rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    peak = float(abs_wave.max())
    crest = float(peak / rms) if rms > 1e-10 else 0.0

    frame_db = _frame_rms_db(wave)
    silence_mask = _silence_mask(frame_db)
    silence_ratio = float(silence_mask.mean())
    leading_s, trailing_s = _leading_trailing_silence_s(frame_db)
    speech_activity_ratio = float(1.0 - silence_ratio)
    quiet_thresh = int(max(1, 0.1 * frame_db.size))
    noise_floor_db = float(np.sort(frame_db)[:quiet_thresh].mean())

    zcr = librosa.feature.zero_crossing_rate(wave, frame_length=_N_FFT, hop_length=_HOP)[0]
    centroid = librosa.feature.spectral_centroid(
        y=wave, sr=sr, n_fft=_N_FFT, hop_length=_HOP, win_length=_WIN
    )[0]
    bandwidth = librosa.feature.spectral_bandwidth(
        y=wave, sr=sr, n_fft=_N_FFT, hop_length=_HOP, win_length=_WIN
    )[0]
    rolloff = librosa.feature.spectral_rolloff(
        y=wave, sr=sr, n_fft=_N_FFT, hop_length=_HOP, win_length=_WIN
    )[0]
    flatness = librosa.feature.spectral_flatness(
        y=wave, n_fft=_N_FFT, hop_length=_HOP, win_length=_WIN
    )[0]

    spec_mag = np.abs(
        librosa.stft(wave, n_fft=_N_FFT, hop_length=_HOP, win_length=_WIN)
    )

    feats: dict[str, float] = {
        "duration_s": duration_s,
        "rms": rms,
        "peak_amplitude": peak,
        "crest_factor": crest,
        "silence_ratio": silence_ratio,
        "leading_silence_s": leading_s,
        "trailing_silence_s": trailing_s,
        "speech_activity_ratio": speech_activity_ratio,
        "noise_floor_db": noise_floor_db,
        "zcr_mean": float(zcr.mean()),
        "zcr_std": float(zcr.std()),
        "spectral_centroid_mean": float(centroid.mean()),
        "spectral_centroid_std": float(centroid.std()),
        "spectral_bandwidth_mean": float(bandwidth.mean()),
        "spectral_bandwidth_std": float(bandwidth.std()),
        "spectral_rolloff_mean": float(rolloff.mean()),
        "spectral_rolloff_std": float(rolloff.std()),
        "spectral_flatness_mean": float(flatness.mean()),
        "spectral_flatness_std": float(flatness.std()),
        "bandwidth_ceiling_hz": _bandwidth_ceiling_hz(spec_mag, sr),
        "hf_energy_ratio": _hf_energy_ratio(spec_mag, sr),
    }

    if with_f0:
        feats.update(_compute_f0_features(wave, sr))

    for k, v in feats.items():
        if not np.isfinite(v):
            raise AcousticFeatureError(f"non_finite_feature:{k}={v}")

    return feats


def _compute_f0_features(wave: np.ndarray, sr: int) -> dict[str, float]:
    import librosa

    f0, voiced_flag, _ = librosa.pyin(
        wave,
        sr=sr,
        fmin=float(librosa.note_to_hz("C2")),   # ~65 Hz
        fmax=float(librosa.note_to_hz("C7")),   # ~2093 Hz
        frame_length=_N_FFT * 4,                 # pyin wants a longer frame
        hop_length=_HOP,
    )

    voiced = np.asarray(voiced_flag, dtype=bool)
    voiced_ratio = float(voiced.mean()) if voiced.size else 0.0

    if voiced.any():
        voiced_f0 = f0[voiced]
        voiced_f0 = voiced_f0[np.isfinite(voiced_f0)]
    else:
        voiced_f0 = np.array([], dtype=np.float32)

    if voiced_f0.size:
        f0_mean = float(voiced_f0.mean())
        f0_std = float(voiced_f0.std())
        f0_range = float(voiced_f0.max() - voiced_f0.min())
    else:
        f0_mean = 0.0
        f0_std = 0.0
        f0_range = 0.0

    return {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_range": f0_range,
        "voiced_frame_ratio": voiced_ratio,
    }
