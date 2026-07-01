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


def test_trim_silence_removes_padding(make_padded_speechlike_wav):
    p = make_padded_speechlike_wav()
    wave, sr = sf.read(str(p))
    assert sr == 16000
    trimmed, fallback = an.trim_silence(wave, sr=sr, top_db=30.0, min_trimmed_s=0.5)
    assert fallback is False
    # Original: 2.0 s. Expected body: ~1.0 s. Allow generous slack.
    assert 0.7 * sr <= trimmed.shape[0] <= 1.4 * sr


def test_trim_silence_falls_back_for_short_clip(make_silent_wav_norm):
    p = make_silent_wav_norm(duration_s=0.3)
    wave, sr = sf.read(str(p))
    trimmed, fallback = an.trim_silence(wave, sr=sr, top_db=30.0, min_trimmed_s=0.5)
    # Silent input trims to zero → below min_trimmed_s → return un-trimmed.
    assert fallback is True
    assert trimmed.shape[0] == wave.shape[0]


def test_trim_silence_deterministic(make_padded_speechlike_wav):
    p = make_padded_speechlike_wav()
    wave, sr = sf.read(str(p))
    a, _ = an.trim_silence(wave, sr=sr)
    b, _ = an.trim_silence(wave, sr=sr)
    np.testing.assert_array_equal(a, b)


def test_lowpass_attenuates_high_frequency(make_hf_tone_wav, make_tone_wav_norm):
    p_hf = make_hf_tone_wav()  # 7500 Hz
    p_lo = make_tone_wav_norm(name="lo_tone.wav", freq_hz=1000.0)  # 1000 Hz

    hf_wave, sr = sf.read(str(p_hf))
    lo_wave, _ = sf.read(str(p_lo))

    hf_out, hf_skipped = an.lowpass(hf_wave, sr=sr, cutoff_hz=7000.0, order=8)
    lo_out, lo_skipped = an.lowpass(lo_wave, sr=sr, cutoff_hz=7000.0, order=8)
    assert hf_skipped is False and lo_skipped is False

    # 7500 Hz sits above the 7000 Hz cutoff → strong attenuation vs 1000 Hz.
    hf_rms = float(np.sqrt(np.mean(hf_out ** 2)))
    lo_rms = float(np.sqrt(np.mean(lo_out ** 2)))
    assert lo_rms > 0
    ratio_db = 20 * np.log10(hf_rms / lo_rms + 1e-12)
    assert ratio_db <= -20.0


def test_lowpass_nyquist_raises():
    wave = np.zeros(16000, dtype=np.float32)
    with pytest.raises(an.LowpassError):
        an.lowpass(wave, sr=16000, cutoff_hz=8000.0)  # == Nyquist
    with pytest.raises(an.LowpassError):
        an.lowpass(wave, sr=16000, cutoff_hz=9000.0)  # above Nyquist


def test_lowpass_short_clip_skips_gracefully():
    # A 20-sample buffer sits below sosfiltfilt's minimum padding
    # requirement (27 for the 8th-order filter used here); the guard
    # returns the input unchanged.
    wave = (0.3 * np.sin(np.linspace(0, 3.14, 20))).astype(np.float32)
    out, skipped = an.lowpass(wave, sr=16000, cutoff_hz=7000.0, order=8)
    assert skipped is True
    np.testing.assert_array_equal(out, wave)


def test_lowpass_26_sample_clip_skips_not_raises():
    # SciPy's sosfiltfilt for order=8 needs input length > 27; a 26-sample
    # buffer must fall back to skipped=True, not raise LowpassError.
    wave = (0.3 * np.sin(np.linspace(0, 3.14, 26))).astype(np.float32)
    out, skipped = an.lowpass(wave, sr=16000, cutoff_hz=7000.0, order=8)
    assert skipped is True
    np.testing.assert_array_equal(out, wave)


def test_lowpass_deterministic(make_tone_wav_norm):
    p = make_tone_wav_norm()
    wave, sr = sf.read(str(p))
    a, _ = an.lowpass(wave, sr=sr)
    b, _ = an.lowpass(wave, sr=sr)
    np.testing.assert_array_equal(a, b)


def test_loudness_normalize_scales_quiet_up():
    sr = 16000
    t = np.arange(sr * 2) / sr
    quiet = (0.01 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    out, skipped = an.loudness_normalize(quiet, sr=sr, target_lufs=-23.0)
    assert skipped is False
    quiet_peak = float(np.abs(quiet).max())
    out_peak = float(np.abs(out).max())
    assert out_peak > quiet_peak * 1.5


def test_loudness_normalize_scales_hot_down():
    sr = 16000
    t = np.arange(sr * 2) / sr
    hot = (0.95 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    out, skipped = an.loudness_normalize(hot, sr=sr, target_lufs=-23.0)
    assert skipped is False
    hot_peak = float(np.abs(hot).max())
    out_peak = float(np.abs(out).max())
    assert out_peak < hot_peak


def test_loudness_normalize_silence_skips():
    silent = np.zeros(int(16000 * 2), dtype=np.float32)
    out, skipped = an.loudness_normalize(silent, sr=16000)
    assert skipped is True
    np.testing.assert_array_equal(out, silent)


def test_loudness_normalize_deterministic():
    sr = 16000
    t = np.arange(sr * 2) / sr
    x = (0.1 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    a, _ = an.loudness_normalize(x, sr=sr)
    b, _ = an.loudness_normalize(x, sr=sr)
    np.testing.assert_array_equal(a, b)


def test_peak_safety_scales_hot_down():
    wave = np.array([0.0, 1.5, -1.2, 0.3], dtype=np.float32)
    out = an.peak_safety(wave, ceiling=0.99)
    assert float(np.abs(out).max()) <= 0.99 + 1e-6
    # Relative ranks preserved.
    assert np.argmax(np.abs(out)) == np.argmax(np.abs(wave))


def test_peak_safety_leaves_quiet_untouched():
    wave = np.array([0.0, 0.3, -0.2, 0.5], dtype=np.float32)
    out = an.peak_safety(wave, ceiling=0.99)
    np.testing.assert_array_equal(out, wave)


def test_peak_safety_empty_raises():
    with pytest.raises(an.PeakSafetyError):
        an.peak_safety(np.zeros(0, dtype=np.float32))


@pytest.mark.parametrize(
    "bad",
    ["", "  ", "/abs", "..", "..\\evil", "a/b", "a\\b", "with nul\x00here",
     "C:evil", "drive:letter"],
)
def test_validate_path_token_rejects_unsafe(bad):
    with pytest.raises(an.PathTokenError):
        an.validate_path_token(bad, field="provider")


@pytest.mark.parametrize(
    "good",
    ["elevenlabs", "google_tts", "openai_tts", "original", "sample_video_id__voice-abc"],
)
def test_validate_path_token_allows_safe(good):
    assert an.validate_path_token(good, field="sample_id") == good
