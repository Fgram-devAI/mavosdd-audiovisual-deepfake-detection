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


def test_lowpass_short_clip_skips_gracefully(make_short_wav):
    p = make_short_wav(duration_s=0.005)  # 80 samples at 16 kHz
    wave, sr = sf.read(str(p))
    out, skipped = an.lowpass(wave, sr=sr, cutoff_hz=7000.0, order=8)
    assert skipped is True
    np.testing.assert_array_equal(out, wave.astype(np.float32))


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
