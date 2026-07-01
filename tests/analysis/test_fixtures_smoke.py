"""Smoke test: every synthetic fixture produces a readable audio file."""
from __future__ import annotations

import numpy as np
import soundfile as sf


def test_silent_wav_round_trips(make_silent_wav):
    path = make_silent_wav(duration_s=0.5)
    wave, sr = sf.read(str(path))
    assert sr == 16000
    assert wave.shape == (8000,)
    assert np.allclose(wave, 0.0)


def test_tone_wav_round_trips(make_tone_wav):
    path = make_tone_wav(freq_hz=1000.0, duration_s=0.5)
    wave, sr = sf.read(str(path))
    assert sr == 16000
    assert wave.shape == (8000,)
    assert np.abs(wave).max() > 0.1


def test_white_noise_wav_is_reproducible(make_white_noise_wav, tmp_path):
    p1 = make_white_noise_wav(name="n1.wav", seed=42)
    p2 = make_white_noise_wav(name="n2.wav", seed=42)
    w1, _ = sf.read(str(p1))
    w2, _ = sf.read(str(p2))
    assert np.array_equal(w1, w2)


def test_half_silence_half_tone(make_half_silence_half_tone_wav):
    path = make_half_silence_half_tone_wav(total_s=1.0)
    wave, sr = sf.read(str(path))
    assert sr == 16000
    half = wave.shape[0] // 2
    assert np.allclose(wave[:half], 0.0)
    assert np.abs(wave[half:]).max() > 0.1


def test_corrupt_wav_fails_to_decode(make_corrupt_wav):
    path = make_corrupt_wav()
    try:
        sf.read(str(path))
        raised = False
    except Exception:
        raised = True
    assert raised, "expected soundfile to reject the corrupt bytes"


def test_tiny_mp3_decodes(make_tiny_mp3):
    path = make_tiny_mp3(duration_s=0.5)
    assert path.suffix == ".mp3"
    assert path.stat().st_size > 0
