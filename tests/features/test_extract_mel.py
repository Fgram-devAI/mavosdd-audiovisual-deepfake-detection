"""Tests for src/features/extract_mel.py."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


CSV_FIELDS = [
    "sample_id", "source_video_id", "split", "media_type", "source_folder",
    "provider", "voice_id_or_name", "audio_path", "video_path",
    "audio_feature_path", "lip_feature_path",
    "audio_label", "audio_label_binary",
    "video_label", "video_label_binary",
    "pair_label", "pair_label_binary",
]


def _row(sample_id: str, *, split: str = "train", provider: str = "original",
         audio_path: str = "/dev/null") -> dict:
    blank = {k: "" for k in CSV_FIELDS}
    blank.update({
        "sample_id": sample_id,
        "source_video_id": sample_id,
        "split": split,
        "media_type": "audio",
        "source_folder": "tts_audio",
        "provider": provider,
        "audio_path": audio_path,
        "audio_label": "spoof",
        "audio_label_binary": "1",
    })
    return blank


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_tone(path: Path, *, sr: int = 16000, seconds: float = 4.0, freq: float = 440.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float32) / sr
    wave = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(str(path), wave, sr, subtype="PCM_16")
    return path


def test_mel_extractor_returns_64_mels_positive_time_float32():
    from src.features.extract_mel import MelExtractor

    mel = MelExtractor(sample_rate=16000, n_fft=400, hop_length=160,
                       win_length=400, n_mels=64)
    wave = np.zeros(int(16000 * 4.0), dtype=np.float32)

    arr = mel.extract(wave)

    assert arr.dtype == np.float32
    assert arr.ndim == 2
    assert arr.shape[0] == 64
    assert arr.shape[1] > 0


def test_mel_extractor_is_deterministic_on_same_input():
    from src.features.extract_mel import MelExtractor

    mel = MelExtractor(sample_rate=16000, n_fft=400, hop_length=160,
                       win_length=400, n_mels=64)
    wave = (0.1 * np.sin(2 * np.pi * 440 * np.arange(64000) / 16000)).astype(np.float32)

    a = mel.extract(wave)
    b = mel.extract(wave)

    assert np.array_equal(a, b)


def test_iter_manifest_rows_split_and_limit(tmp_path):
    from src.features.extract_mel import iter_manifest_rows

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [
        _row("a", split="train"),
        _row("b", split="val"),
        _row("c", split="train"),
        _row("d", split="train"),
    ])

    out = list(iter_manifest_rows(manifest, split="train", limit=2))

    assert [r["sample_id"] for r in out] == ["a", "c"]


def _fake_mel_extractor(monkeypatch):
    """Build a MelExtractor stub whose .extract returns a fixed (64, 5) float32 array."""
    class _Stub:
        sample_rate = 16000
        n_mels = 64
        def extract(self, wave):
            return np.full((64, 5), 1.5, dtype=np.float32)
    return _Stub()


def test_extract_writes_one_npy_per_row_using_sample_id(tmp_path):
    from src.features.extract_mel import extract

    wav_a = _write_tone(tmp_path / "a.wav")
    wav_b = _write_tone(tmp_path / "b.wav")
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [
        _row("alpha", audio_path=str(wav_a)),
        _row("beta", audio_path=str(wav_b)),
    ])
    out_dir = tmp_path / "out"
    mel = _fake_mel_extractor(None)

    counts = extract(
        manifest, mel, out_dir,
        split=None, limit=None, overwrite=False, seconds=4.0,
    )

    assert counts == {"written": 2, "skipped": 0, "failed": 0}
    a = np.load(out_dir / "alpha.npy")
    b = np.load(out_dir / "beta.npy")
    assert a.shape == (64, 5) and a.dtype == np.float32
    assert b.shape == (64, 5) and b.dtype == np.float32


def test_extract_skips_existing_unless_overwrite(tmp_path):
    from src.features.extract_mel import extract

    wav = _write_tone(tmp_path / "alpha.wav")
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("alpha", audio_path=str(wav))])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    np.save(out_dir / "alpha.npy", np.zeros((1, 1), dtype=np.float32))

    mel = _fake_mel_extractor(None)
    counts_skip = extract(
        manifest, mel, out_dir,
        split=None, limit=None, overwrite=False, seconds=4.0,
    )
    counts_over = extract(
        manifest, mel, out_dir,
        split=None, limit=None, overwrite=True, seconds=4.0,
    )

    assert counts_skip == {"written": 0, "skipped": 1, "failed": 0}
    assert counts_over == {"written": 1, "skipped": 0, "failed": 0}


def test_extract_logs_and_skips_missing_audio_path(tmp_path, capsys):
    from src.features.extract_mel import extract

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [
        _row("ghost", audio_path=str(tmp_path / "does_not_exist.wav")),
    ])
    out_dir = tmp_path / "out"
    mel = _fake_mel_extractor(None)

    counts = extract(
        manifest, mel, out_dir,
        split=None, limit=None, overwrite=False, seconds=4.0,
    )

    assert counts == {"written": 0, "skipped": 0, "failed": 1}
    assert not (out_dir / "ghost.npy").exists()


def test_extract_logs_and_skips_when_sample_id_column_missing(tmp_path):
    from src.features.extract_mel import extract

    manifest = tmp_path / "m.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "audio_path"])
        w.writeheader()
        w.writerow({"split": "train", "audio_path": str(tmp_path / "a.wav")})

    out_dir = tmp_path / "out"
    mel = _fake_mel_extractor(None)

    counts = extract(
        manifest, mel, out_dir,
        split=None, limit=None, overwrite=False, seconds=4.0,
    )

    assert counts == {"written": 0, "skipped": 0, "failed": 1}


def test_main_runs_with_manifest_and_explicit_out_dir(tmp_path, monkeypatch):
    from src.features import extract_mel as cli

    wav = _write_tone(tmp_path / "alpha.wav")
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("alpha", audio_path=str(wav))])
    out_dir = tmp_path / "out"

    rc = cli.main([
        "--manifest", str(manifest),
        "--out-dir", str(out_dir),
        "--limit", "1",
    ])

    assert rc == 0
    arr = np.load(out_dir / "alpha.npy")
    assert arr.shape[0] == 64 and arr.shape[1] > 0 and arr.dtype == np.float32


def test_main_split_filter_only_processes_requested_split(tmp_path):
    from src.features import extract_mel as cli

    wav_t = _write_tone(tmp_path / "t.wav")
    wav_v = _write_tone(tmp_path / "v.wav")
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [
        _row("t1", split="train", audio_path=str(wav_t)),
        _row("v1", split="val", audio_path=str(wav_v)),
    ])
    out_dir = tmp_path / "out"

    cli.main([
        "--manifest", str(manifest),
        "--out-dir", str(out_dir),
        "--split", "train",
    ])

    assert (out_dir / "t1.npy").exists()
    assert not (out_dir / "v1.npy").exists()


def test_main_uses_default_out_dir_when_omitted(tmp_path, monkeypatch):
    from src.features import extract_mel as cli

    wav = _write_tone(tmp_path / "alpha.wav")
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("alpha", audio_path=str(wav))])

    default_dir = tmp_path / "default"
    monkeypatch.setattr(cli.common, "FEAT_AUDIO_MEL_DIR", default_dir)

    cli.main(["--manifest", str(manifest)])

    assert (default_dir / "alpha.npy").exists()
