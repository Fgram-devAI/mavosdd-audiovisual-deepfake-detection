"""End-to-end CLI smoke tests for src.data.normalize_audio_channel."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.data import normalize_audio_channel as cli


REQUIRED_MANIFEST_COLUMNS = [
    "sample_id", "source_video_id", "split", "media_type", "source_folder",
    "provider", "voice_id_or_name", "audio_path", "video_path",
    "audio_feature_path", "lip_feature_path", "audio_label",
    "audio_label_binary", "video_label", "video_label_binary",
    "pair_label", "pair_label_binary",
]


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in REQUIRED_MANIFEST_COLUMNS})


def test_cli_missing_manifest_exits_1(tmp_path):
    rc = cli.main([
        "--manifest", str(tmp_path / "no_such.csv"),
        "--out-manifest", str(tmp_path / "out.csv"),
        "--out-dir", str(tmp_path / "out_dir"),
        "--failures-csv", str(tmp_path / "fail.csv"),
    ])
    assert rc == 1


def test_cli_missing_required_column_exits_1(tmp_path, make_tone_wav_norm):
    p = make_tone_wav_norm()
    manifest = tmp_path / "bad_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    # Missing audio_path column.
    manifest.write_text("sample_id,split,provider\nsid,train,original\n")
    rc = cli.main([
        "--manifest", str(manifest),
        "--out-manifest", str(tmp_path / "out.csv"),
        "--out-dir", str(tmp_path / "out_dir"),
        "--failures-csv", str(tmp_path / "fail.csv"),
    ])
    assert rc == 1


def test_cli_lowpass_above_nyquist_exits_1(tmp_path, make_tone_wav_norm):
    p = make_tone_wav_norm()
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [{
        "sample_id": "s1", "split": "train", "provider": "original",
        "audio_path": str(p), "audio_label_binary": "0", "source_folder": "real",
    }])
    rc = cli.main([
        "--manifest", str(manifest),
        "--out-manifest", str(tmp_path / "out.csv"),
        "--out-dir", str(tmp_path / "out_dir"),
        "--failures-csv", str(tmp_path / "fail.csv"),
        "--target-sr", "16000",
        "--lowpass-hz", "9000",
    ])
    assert rc == 1


import soundfile as sf


def _row(sample_id, provider, audio_path, split, source_folder, label_bin):
    return {
        "sample_id": sample_id,
        "source_video_id": sample_id,
        "split": split,
        "media_type": "audio",
        "source_folder": source_folder,
        "provider": provider,
        "voice_id_or_name": "",
        "audio_path": str(audio_path),
        "video_path": "",
        "audio_feature_path": "",
        "lip_feature_path": "",
        "audio_label": "bonafide" if label_bin == "0" else "spoof",
        "audio_label_binary": label_bin,
        "video_label": "na",
        "video_label_binary": "",
        "pair_label": "na",
        "pair_label_binary": "",
    }


def test_cli_end_to_end_happy_path(
    tmp_path,
    make_tone_wav_norm,
    make_hot_wav,
    make_hf_tone_wav,
    make_padded_speechlike_wav,
    make_broken_wav,
):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    p_clean = make_tone_wav_norm(name="clean.wav")
    p_hot = make_hot_wav(name="hot.wav")
    p_hf = make_hf_tone_wav(name="hf.wav")
    p_padded = make_padded_speechlike_wav(name="padded.wav")
    p_broken = make_broken_wav(name="broken.wav")

    manifest = tmp_path / "in.csv"
    _write_manifest(manifest, [
        _row("s1", "original", p_clean, "train", "real", "0"),
        _row("s2", "elevenlabs", p_hot, "val", "real", "1"),
        _row("s3", "google_tts", p_hf, "train", "real", "1"),
        _row("s4", "openai_tts", p_padded, "test", "real", "1"),
        _row("s5", "elevenlabs", p_broken, "train", "real", "1"),
    ])

    out_dir = tmp_path / "audio_normalized"
    out_manifest = tmp_path / "out.csv"
    fail_csv = tmp_path / "fail.csv"

    rc = cli.main([
        "--manifest", str(manifest),
        "--out-manifest", str(out_manifest),
        "--out-dir", str(out_dir),
        "--failures-csv", str(fail_csv),
    ])
    assert rc == 0

    # WAVs exist for the four valid rows.
    for sid, prov in [("s1", "original"), ("s2", "elevenlabs"), ("s3", "google_tts"), ("s4", "openai_tts")]:
        wav = out_dir / prov / f"{sid}.wav"
        assert wav.exists(), wav
        info = sf.info(str(wav))
        assert info.samplerate == 16000
        assert info.channels == 1
        assert info.subtype == "PCM_16"

    # Output manifest has 4 rows and 18 columns.
    with out_manifest.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert "original_audio_path" in rows[0]
    for r in rows:
        assert r["audio_path"].startswith(str(out_dir))
        assert Path(r["audio_path"]).exists()
        assert Path(r["original_audio_path"]).exists()

    # Failures CSV has exactly 1 row for the broken file.
    with fail_csv.open() as f:
        fails = list(csv.DictReader(f))
    assert len(fails) == 1
    assert fails[0]["sample_id"] == "s5"
    assert fails[0]["stage"] == "decode"


def test_cli_resume_skips_existing(tmp_path, make_tone_wav_norm):
    p = make_tone_wav_norm(name="clean.wav")
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("s1", "original", p, "train", "real", "0")])
    args = [
        "--manifest", str(manifest),
        "--out-manifest", str(tmp_path / "out.csv"),
        "--out-dir", str(tmp_path / "out_dir"),
        "--failures-csv", str(tmp_path / "fail.csv"),
    ]
    assert cli.main(args) == 0
    wav_path = tmp_path / "out_dir" / "original" / "s1.wav"
    mtime_1 = wav_path.stat().st_mtime_ns
    assert cli.main(args) == 0
    mtime_2 = wav_path.stat().st_mtime_ns
    assert mtime_1 == mtime_2  # not overwritten


def test_cli_overwrite_rewrites(tmp_path, make_tone_wav_norm):
    import time
    p = make_tone_wav_norm(name="clean.wav")
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("s1", "original", p, "train", "real", "0")])
    args = [
        "--manifest", str(manifest),
        "--out-manifest", str(tmp_path / "out.csv"),
        "--out-dir", str(tmp_path / "out_dir"),
        "--failures-csv", str(tmp_path / "fail.csv"),
    ]
    assert cli.main(args) == 0
    wav_path = tmp_path / "out_dir" / "original" / "s1.wav"
    mtime_1 = wav_path.stat().st_mtime_ns
    time.sleep(0.01)
    assert cli.main(args + ["--overwrite"]) == 0
    mtime_2 = wav_path.stat().st_mtime_ns
    assert mtime_2 > mtime_1


def test_cli_dry_run_writes_no_artifacts(tmp_path, make_tone_wav_norm):
    p = make_tone_wav_norm(name="clean.wav")
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("s1", "original", p, "train", "real", "0")])
    out_dir = tmp_path / "out_dir"
    out_manifest = tmp_path / "out.csv"
    args = [
        "--manifest", str(manifest),
        "--out-manifest", str(out_manifest),
        "--out-dir", str(out_dir),
        "--failures-csv", str(tmp_path / "fail.csv"),
        "--dry-run",
    ]
    assert cli.main(args) == 0
    # Dry-run must not write audio or the output manifest. Diagnostics
    # (failures/fallback CSVs and _run.json in Task 10) may still be written.
    assert not out_manifest.exists()
    assert not (out_dir / "original" / "s1.wav").exists()


def test_cli_limit_stops_after_n(tmp_path, make_tone_wav_norm):
    manifest = tmp_path / "m.csv"
    rows = [
        _row(f"s{i}", "original", make_tone_wav_norm(name=f"clean{i}.wav"),
             "train", "real", "0")
        for i in range(3)
    ]
    _write_manifest(manifest, rows)
    args = [
        "--manifest", str(manifest),
        "--out-manifest", str(tmp_path / "out.csv"),
        "--out-dir", str(tmp_path / "out_dir"),
        "--failures-csv", str(tmp_path / "fail.csv"),
        "--limit", "2",
    ]
    assert cli.main(args) == 0
    with (tmp_path / "out.csv").open() as f:
        n = sum(1 for _ in csv.DictReader(f))
    assert n == 2


def test_cli_no_flags_skip_stages(tmp_path, make_tone_wav_norm):
    p = make_tone_wav_norm(name="clean.wav")
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("s1", "original", p, "train", "real", "0")])
    args = [
        "--manifest", str(manifest),
        "--out-manifest", str(tmp_path / "out.csv"),
        "--out-dir", str(tmp_path / "out_dir"),
        "--failures-csv", str(tmp_path / "fail.csv"),
        "--no-trim", "--no-loudness", "--no-lowpass",
    ]
    assert cli.main(args) == 0
    # Skipping all transforms other than resample: output is still 16 kHz mono
    # (resample stage always runs), and no fallback CSV exists.
    wav_path = tmp_path / "out_dir" / "original" / "s1.wav"
    assert wav_path.exists()
    fb = tmp_path / "out_dir" / "_fallbacks.csv"
    assert not fb.exists()


def test_cli_unsafe_provider_token_logged_as_path_failure(tmp_path, make_tone_wav_norm):
    p = make_tone_wav_norm(name="clean.wav")
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("s1", "..", p, "train", "real", "0")])
    fail_csv = tmp_path / "fail.csv"
    args = [
        "--manifest", str(manifest),
        "--out-manifest", str(tmp_path / "out.csv"),
        "--out-dir", str(tmp_path / "out_dir"),
        "--failures-csv", str(fail_csv),
    ]
    assert cli.main(args) == 0
    with fail_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["stage"] == "path"
