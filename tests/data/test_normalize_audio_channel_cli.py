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
        # Loosen thresholds: this test has 1 failure out of 5 rows (rate=0.2).
        "--max-failure-rate", "1.0",
        "--max-group-failure-rate", "1.0",
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
        # Loosen thresholds: 1/1 failure rate would otherwise trip exit 2.
        "--max-failure-rate", "1.0",
        "--max-group-failure-rate", "1.0",
    ]
    assert cli.main(args) == 0
    with fail_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["stage"] == "path"


# ── Task 10: _run.json provenance + threshold guards ────────────────────────


def test_cli_run_json_records_counts_and_grouped_summaries(
    tmp_path, make_tone_wav_norm, make_broken_wav
):
    p_ok = make_tone_wav_norm(name="ok.wav")
    p_broken = make_broken_wav(name="broken.wav")
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [
        _row("s1", "original", p_ok, "train", "real", "0"),
        _row("s2", "elevenlabs", p_broken, "val", "real", "1"),
    ])
    out_dir = tmp_path / "out_dir"
    args = [
        "--manifest", str(manifest),
        "--out-manifest", str(tmp_path / "out.csv"),
        "--out-dir", str(out_dir),
        "--failures-csv", str(tmp_path / "fail.csv"),
        # Loosen thresholds so this single failure doesn't trip exit 2.
        "--max-failure-rate", "1.0",
        "--max-group-failure-rate", "1.0",
    ]
    assert cli.main(args) == 0
    run_json = json.loads((out_dir / "_run.json").read_text())
    assert run_json["n_rows_in"] == 2
    assert run_json["n_rows_valid"] == 1
    assert run_json["n_rows_written"] == 1  # not a dry-run
    assert run_json["n_rows_failed"] == 1
    assert "params" in run_json and run_json["params"]["target_sr"] == 16000
    grouped = run_json["failure_summary"]["by_split_label_provider"]
    # One entry keyed by "val|1|elevenlabs" (or similar tuple form).
    assert any("elevenlabs" in k for k in grouped)


def test_cli_dry_run_reports_valid_but_written_zero(
    tmp_path, make_tone_wav_norm
):
    p = make_tone_wav_norm(name="clean.wav")
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("s1", "original", p, "train", "real", "0")])
    out_dir = tmp_path / "out_dir"
    args = [
        "--manifest", str(manifest),
        "--out-manifest", str(tmp_path / "out.csv"),
        "--out-dir", str(out_dir),
        "--failures-csv", str(tmp_path / "fail.csv"),
        "--dry-run",
    ]
    assert cli.main(args) == 0
    run_json = json.loads((out_dir / "_run.json").read_text())
    assert run_json["n_rows_valid"] == 1
    assert run_json["n_rows_written"] == 0  # dry-run wrote nothing
    assert run_json["params"]["dry_run"] is True


def test_cli_global_failure_rate_exceeded_exits_2(
    tmp_path, make_tone_wav_norm, make_broken_wav
):
    p_ok = make_tone_wav_norm(name="ok.wav")
    p_broken_a = make_broken_wav(name="a.wav")
    p_broken_b = make_broken_wav(name="b.wav")
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [
        _row("s1", "original", p_ok, "train", "real", "0"),
        _row("s2", "elevenlabs", p_broken_a, "train", "real", "1"),
        _row("s3", "elevenlabs", p_broken_b, "train", "real", "1"),
    ])
    args = [
        "--manifest", str(manifest),
        "--out-manifest", str(tmp_path / "out.csv"),
        "--out-dir", str(tmp_path / "out_dir"),
        "--failures-csv", str(tmp_path / "fail.csv"),
        "--max-failure-rate", "0.1",   # 2/3 > 0.1 → exit 2
        "--max-group-failure-rate", "1.0",
    ]
    assert cli.main(args) == 2
    # Diagnostics still written.
    assert (tmp_path / "out_dir" / "_run.json").exists()


def test_cli_group_failure_rate_exceeded_exits_2(
    tmp_path, make_tone_wav_norm, make_broken_wav
):
    # Global rate 0.25 (1/4), but one (train,1,elevenlabs) group is 1/1 = 1.0.
    p_ok1 = make_tone_wav_norm(name="a.wav")
    p_ok2 = make_tone_wav_norm(name="b.wav")
    p_ok3 = make_tone_wav_norm(name="c.wav")
    p_broken = make_broken_wav(name="d.wav")
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [
        _row("s1", "original", p_ok1, "train", "real", "0"),
        _row("s2", "original", p_ok2, "val", "real", "0"),
        _row("s3", "google_tts", p_ok3, "train", "real", "1"),
        _row("s4", "elevenlabs", p_broken, "train", "real", "1"),
    ])
    args = [
        "--manifest", str(manifest),
        "--out-manifest", str(tmp_path / "out.csv"),
        "--out-dir", str(tmp_path / "out_dir"),
        "--failures-csv", str(tmp_path / "fail.csv"),
        "--max-failure-rate", "1.0",
        "--max-group-failure-rate", "0.5",   # elevenlabs/train/1 = 1.0 > 0.5
    ]
    assert cli.main(args) == 2
