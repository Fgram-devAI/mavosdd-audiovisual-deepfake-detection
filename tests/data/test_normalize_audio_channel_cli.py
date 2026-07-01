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
