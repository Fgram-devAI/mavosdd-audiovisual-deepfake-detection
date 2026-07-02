"""Tests for src/features/extract_syncnet_embeddings.py."""
from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


PAIR_FIELDS = [
    "pair_id", "split", "source_video_id", "lip_feature_path",
    "audio_sample_id", "audio_path", "audio_feature_path", "audio_provider",
    "audio_label", "sync_label", "sync_label_binary", "negative_type",
    "source_folder", "voice_id_or_name",
]


def _row(pair_id: str, *, split: str = "train", vid: str = "vid_a",
         audio_sample_id: str | None = None, audio_path: str = "/tmp/a.wav",
         source_folder: str = "real") -> dict:
    row = {k: "" for k in PAIR_FIELDS}
    row.update({
        "pair_id": pair_id,
        "split": split,
        "source_video_id": vid,
        "audio_sample_id": audio_sample_id or vid,
        "audio_path": audio_path,
        "source_folder": source_folder,
        "sync_label": "sync",
        "sync_label_binary": "0",
        "audio_provider": "original",
    })
    return row


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PAIR_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_iter_pair_manifest_rows_filters_splits_and_sorts_by_pair_id(tmp_path):
    from src.features.extract_syncnet_embeddings import iter_pair_manifest_rows

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [
        _row("z", split="val"),
        _row("a", split="train"),
        _row("b", split="test"),
    ])
    out = list(iter_pair_manifest_rows(manifest, splits=("train", "val"), limit=None))
    assert [r["pair_id"] for r in out] == ["a", "z"]


def test_iter_pair_manifest_rows_refuses_test_split(tmp_path):
    from src.features.extract_syncnet_embeddings import iter_pair_manifest_rows

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("a", split="test")])
    with pytest.raises(ValueError, match=r"test split"):
        list(iter_pair_manifest_rows(manifest, splits=("train", "val", "test"), limit=None))


def test_unique_video_and_audio_units_dedupes(tmp_path):
    from src.features.extract_syncnet_embeddings import unique_visual_and_audio_units

    rows = [
        _row("p1", vid="v1", audio_sample_id="a1"),
        _row("p2", vid="v1", audio_sample_id="a2"),
        _row("p3", vid="v2", audio_sample_id="a1"),
    ]
    visuals, audios = unique_visual_and_audio_units(rows, raw_video_root=Path("data/raw"))
    assert sorted(v for v, _ in visuals) == ["v1", "v2"]
    assert sorted(s for s, _ in audios) == ["a1", "a2"]


def test_log_failure_writes_canonical_schema(tmp_path):
    from src.features.extract_syncnet_embeddings import log_failure

    csv_path = tmp_path / "fail.csv"
    log_failure(csv_path, sample_id="s1", stage="face_detect", error=RuntimeError("no face"))
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows[0]["sample_id"] == "s1"
    assert rows[0]["stage"] == "face_detect"
    assert rows[0]["error_type"] == "RuntimeError"
    assert "no face" in rows[0]["error_message"]
    assert rows[0]["timestamp"]


def test_main_skips_existing_and_writes_new(tmp_path, monkeypatch):
    from src.features import extract_syncnet_embeddings as ex

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("p1", vid="v1", audio_sample_id="a1")])

    out_visual = tmp_path / "sv"
    out_audio = tmp_path / "sa"
    fail_csv = tmp_path / "sf.csv"

    fake_adapter = MagicMock()
    fake_adapter.encode_visual.return_value = np.zeros((2, 512), dtype=np.float32)
    fake_adapter.encode_audio.return_value = np.zeros((2, 512), dtype=np.float32)

    with patch("src.features.extract_syncnet_embeddings.SyncNetBackend") as SB, \
         patch("src.features.extract_syncnet_embeddings.extract_mouth_crops",
               return_value=np.zeros((2, 5, 3, 96, 96), dtype=np.float16)), \
         patch("src.features.extract_syncnet_embeddings._compute_mel",
               return_value=np.zeros((2, 1, 80, 16), dtype=np.float32)):
        SB.from_checkpoint.return_value = fake_adapter
        rc = ex.main([
            "--manifest", str(manifest),
            "--splits", "train",
            "--raw-video-root", str(tmp_path),
            "--out-visual-dir", str(out_visual),
            "--out-audio-dir", str(out_audio),
            "--checkpoint", str(tmp_path / "any.pt"),
            "--failures-csv", str(fail_csv),
        ])
    assert rc == 0
    assert (out_visual / "v1.npy").exists()
    assert (out_audio / "a1.npy").exists()


def test_main_records_failure_on_mouth_detection_error(tmp_path):
    from src.features import extract_syncnet_embeddings as ex
    from src.features.mouth_crop_extract import MouthDetectionError

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("p1", vid="v1", audio_sample_id="a1")])

    out_visual = tmp_path / "sv"
    out_audio = tmp_path / "sa"
    fail_csv = tmp_path / "sf.csv"

    fake_adapter = MagicMock()
    fake_adapter.encode_audio.return_value = np.zeros((2, 512), dtype=np.float32)

    with patch("src.features.extract_syncnet_embeddings.SyncNetBackend") as SB, \
         patch("src.features.extract_syncnet_embeddings.extract_mouth_crops",
               side_effect=MouthDetectionError("face_detect", "no face")), \
         patch("src.features.extract_syncnet_embeddings._compute_mel",
               return_value=np.zeros((2, 1, 80, 16), dtype=np.float32)):
        SB.from_checkpoint.return_value = fake_adapter
        rc = ex.main([
            "--manifest", str(manifest),
            "--splits", "train",
            "--raw-video-root", str(tmp_path),
            "--out-visual-dir", str(out_visual),
            "--out-audio-dir", str(out_audio),
            "--checkpoint", str(tmp_path / "any.pt"),
            "--failures-csv", str(fail_csv),
        ])
    assert rc == 0
    assert fail_csv.exists()
    with fail_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["sample_id"] == "v1"
    assert rows[0]["stage"] == "face_detect"
