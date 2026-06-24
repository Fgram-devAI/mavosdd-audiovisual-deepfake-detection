"""Tests for src/data/build_speech_manifests.py."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import pytest


def _write_split_csv(path: Path, video_ids: list[str], source_folder: str = "real") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "relative_path", "source_folder", "binary_label",
                    "duration_s", "fps", "n_frames"])
        for vid in video_ids:
            w.writerow([vid, f"data/raw/{source_folder}/{vid}.mp4", source_folder,
                        0 if source_folder == "real" else 1, "5.0", "24.0", "120"])


def test_load_split_map_maps_each_video_to_its_split(tmp_path):
    from src.data.build_speech_manifests import load_split_map

    splits_dir = tmp_path / "splits"
    _write_split_csv(splits_dir / "train.csv", ["a", "b"])
    _write_split_csv(splits_dir / "val.csv", ["c"])
    _write_split_csv(splits_dir / "test.csv", ["d"])

    m = load_split_map(splits_dir)

    assert m == {"a": "train", "b": "train", "c": "val", "d": "test"}


def test_load_split_map_raises_on_duplicate_across_splits(tmp_path):
    from src.data.build_speech_manifests import load_split_map

    splits_dir = tmp_path / "splits"
    _write_split_csv(splits_dir / "train.csv", ["dup", "x"])
    _write_split_csv(splits_dir / "val.csv", ["dup"])
    _write_split_csv(splits_dir / "test.csv", ["y"])

    with pytest.raises(ValueError, match=r"^split leakage:.*dup"):
        load_split_map(splits_dir)


def _write_manifest_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    """rows = [(video_id, source_folder), ...]"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "relative_path", "source_folder", "binary_label",
                    "duration_s", "fps", "n_frames"])
        for vid, src in rows:
            label = 0 if src == "real" else 1
            w.writerow([vid, f"data/raw/{src}/{vid}.mp4", src, label, "5.0", "24.0", "120"])


def test_iter_native_rows_real_is_bonafide_audio_and_real_video(tmp_path):
    from src.data.build_speech_manifests import iter_native_rows

    manifest = tmp_path / "manifest.csv"
    _write_manifest_csv(manifest, [("vid_real", "real")])
    split_map = {"vid_real": "train"}

    rows = list(iter_native_rows(manifest, split_map))

    assert len(rows) == 1
    r = rows[0]
    assert r["source_video_id"] == "vid_real"
    assert r["sample_id"] == "vid_real"
    assert r["split"] == "train"
    assert r["media_type"] == "video"
    assert r["source_folder"] == "real"
    assert r["provider"] == "original"
    assert r["voice_id_or_name"] == ""
    assert r["audio_label"] == "bonafide"
    assert r["audio_label_binary"] == 0
    assert r["video_label"] == "real"
    assert r["video_label_binary"] == 0
    assert r["pair_label"] == "na"
    assert r["pair_label_binary"] == ""
    assert r["video_path"] == "data/raw/real/vid_real.mp4"
    assert r["audio_feature_path"] == "data/features/audio/vid_real.npy"


def test_iter_native_rows_echomimic_is_bonafide_audio_and_fake_video(tmp_path):
    from src.data.build_speech_manifests import iter_native_rows

    manifest = tmp_path / "manifest.csv"
    _write_manifest_csv(manifest, [("vid_em", "echomimic")])
    rows = list(iter_native_rows(manifest, {"vid_em": "val"}))

    r = rows[0]
    assert r["source_folder"] == "echomimic"
    assert r["audio_label"] == "bonafide"
    assert r["audio_label_binary"] == 0
    assert r["video_label"] == "fake"
    assert r["video_label_binary"] == 1


def test_iter_native_rows_memo_is_bonafide_audio_and_fake_video(tmp_path):
    from src.data.build_speech_manifests import iter_native_rows

    manifest = tmp_path / "manifest.csv"
    _write_manifest_csv(manifest, [("vid_memo", "memo")])
    rows = list(iter_native_rows(manifest, {"vid_memo": "test"}))

    r = rows[0]
    assert r["source_folder"] == "memo"
    assert r["audio_label"] == "bonafide"
    assert r["video_label"] == "fake"
    assert r["video_label_binary"] == 1


def test_iter_native_rows_skips_videos_missing_from_split_map(tmp_path, caplog):
    from src.data.build_speech_manifests import iter_native_rows

    manifest = tmp_path / "manifest.csv"
    _write_manifest_csv(manifest, [("known", "real"), ("orphan", "real")])

    with caplog.at_level(logging.WARNING):
        rows = list(iter_native_rows(manifest, {"known": "train"}))

    assert [r["source_video_id"] for r in rows] == ["known"]
    assert any("orphan" in rec.message for rec in caplog.records)
