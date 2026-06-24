"""Tests for src/data/build_speech_manifests.py."""
from __future__ import annotations

import csv
import json
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
