import csv
from pathlib import Path

import numpy as np
import pytest


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "sample_id", "source_video_id", "split", "source_folder",
        "final_label_binary",
        "audio_fake_score", "audio_backend",
        "video_av_fake_score", "video_av_backend",
        "sync_inconsistent_score", "sync_backend",
        "visual_fake_score", "missing_features",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _row(**kw):
    base = {"sample_id": "v", "source_video_id": "v", "split": "train",
            "source_folder": "real", "final_label_binary": "0",
            "audio_fake_score": "0.1", "audio_backend": "wavlm_normalized",
            "video_av_fake_score": "0.2", "video_av_backend": "avhubert_fixed25",
            "sync_inconsistent_score": "0.3", "sync_backend": "syncnet",
            "visual_fake_score": "", "missing_features": ""}
    base.update(kw)
    return base


def test_split_filter_and_x_shape(tmp_path):
    from src.data.final_fusion_dataset import FinalFusionDataset

    csv_path = tmp_path / "scores.csv"
    _write_csv(csv_path, [
        _row(sample_id="a", split="train"),
        _row(sample_id="b", split="val", source_folder="echomimic", final_label_binary="1"),
    ])
    ds = FinalFusionDataset(score_csv=csv_path, split="train")
    assert ds.X.shape == (1, 3)
    assert ds.y.tolist() == [0]
    assert ds.sample_ids == ["a"]
    assert ds.source_folders == ["real"]


def test_test_split_refused(tmp_path):
    from src.data.final_fusion_dataset import FinalFusionDataset

    csv_path = tmp_path / "scores.csv"
    _write_csv(csv_path, [_row(split="test")])
    with pytest.raises(ValueError, match="test split is locked"):
        FinalFusionDataset(score_csv=csv_path, split="test")


def test_missing_row_dropped_and_counted(tmp_path):
    from src.data.final_fusion_dataset import FinalFusionDataset

    csv_path = tmp_path / "scores.csv"
    _write_csv(csv_path, [
        _row(sample_id="a", split="train"),
        _row(sample_id="b", split="train", audio_fake_score="",
             missing_features="audio_fake_score"),
    ])
    ds = FinalFusionDataset(score_csv=csv_path, split="train")
    assert ds.X.shape == (1, 3)
    assert ds.excluded_missing == 1
    assert ds.sample_ids == ["a"]


def test_source_folder_not_a_feature(tmp_path):
    from src.data.final_fusion_dataset import FinalFusionDataset

    csv_path = tmp_path / "scores.csv"
    _write_csv(csv_path, [_row(split="train")])
    with pytest.raises(ValueError, match="source_folder"):
        FinalFusionDataset(score_csv=csv_path, split="train",
                           feature_columns=("audio_fake_score", "source_folder"))


def test_x_dtype_float32(tmp_path):
    from src.data.final_fusion_dataset import FinalFusionDataset

    csv_path = tmp_path / "scores.csv"
    _write_csv(csv_path, [_row(split="train")])
    ds = FinalFusionDataset(score_csv=csv_path, split="train")
    assert ds.X.dtype == np.float32
    assert ds.y.dtype == np.int64
