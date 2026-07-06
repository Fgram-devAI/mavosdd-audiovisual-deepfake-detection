import csv
from pathlib import Path

import numpy as np
import pytest


def _write_scores(path: Path, rows: list[dict]) -> None:
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


def _row(a, v, s, split, label, source="real", sid=None):
    return {"sample_id": sid or f"{split}_{label}_{a}", "source_video_id": sid or f"x",
            "split": split, "source_folder": source, "final_label_binary": str(label),
            "audio_fake_score": f"{a:.6f}", "audio_backend": "wavlm_normalized",
            "video_av_fake_score": f"{v:.6f}", "video_av_backend": "avhubert_fixed25",
            "sync_inconsistent_score": f"{s:.6f}", "sync_backend": "syncnet",
            "visual_fake_score": "", "missing_features": ""}


@pytest.fixture
def scores(tmp_path: Path) -> tuple[Path, Path]:
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    rng = np.random.default_rng(42)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for i in range(80):
        label = int(i % 2)
        a = float(rng.uniform(0.55, 0.9) if label else rng.uniform(0.1, 0.45))
        v = float(rng.uniform(0.55, 0.9) if label else rng.uniform(0.1, 0.45))
        s = float(rng.uniform(0.2, 0.6))
        source = "real" if label == 0 else "echomimic"
        train_rows.append(_row(a, v, s, "train", label, source, sid=f"tr{i}"))
    for i in range(40):
        label = int(i % 2)
        a = float(rng.uniform(0.55, 0.9) if label else rng.uniform(0.1, 0.45))
        v = float(rng.uniform(0.55, 0.9) if label else rng.uniform(0.1, 0.45))
        s = float(rng.uniform(0.2, 0.6))
        source = "real" if label == 0 else "memo"
        val_rows.append(_row(a, v, s, "val", label, source, sid=f"va{i}"))
    _write_scores(train, train_rows)
    _write_scores(val, val_rows)
    return train, val


def test_train_logreg_only(scores, tmp_path):
    from src.train_final_fusion import main

    train_csv, val_csv = scores
    logreg_out = tmp_path / "logreg.pt"
    rc = main([
        "--train-scores", str(train_csv), "--val-scores", str(val_csv),
        "--models", "logreg",
        "--logreg-out", str(logreg_out),
    ])
    assert rc == 0
    assert logreg_out.exists()


def test_train_mlp_smoke(scores, tmp_path):
    from src.train_final_fusion import main

    train_csv, val_csv = scores
    mlp_out = tmp_path / "mlp.pt"
    rc = main([
        "--train-scores", str(train_csv), "--val-scores", str(val_csv),
        "--models", "mlp",
        "--mlp-out", str(mlp_out),
        "--epochs", "5", "--patience", "5",
        "--hidden", "16", "--seed", "42",
    ])
    assert rc == 0
    assert mlp_out.exists()


def test_refuses_test_split_via_dataset(scores, tmp_path):
    from src.data.final_fusion_dataset import FinalFusionDataset
    train_csv, _ = scores
    with pytest.raises(ValueError, match="test split is locked"):
        FinalFusionDataset(score_csv=train_csv, split="test")
