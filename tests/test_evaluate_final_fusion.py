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


def _row(a, v, s, split, label, source, sid):
    return {"sample_id": sid, "source_video_id": sid, "split": split,
            "source_folder": source, "final_label_binary": str(label),
            "audio_fake_score": f"{a:.6f}", "audio_backend": "wavlm_normalized",
            "video_av_fake_score": f"{v:.6f}", "video_av_backend": "avhubert_fixed25",
            "sync_inconsistent_score": f"{s:.6f}", "sync_backend": "syncnet",
            "visual_fake_score": "", "missing_features": ""}


def test_comparison_writer_has_all_rows(tmp_path):
    from src.evaluate_final_fusion import main
    from src.train_final_fusion import main as train_main

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    rng = np.random.default_rng(42)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for i in range(80):
        label = int(i % 2)
        a = float(rng.uniform(0.55, 0.9) if label else rng.uniform(0.1, 0.45))
        v = float(rng.uniform(0.55, 0.9) if label else rng.uniform(0.1, 0.45))
        s = float(rng.uniform(0.2, 0.6))
        src = "real" if label == 0 else "echomimic"
        train_rows.append(_row(a, v, s, "train", label, src, f"tr{i}"))
    for i in range(40):
        label = int(i % 2)
        a = float(rng.uniform(0.55, 0.9) if label else rng.uniform(0.1, 0.45))
        v = float(rng.uniform(0.55, 0.9) if label else rng.uniform(0.1, 0.45))
        s = float(rng.uniform(0.2, 0.6))
        src = "real" if label == 0 else "liveportrait"
        val_rows.append(_row(a, v, s, "val", label, src, f"va{i}"))
    _write_scores(train_csv, train_rows)
    _write_scores(val_csv, val_rows)

    logreg_out = tmp_path / "logreg.pt"
    mlp_out = tmp_path / "mlp.pt"
    train_main([
        "--train-scores", str(train_csv), "--val-scores", str(val_csv),
        "--models", "logreg", "mlp",
        "--logreg-out", str(logreg_out), "--mlp-out", str(mlp_out),
        "--epochs", "10", "--patience", "5", "--hidden", "16",
    ])

    txt = tmp_path / "val.txt"
    md = tmp_path / "comparison.md"
    rc = main([
        "--val-scores", str(val_csv),
        "--logreg-ckpt", str(logreg_out),
        "--mlp-ckpt", str(mlp_out),
        "--out", str(txt), "--comparison", str(md),
    ])
    assert rc == 0
    body = md.read_text()
    for baseline in ("audio_only", "video_av_only", "sync_only",
                     "max_audio_video_av", "max_available",
                     "logistic_fusion", "mlp_fusion",
                     "visual_frame_baseline_notebook_only"):
        assert baseline in body
    assert "roc_auc" in body.lower()
    assert "liveportrait" in body.lower()
    txt_body = txt.read_text()
    assert "logistic_fusion" in txt_body


def test_refuses_test_split(tmp_path):
    from src.evaluate_final_fusion import main

    val_csv = tmp_path / "val.csv"
    _write_scores(val_csv, [_row(0.1, 0.1, 0.1, "test", 0, "real", "x")])
    with pytest.raises(ValueError, match="test split is locked"):
        main([
            "--val-scores", str(val_csv),
            "--split", "test",
            "--out", str(tmp_path / "v.txt"),
            "--comparison", str(tmp_path / "c.md"),
        ])
