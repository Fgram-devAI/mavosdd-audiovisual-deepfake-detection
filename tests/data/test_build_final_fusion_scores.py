import csv
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tiny_manifest(tmp_path: Path) -> Path:
    p = tmp_path / "video_av_manifest.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sample_id", "source_video_id", "split", "source_folder",
            "video_path", "audio_path", "audio_sample_id",
            "video_label", "video_label_binary",
        ])
        w.writeheader()
        w.writerow({"sample_id": "v1", "source_video_id": "v1", "split": "train",
                    "source_folder": "real", "video_path": "", "audio_path": "",
                    "audio_sample_id": "v1", "video_label": "real", "video_label_binary": "0"})
        w.writerow({"sample_id": "v2", "source_video_id": "v2", "split": "val",
                    "source_folder": "echomimic", "video_path": "", "audio_path": "",
                    "audio_sample_id": "v2", "video_label": "fake", "video_label_binary": "1"})
    return p


def test_refuses_test_split(tiny_manifest, tmp_path):
    from src.data import build_final_fusion_scores as m

    with pytest.raises(ValueError, match="test split is locked"):
        m.build_score_table(
            manifest=tiny_manifest, split="test",
            audio_ckpt=tmp_path / "a.pt", sync_ckpt=tmp_path / "s.pt",
            video_av_ckpt=tmp_path / "v.pt", out=tmp_path / "out.csv",
            device="cpu",
        )


def test_writes_expected_schema(tiny_manifest, tmp_path, monkeypatch):
    from src.data import build_final_fusion_scores as m

    def _fake_score_row(row, *, audio, sync, video_av):
        return {
            "audio_fake_score": 0.7,
            "video_av_fake_score": 0.8,
            "sync_inconsistent_score": 0.2,
            "missing_features": "",
        }

    monkeypatch.setattr(m, "score_row", _fake_score_row)
    monkeypatch.setattr(m, "load_heads", lambda **kw: ("audio", "sync", "vav"))
    monkeypatch.setattr(m, "backend_names",
                        lambda **kw: {"audio": "wavlm_normalized", "sync": "syncnet", "video_av": "avhubert_fixed25"})

    out = tmp_path / "scores_train.csv"
    m.build_score_table(
        manifest=tiny_manifest, split="train",
        audio_ckpt=tmp_path / "a.pt", sync_ckpt=tmp_path / "s.pt",
        video_av_ckpt=tmp_path / "v.pt", out=out, device="cpu",
    )
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert [r["sample_id"] for r in rows] == ["v1"]
    assert set(rows[0].keys()) == set(m.FIELDS)
    assert rows[0]["final_label_binary"] == "0"
    assert rows[0]["audio_backend"] == "wavlm_normalized"
    assert rows[0]["video_av_backend"] == "avhubert_fixed25"
    assert rows[0]["sync_backend"] == "syncnet"
    assert rows[0]["visual_fake_score"] == ""


def test_records_missing_features(tiny_manifest, tmp_path, monkeypatch):
    from src.data import build_final_fusion_scores as m

    def _fake_score_row(row, *, audio, sync, video_av):
        if row["sample_id"] == "v2":
            return {"audio_fake_score": None, "video_av_fake_score": 0.5,
                    "sync_inconsistent_score": None,
                    "missing_features": "audio_fake_score|sync_inconsistent_score"}
        return {"audio_fake_score": 0.3, "video_av_fake_score": 0.4,
                "sync_inconsistent_score": 0.1, "missing_features": ""}

    monkeypatch.setattr(m, "score_row", _fake_score_row)
    monkeypatch.setattr(m, "load_heads", lambda **kw: (None, None, None))
    monkeypatch.setattr(m, "backend_names", lambda **kw:
                        {"audio": "wavlm_normalized", "sync": "syncnet", "video_av": "avhubert_fixed25"})

    out = tmp_path / "scores_val.csv"
    m.build_score_table(
        manifest=tiny_manifest, split="val",
        audio_ckpt=tmp_path / "a.pt", sync_ckpt=tmp_path / "s.pt",
        video_av_ckpt=tmp_path / "v.pt", out=out, device="cpu",
    )
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["missing_features"] == "audio_fake_score|sync_inconsistent_score"
    assert rows[0]["audio_fake_score"] == ""
    assert rows[0]["sync_inconsistent_score"] == ""
    assert rows[0]["video_av_fake_score"] == "0.500000"


def test_final_label_binary_from_source_folder(tiny_manifest, tmp_path, monkeypatch):
    from src.data import build_final_fusion_scores as m
    monkeypatch.setattr(m, "score_row",
                        lambda row, **kw: {"audio_fake_score": 0.0, "video_av_fake_score": 0.0,
                                           "sync_inconsistent_score": 0.0, "missing_features": ""})
    monkeypatch.setattr(m, "load_heads", lambda **kw: (None, None, None))
    monkeypatch.setattr(m, "backend_names", lambda **kw:
                        {"audio": "x", "sync": "x", "video_av": "x"})

    out = tmp_path / "scores_train.csv"
    m.build_score_table(manifest=tiny_manifest, split="train",
                        audio_ckpt=tmp_path / "a.pt", sync_ckpt=tmp_path / "s.pt",
                        video_av_ckpt=tmp_path / "v.pt", out=out, device="cpu")
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["source_folder"] == "real"
    assert rows[0]["final_label_binary"] == "0"


def test_audio_score_applies_checkpoint_normalization(monkeypatch):
    from src.data import build_final_fusion_scores as m

    seen = {}

    class DummyModel:
        def __call__(self, *, audio):
            seen["mean"] = float(audio.mean())
            import torch
            return torch.tensor([0.0])

    audio = {
        "model": DummyModel(),
        "audio_dir": Path("."),
        "backend": "wavlm",
        "norm_stats": {
            "audio_mean": np.array([1.0], dtype=np.float32),
            "audio_std": np.array([2.0], dtype=np.float32),
            "eps": 1e-6,
        },
    }
    monkeypatch.setattr(m.np, "load", lambda path: np.array([[3.0], [5.0]], dtype=np.float32))
    monkeypatch.setattr(Path, "exists", lambda self: True)
    score = m._score_audio({"audio_sample_id": "x"}, audio)
    assert score == 0.5
    assert seen["mean"] == 1.5
