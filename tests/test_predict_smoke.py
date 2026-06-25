"""Skipif-gated smoke test: run predict.py against a real .mp4 if available."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src import predict
from src.models.late_fusion import LateFusionClassifier


_REAL_DIR = Path("data/raw/real")
_REAL_VIDEOS = sorted(_REAL_DIR.glob("*.mp4")) if _REAL_DIR.exists() else []


@pytest.mark.skipif(not _REAL_VIDEOS,
                    reason="data/raw/real/*.mp4 not present; smoke test skipped")
def test_predict_smoke_first_real_video(tmp_path, capsys):
    video = _REAL_VIDEOS[0]

    # Build a randomly-initialized AUDIO checkpoint so we don't need a trained model on disk.
    model = LateFusionClassifier("audio", emb=128, p=0.3)
    ckpt_path = tmp_path / "smoke_audio.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "modality": "audio",
        "backend": "wav2vec2",
        "audio_dir": "data/features/audio_wav2vec2",  # non-_codec → codec_match=False
        "model_hparams": {"modality": "audio", "emb": 128, "dropout": 0.3},
        "norm_stats": {
            "audio_mean": np.zeros(768, dtype=np.float32),
            "audio_std": np.ones(768, dtype=np.float32),
            "eps": 1e-6,
        },
    }, ckpt_path)

    rc = predict.main([
        "--video", str(video), "--checkpoint", str(ckpt_path),
        "--device", "cpu", "--json",
    ])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out.strip())
    assert 0.0 <= parsed["p_spoof"] <= 1.0
    assert parsed["label"] in {"bonafide", "spoof"}
