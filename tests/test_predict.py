"""Tests for src/predict.py — single-video real/fake scoring CLI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import predict


class TestArgparse:
    def test_missing_video_exits_nonzero(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            predict.main(["--checkpoint", str(tmp_path / "x.pt")])
        assert exc.value.code != 0

    def test_missing_checkpoint_exits_nonzero(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            predict.main(["--video", str(tmp_path / "x.mp4")])
        assert exc.value.code != 0

    def test_parser_has_all_spec_flags(self):
        parser = predict._build_parser()
        dests = {a.dest for a in parser._actions}
        assert {"video", "checkpoint", "device", "json", "threshold",
                "no_codec_match"}.issubset(dests)

    def test_threshold_default_is_half(self):
        parser = predict._build_parser()
        ns = parser.parse_args(["--video", "v.mp4", "--checkpoint", "c.pt"])
        assert ns.threshold == 0.5
        assert ns.json is False
        assert ns.no_codec_match is False
        assert ns.device == "auto"


class TestCheckpointValidator:
    def test_missing_state_dict_raises(self):
        ckpt = {
            "modality": "audio", "backend": "wav2vec2",
            "model_hparams": {"modality": "audio", "emb": 128, "dropout": 0.3},
            "norm_stats": {"eps": 1e-6},
        }
        with pytest.raises(ValueError, match="state_dict"):
            predict._validate_checkpoint(ckpt)

    def test_missing_norm_stats_raises(self):
        ckpt = {
            "state_dict": {}, "modality": "audio", "backend": "wav2vec2",
            "model_hparams": {"modality": "audio", "emb": 128, "dropout": 0.3},
        }
        with pytest.raises(ValueError, match="norm_stats"):
            predict._validate_checkpoint(ckpt)

    def test_bad_modality_raises(self):
        ckpt = {
            "state_dict": {}, "modality": "speech", "backend": "wav2vec2",
            "model_hparams": {"modality": "speech", "emb": 128, "dropout": 0.3},
            "norm_stats": {"eps": 1e-6},
        }
        with pytest.raises(ValueError, match="modality"):
            predict._validate_checkpoint(ckpt)

    def test_non_visual_with_none_backend_raises(self):
        ckpt = {
            "state_dict": {}, "modality": "audio", "backend": None,
            "model_hparams": {"modality": "audio", "emb": 128, "dropout": 0.3},
            "norm_stats": {"eps": 1e-6},
        }
        with pytest.raises(ValueError, match="backend"):
            predict._validate_checkpoint(ckpt)

    def test_visual_with_none_backend_is_ok(self):
        ckpt = {
            "state_dict": {}, "modality": "visual", "backend": None,
            "model_hparams": {"modality": "visual", "emb": 128, "dropout": 0.3},
            "norm_stats": {"eps": 1e-6, "lips_mean": [0.0] * 84, "lips_std": [1.0] * 84},
        }
        predict._validate_checkpoint(ckpt)  # no raise
