"""Tests for src/predict.py — single-video real/fake scoring CLI."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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


class TestCodecMatchSignal:
    def test_visual_modality_always_skips(self):
        ckpt = {"modality": "visual", "audio_dir": None}
        assert predict._should_codec_match(ckpt, True) is False

    def test_no_codec_match_flag_skips(self):
        ckpt = {"modality": "audio", "audio_dir": "data/features/audio_wav2vec2_codec"}
        assert predict._should_codec_match(ckpt, False) is False

    def test_non_codec_audio_dir_skips(self):
        ckpt = {"modality": "audio", "audio_dir": "data/features/audio_wav2vec2"}
        assert predict._should_codec_match(ckpt, True) is False

    def test_codec_audio_dir_enables(self):
        ckpt = {"modality": "audio", "audio_dir": "data/features/audio_wav2vec2_codec"}
        assert predict._should_codec_match(ckpt, True) is True

    def test_fusion_codec_audio_dir_enables(self):
        ckpt = {"modality": "fusion", "audio_dir": "data/features/audio_hubert_codec"}
        assert predict._should_codec_match(ckpt, True) is True

    def test_missing_audio_dir_skips(self):
        ckpt = {"modality": "audio"}
        assert predict._should_codec_match(ckpt, True) is False


class TestNormalize:
    def test_returns_input_when_mean_is_none(self):
        x = np.ones((3, 4), dtype=np.float32)
        out = predict._normalize(x, None, np.ones(4, dtype=np.float32), 1e-6)
        assert out is x

    def test_returns_input_when_std_is_none(self):
        x = np.ones((3, 4), dtype=np.float32)
        out = predict._normalize(x, np.zeros(4, dtype=np.float32), None, 1e-6)
        assert out is x

    def test_centers_and_scales(self):
        x = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        mean = np.array([4.0, 6.0], dtype=np.float32)
        std = np.array([2.0, 2.0], dtype=np.float32)
        out = predict._normalize(x, mean, std, 0.0)
        np.testing.assert_allclose(out, np.array([[-1.0, -1.0], [1.0, 1.0]]))

    def test_eps_prevents_divide_by_zero(self):
        x = np.zeros((1, 2), dtype=np.float32)
        out = predict._normalize(x, np.zeros(2, dtype=np.float32),
                                  np.zeros(2, dtype=np.float32), 1e-6)
        assert np.isfinite(out).all()
