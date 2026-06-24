"""Tests for src/train.py harness and LateFusionClassifier audio-only path."""
from __future__ import annotations

import pytest
import torch

from src.models.late_fusion import LateFusionClassifier


class TestLateFusionAudioOnlyForward:
    def test_audio_forward_accepts_audio_only(self):
        model = LateFusionClassifier("audio")
        logits = model(torch.randn(4, 199, 768))
        assert logits.shape == (4,)

    def test_fusion_forward_still_three_args(self):
        model = LateFusionClassifier("fusion")
        logits = model(
            torch.randn(4, 199, 768),
            torch.randn(4, 20, 84),
            torch.ones(4, 20),
        )
        assert logits.shape == (4,)

    def test_visual_forward_requires_lips_and_mask(self):
        model = LateFusionClassifier("visual")
        with pytest.raises(ValueError, match="visual"):
            model(torch.randn(4, 199, 768))

    def test_fusion_forward_requires_lips_and_mask(self):
        model = LateFusionClassifier("fusion")
        with pytest.raises(ValueError, match="fusion"):
            model(torch.randn(4, 199, 768))
