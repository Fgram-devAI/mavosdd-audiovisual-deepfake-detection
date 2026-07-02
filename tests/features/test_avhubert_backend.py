"""Tests for src/features/avhubert_backend.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src import common


CHECKPOINT_MISSING = not common.AVHUBERT_CKPT_PATH.exists()


def test_backend_declares_embedding_dim_and_mouth_spec():
    from src.features.avhubert_backend import AVHubertBackend
    from src.features.mouth_crop_extract import AVHUBERT_SPEC

    assert AVHubertBackend.embedding_dim == 768
    assert AVHubertBackend.mouth_spec == AVHUBERT_SPEC


def test_encode_visual_returns_time_by_768_from_fake_module():
    import torch

    from src.features import avhubert_backend as ab

    class _Fake:
        def encode_visual_tokens(self, x):
            return torch.zeros(x.shape[0], 50, 768)

        def encode_audio_tokens(self, x):
            return torch.zeros(x.shape[0], 100, 768)

    backend = ab.AVHubertBackend(model=_Fake(), checkpoint_sha256="abc")
    frames = np.random.rand(1, 25, 88, 88).astype(np.float32)
    out = backend.encode_visual(frames)
    assert out.shape == (50, 768)


def test_encode_audio_from_waveform_shape():
    import torch

    from src.features import avhubert_backend as ab

    class _Fake:
        def encode_visual_tokens(self, x):
            return torch.zeros(x.shape[0], 50, 768)

        def encode_audio_tokens(self, x):
            return torch.zeros(x.shape[0], 100, 768)

    backend = ab.AVHubertBackend(model=_Fake(), checkpoint_sha256="abc")
    wave = np.random.rand(64_000).astype(np.float32)
    out = backend.encode_audio(wave)
    assert out.shape == (100, 768)


@pytest.mark.skipif(CHECKPOINT_MISSING, reason="AV-HuBERT checkpoint not present; skipping load test")
def test_from_checkpoint_loads_real_weights():
    from src.features.avhubert_backend import AVHubertBackend

    backend = AVHubertBackend.from_checkpoint(common.AVHUBERT_CKPT_PATH)
    assert backend.embedding_dim == 768
    assert isinstance(backend.checkpoint_sha256, str) and len(backend.checkpoint_sha256) == 64
