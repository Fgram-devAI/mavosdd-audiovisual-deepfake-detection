"""Tests for src/features/syncnet_backend.py.

Checkpoint-loading tests are skipped when the checkpoint file is absent so CI
stays green; they run locally once download_syncnet_checkpoint.py has been run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src import common


CHECKPOINT_MISSING = not common.SYNCNET_CKPT_PATH.exists()


def test_backend_declares_embedding_dim_and_mouth_spec():
    from src.features.syncnet_backend import SyncNetBackend
    from src.features.mouth_crop_extract import SYNCNET_SPEC

    assert SyncNetBackend.embedding_dim == 512
    assert SyncNetBackend.mouth_spec == SYNCNET_SPEC


def test_encode_visual_from_synthetic_module_returns_expected_shape(monkeypatch):
    from src.features import syncnet_backend as sb

    class _FakeSyncNet:
        def __init__(self):
            pass

        def visual_forward(self, x):
            import torch

            return torch.zeros(x.shape[0], 512)

        def audio_forward(self, x):
            import torch

            return torch.zeros(x.shape[0], 512)

    fake = _FakeSyncNet()
    backend = sb.SyncNetBackend(model=fake, checkpoint_sha256="abc")

    stacks = np.random.rand(4, 5, 3, 48, 96).astype(np.float32)
    out = backend.encode_visual(stacks)
    assert out.shape == (4, 512)
    assert out.dtype == np.float32


def test_encode_audio_from_synthetic_module_returns_expected_shape():
    import torch

    from src.features import syncnet_backend as sb

    class _FakeSyncNet:
        def visual_forward(self, x):
            return torch.zeros(x.shape[0], 512)

        def audio_forward(self, x):
            return torch.zeros(x.shape[0], 512)

    backend = sb.SyncNetBackend(model=_FakeSyncNet(), checkpoint_sha256="abc")
    mel = np.random.rand(3, 1, 80, 16).astype(np.float32)
    out = backend.encode_audio(mel)
    assert out.shape == (3, 512)


@pytest.mark.skipif(CHECKPOINT_MISSING, reason="SyncNet checkpoint not present; skipping load test")
def test_from_checkpoint_loads_real_weights():
    from src.features.syncnet_backend import SyncNetBackend

    backend = SyncNetBackend.from_checkpoint(common.SYNCNET_CKPT_PATH)
    assert backend.embedding_dim == 512
    assert isinstance(backend.checkpoint_sha256, str) and len(backend.checkpoint_sha256) == 64
