from __future__ import annotations

import torch


def test_lipsync_model_forward_returns_batched_logits():
    from src.models.lipsync_consistency import LipSyncConsistencyModel

    model = LipSyncConsistencyModel()
    audio = torch.randn(4, 199, 768)
    lips = torch.randn(4, 20, 84)
    mask = torch.ones(4, 20)

    logits = model(audio, lips, mask)

    assert logits.shape == (4,)
    assert logits.dtype == torch.float32


def test_lipsync_model_respects_mask_when_pooling_lips():
    from src.models.lipsync_consistency import LipSyncConsistencyModel

    model = LipSyncConsistencyModel()
    audio = torch.zeros(2, 199, 768)
    lips = torch.randn(2, 20, 84)
    mask_full = torch.ones(2, 20)
    mask_partial = torch.zeros(2, 20)
    mask_partial[:, :10] = 1.0

    logits_full = model(audio, lips, mask_full)
    logits_partial = model(audio, lips, mask_partial)

    assert torch.allclose(logits_full, logits_partial, atol=1e-6) is False


def test_lipsync_model_stays_under_two_million_params():
    from src.models.lipsync_consistency import LipSyncConsistencyModel

    model = LipSyncConsistencyModel()

    assert model.param_count(trainable_only=True) < 2_000_000
