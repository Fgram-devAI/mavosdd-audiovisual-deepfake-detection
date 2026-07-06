"""Tests for src/models/lipsync_pretrained_head.py."""
from __future__ import annotations

import pytest
import torch


def test_forward_returns_logits_of_batch_shape():
    from src.models.lipsync_pretrained_head import LipSyncPretrainedHead

    head = LipSyncPretrainedHead(sync_feature_dim=7, embed_dim=512)
    sf = torch.randn(4, 7)
    v = torch.randn(4, 512)
    a = torch.randn(4, 512)
    logits = head(sf, v, a)
    assert logits.shape == (4,)


def test_param_count_under_budget_for_syncnet_dims():
    from src.models.lipsync_pretrained_head import LipSyncPretrainedHead

    head = LipSyncPretrainedHead(sync_feature_dim=7, embed_dim=512)
    assert head.param_count() < 500_000


def test_param_count_under_budget_for_avhubert_dims():
    from src.models.lipsync_pretrained_head import LipSyncPretrainedHead

    head = LipSyncPretrainedHead(sync_feature_dim=7, embed_dim=768)
    assert head.param_count() < 500_000


def test_module_smoke_runs_via_module_main():
    from src.models import lipsync_pretrained_head as m

    m._smoke()
