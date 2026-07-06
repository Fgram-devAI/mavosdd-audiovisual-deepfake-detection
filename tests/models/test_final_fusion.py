from pathlib import Path

import numpy as np
import pytest
import torch


def test_rule_baselines():
    from src.models.final_fusion import (
        rule_score_audio_only,
        rule_score_max_audio_video_av,
        rule_score_max_available,
        rule_score_sync_only,
        rule_score_video_av_only,
    )
    row = {"audio_fake_score": 0.6, "video_av_fake_score": 0.4,
           "sync_inconsistent_score": 0.9}
    assert rule_score_audio_only(row) == 0.6
    assert rule_score_video_av_only(row) == 0.4
    assert rule_score_sync_only(row) == 0.9
    assert rule_score_max_audio_video_av(row) == 0.6
    assert rule_score_max_available(row) == 0.9


def test_max_available_ignores_missing():
    from src.models.final_fusion import rule_score_max_available
    row = {"audio_fake_score": None, "video_av_fake_score": 0.4,
           "sync_inconsistent_score": None}
    assert rule_score_max_available(row) == 0.4
    row2 = {"audio_fake_score": None, "video_av_fake_score": None,
            "sync_inconsistent_score": None}
    with pytest.raises(ValueError):
        rule_score_max_available(row2)


def test_mlp_forward_and_param_budget():
    from src.models.final_fusion import FinalFusionMLP

    mlp = FinalFusionMLP(input_dim=3, hidden=32, dropout=0.3)
    n = mlp.param_count()
    assert n < 50_000, f"mlp budget exceeded: {n:,}"
    x = torch.randn(4, 3)
    logits = mlp(x)
    assert logits.shape == (4,)


def test_logreg_roundtrip(tmp_path: Path):
    from src.models.final_fusion import FinalFusionLogReg

    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(np.int64)
    ckpt = FinalFusionLogReg.fit(
        X_train=X, y_train=y, X_val=X, y_val=y,
        feature_columns=("audio_fake_score", "video_av_fake_score", "sync_inconsistent_score"),
    )
    out = tmp_path / "logreg.pt"
    ckpt.save(out)
    loaded = FinalFusionLogReg.load(out)
    assert loaded.feature_columns == ckpt.feature_columns
    np.testing.assert_allclose(loaded.coef, ckpt.coef, rtol=1e-6)
    p1 = ckpt.predict_proba(X)
    p2 = loaded.predict_proba(X)
    np.testing.assert_allclose(p1, p2, rtol=1e-6)
    assert 0.0 <= float(ckpt.threshold) <= 1.0
