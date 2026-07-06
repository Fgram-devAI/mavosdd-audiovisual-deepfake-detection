"""Small consistency MLP over sync features + pooled AV embeddings (< 500K params)."""
from __future__ import annotations

import torch
from torch import nn


class LipSyncPretrainedHead(nn.Module):
    def __init__(
        self,
        *,
        sync_feature_dim: int,
        embed_dim: int,
        hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.sync_feature_dim = sync_feature_dim
        self.embed_dim = embed_dim
        input_dim = sync_feature_dim + 4 * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        sync_features: torch.Tensor,
        pooled_visual: torch.Tensor,
        pooled_audio: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([
            sync_features,
            torch.abs(pooled_visual - pooled_audio),
            pooled_visual * pooled_audio,
            pooled_visual,
            pooled_audio,
        ], dim=-1)
        return self.mlp(combined).squeeze(-1)

    def param_count(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)


def _smoke() -> None:
    for d in (512, 768):
        head = LipSyncPretrainedHead(sync_feature_dim=7, embed_dim=d)
        n = head.param_count()
        assert n < 500_000, f"budget exceeded at dim {d}: {n:,}"
        out = head(torch.randn(2, 7), torch.randn(2, d), torch.randn(2, d))
        assert out.shape == (2,), out.shape


if __name__ == "__main__":
    _smoke()
    print("OK")
