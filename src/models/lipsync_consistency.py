"""Cross-modal lip-sync consistency head: audio projector + lip BiGRU + concat similarity MLP."""
from __future__ import annotations

import torch
import torch.nn as nn


class _AudioProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim). Mean-pool over time.
        pooled = x.mean(dim=1)
        return self.proj(pooled)


class _LipProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, gru_hidden: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.proj = nn.Sequential(
            nn.Linear(2 * gru_hidden, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x.float())
        m = mask.float().unsqueeze(-1)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.proj(pooled)


class LipSyncConsistencyModel(nn.Module):
    def __init__(
        self,
        audio_dim: int = 768,
        lip_dim: int = 84,
        emb_dim: int = 128,
        gru_hidden: int = 96,
        mlp_hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.audio_encoder = _AudioProjector(audio_dim, emb_dim, dropout)
        self.lip_encoder = _LipProjector(lip_dim, emb_dim, gru_hidden, dropout)
        self.similarity = nn.Sequential(
            nn.Linear(4 * emb_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        audio: torch.Tensor,
        lips: torch.Tensor,
        lips_mask: torch.Tensor,
    ) -> torch.Tensor:
        a = self.audio_encoder(audio.float())
        v = self.lip_encoder(lips.float(), lips_mask.float())
        combined = torch.cat([torch.abs(a - v), a * v, a, v], dim=-1)
        return self.similarity(combined).squeeze(-1)

    def param_count(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)


if __name__ == "__main__":
    m = LipSyncConsistencyModel()
    n = m.param_count()
    assert n < 2_000_000, f"budget exceeded: {n:,}"
    out = m(torch.randn(2, 199, 768), torch.randn(2, 20, 84), torch.ones(2, 20))
    assert out.shape == (2,), out.shape
    print(f"OK - {n:,} trainable params | logits {tuple(out.shape)}")
