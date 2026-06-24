"""Late fusion classifier for frozen audio embeddings and lip-landmark sequences."""
from __future__ import annotations

import torch
import torch.nn as nn


class AudioHead(nn.Module):
    """(B, T, 768) frozen embeddings -> temporal mean-pool -> (B, 128)."""

    def __init__(self, in_dim: int = 768, out_dim: int = 128, p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.mean(dim=1))


class VisualHead(nn.Module):
    """(B, 20, 84) lip features + (B, 20) mask -> BiGRU -> (B, 128)."""

    def __init__(self, in_dim: int = 84, hidden: int = 128, out_dim: int = 128, p: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            in_dim,
            hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=p,
        )
        self.proj = nn.Sequential(nn.Linear(2 * hidden, out_dim), nn.ReLU(), nn.Dropout(p))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x.float())
        m = mask.float().unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.proj(pooled)


class LateFusionClassifier(nn.Module):
    def __init__(self, modality: str = "fusion", emb: int = 128, p: float = 0.3):
        super().__init__()
        if modality not in {"audio", "visual", "fusion"}:
            raise ValueError(f"Unknown modality: {modality}")
        self.modality = modality
        self.audio = AudioHead(out_dim=emb, p=p) if modality != "visual" else None
        self.visual = VisualHead(out_dim=emb, p=p) if modality != "audio" else None
        fused = emb * (2 if modality == "fusion" else 1)
        self.classifier = nn.Sequential(
            nn.Linear(fused, 64),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        audio: torch.Tensor,
        lips: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = []
        if self.audio is not None:
            parts.append(self.audio(audio.float()))
        if self.visual is not None:
            if lips is None or mask is None:
                raise ValueError(
                    f"modality={self.modality!r} requires both 'lips' and 'mask'"
                )
            parts.append(self.visual(lips.float(), mask.float()))
        return self.classifier(torch.cat(parts, dim=-1)).squeeze(-1)


if __name__ == "__main__":
    model = LateFusionClassifier("audio")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params < 2_000_000, f"C5 violated: {n_params:,} params"
    logits = model(torch.randn(8, 199, 768))
    assert logits.shape == (8,), logits.shape
    print(f"OK - {n_params:,} trainable params | logits {tuple(logits.shape)}")
