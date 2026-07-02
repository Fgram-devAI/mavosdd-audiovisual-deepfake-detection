"""Adapter around the pretrained SyncNet (Prajwal/Wav2Lip lineage) checkpoint.

Architecture pinned to the SyncNet expert from Rudrabha/Wav2Lip
(https://github.com/Rudrabha/Wav2Lip, commit ``18f5b0d`` — file
``models/syncnet.py``, class ``SyncNet_color``). The class ``_SyncNetColor``
below is a byte-faithful copy of that upstream definition, so
``load_state_dict(strict=True)`` succeeds on the released expert weights.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.features.mouth_crop_extract import SYNCNET_SPEC, MouthCropSpec

# ---------------------------------------------------------------------------
# Vendored architecture — Rudrabha/Wav2Lip @ 18f5b0d (MIT license).
#   _Conv2d: models/conv.py::Conv2d
#   _SyncNetColor: models/syncnetv2.py::SyncNet_color
# Copied verbatim so torch.load_state_dict(strict=True) succeeds on the
# released SyncNet expert weights. If a future weight release changes tensor
# shapes, re-pin the upstream commit and re-copy these classes.
# ---------------------------------------------------------------------------


class _Conv2d(nn.Module):
    """Wav2Lip Conv2d helper: Conv+BN+ReLU with optional residual add.

    Copied verbatim from Rudrabha/Wav2Lip models/conv.py to preserve state_dict
    key names (``conv_block.0.weight`` / ``conv_block.1.weight`` etc.).
    """

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out = out + x
        return self.act(out)


class _SyncNetColor(nn.Module):
    """Byte-faithful copy of Wav2Lip's SyncNet_color expert.

    Shape contract (from upstream):
      - visual_forward:  (B, 15, 48, 96) — 5 stacked BGR mouth frames concat on
        channels; input is the lower-half mouth crop at 48x96 -> (B, 512)
      - audio_forward:   (B, 1, 80, 16) log-mel window -> (B, 512)
    """

    def __init__(self) -> None:
        super().__init__()
        self.face_encoder = nn.Sequential(
            _Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),
            _Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            _Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            _Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            _Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            _Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            _Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            _Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            _Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            _Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            _Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            _Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )
        self.audio_encoder = nn.Sequential(
            _Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            _Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            _Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            _Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            _Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            _Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            _Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            _Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            _Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            _Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            _Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            _Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            _Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

    def visual_forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.face_encoder(x)
        emb = emb.view(emb.size(0), -1)
        return nn.functional.normalize(emb, p=2, dim=1)

    def audio_forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.audio_encoder(x)
        emb = emb.view(emb.size(0), -1)
        return nn.functional.normalize(emb, p=2, dim=1)


class SyncNetBackend:
    embedding_dim: int = 512
    mouth_spec: MouthCropSpec = SYNCNET_SPEC

    def __init__(self, *, model: nn.Module | _SyncNetColor, checkpoint_sha256: str) -> None:
        self._model = model
        self.checkpoint_sha256 = checkpoint_sha256
        if hasattr(model, "eval"):
            model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint: Path) -> "SyncNetBackend":
        if not checkpoint.exists():
            raise FileNotFoundError(f"SyncNet checkpoint missing: {checkpoint}")
        sha = _sha256(checkpoint)
        state = torch.load(checkpoint, map_location="cpu")
        model = _SyncNetColor()
        raw = state.get("state_dict", state) if isinstance(state, dict) else state
        clean = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                 for k, v in raw.items()}
        model.load_state_dict(clean, strict=True)
        return cls(model=model, checkpoint_sha256=sha)

    @torch.no_grad()
    def encode_visual(self, mouth_stacks: np.ndarray) -> np.ndarray:
        n, stack, c, h, w = mouth_stacks.shape
        flat = mouth_stacks.reshape(n, stack * c, h, w).astype(np.float32)
        tensor = torch.from_numpy(flat)
        out = self._model.visual_forward(tensor)
        return out.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_audio(self, mel: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(mel.astype(np.float32))
        out = self._model.audio_forward(tensor)
        return out.detach().cpu().numpy().astype(np.float32)


def _sha256(path: Path, chunk_size: int = 1_048_576) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
