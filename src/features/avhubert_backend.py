"""Adapter around the pretrained AV-HuBERT checkpoint (Meta / fairseq)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from src.features.mouth_crop_extract import AVHUBERT_SPEC, MouthCropSpec


class AVHubertBackend:
    embedding_dim: int = 768
    mouth_spec: MouthCropSpec = AVHUBERT_SPEC

    def __init__(self, *, model, checkpoint_sha256: str) -> None:
        self._model = model
        self.checkpoint_sha256 = checkpoint_sha256
        if hasattr(model, "eval"):
            model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint: Path) -> "AVHubertBackend":
        if not checkpoint.exists():
            raise FileNotFoundError(f"AV-HuBERT checkpoint missing: {checkpoint}")
        try:
            from fairseq import checkpoint_utils
        except ImportError as e:
            raise ImportError(
                "fairseq is required for AV-HuBERT. See "
                "report/val_eval/task0_avhubert_feasibility.md for env details."
            ) from e
        sha = _sha256(checkpoint)
        models, _cfg, _task = checkpoint_utils.load_model_ensemble_and_task([str(checkpoint)])
        return cls(model=models[0], checkpoint_sha256=sha)

    @torch.no_grad()
    def encode_visual(self, face_frames: np.ndarray) -> np.ndarray:
        arr = face_frames
        if arr.ndim == 3:
            arr = arr[None, None, ...]
        elif arr.ndim == 4:
            arr = arr[None, ...]
        tensor = torch.from_numpy(arr.astype(np.float32))
        tokens = self._model.encode_visual_tokens(tensor)
        return tokens.squeeze(0).detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_audio(self, waveform: np.ndarray) -> np.ndarray:
        arr = waveform
        if arr.ndim == 1:
            arr = arr[None, :]
        tensor = torch.from_numpy(arr.astype(np.float32))
        tokens = self._model.encode_audio_tokens(tensor)
        return tokens.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _sha256(path: Path, chunk_size: int = 1_048_576) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
