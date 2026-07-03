"""Adapter around the pretrained AV-HuBERT checkpoint (Meta / fairseq)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

from src.features.mouth_crop_extract import AVHUBERT_SPEC, MouthCropSpec


AVHUBERT_AUDIO_STACK_ORDER = 4


def _register_avhubert_modules() -> None:
    """Register AV-HuBERT's custom Fairseq task/model modules.

    The editable Fairseq install only registers built-in Fairseq tasks. The
    AV-HuBERT checkpoint needs the sibling repo's `avhubert` package imported
    so `av_hubert_pretraining` and `av_hubert` land in Fairseq registries.
    """
    import fairseq  # noqa: F401

    candidates = [
        Path(__file__).resolve().parents[3] / "av_hubert",
        Path(__file__).resolve().parents[4] / "av_hubert",
        Path.home() / "Documents" / "Projects" / "av_hubert",
    ]
    for candidate in candidates:
        if (candidate / "avhubert" / "__init__.py").exists():
            if str(candidate) not in sys.path:
                # Append, do not prepend: the repo root also has a `fairseq/`
                # directory that can shadow the editable Fairseq package.
                sys.path.append(str(candidate))
            break

    added_argv = False
    if len(sys.argv) == 1:
        # AV-HuBERT's modules switch into a local debug import path when
        # len(sys.argv) == 1. Avoid that branch for `python - <<'PY'` probes.
        sys.argv.append("avhubert_import")
        added_argv = True
    try:
        import avhubert.hubert_pretraining  # noqa: F401
        import avhubert.hubert  # noqa: F401
    finally:
        if added_argv:
            sys.argv.pop()


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
        _register_avhubert_modules()
        sha = _sha256(checkpoint)
        models, _cfg, _task = checkpoint_utils.load_model_ensemble_and_task([str(checkpoint)])
        return cls(model=models[0], checkpoint_sha256=sha)

    @torch.no_grad()
    def encode_visual(self, face_frames: np.ndarray) -> np.ndarray:
        tensor = _prepare_visual_tensor(face_frames)
        tokens, _padding = self._model.extract_finetune(
            {"audio": None, "video": tensor},
            mask=False,
        )
        return tokens.squeeze(0).detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_audio(self, waveform: np.ndarray) -> np.ndarray:
        tensor = _prepare_audio_tensor(waveform)
        tokens, _padding = self._model.extract_finetune(
            {"audio": tensor, "video": None},
            mask=False,
        )
        return tokens.squeeze(0).detach().cpu().numpy().astype(np.float32)


def waveform_to_avhubert_features(
    waveform: np.ndarray,
    *,
    sample_rate: int = 16_000,
    stack_order: int = AVHUBERT_AUDIO_STACK_ORDER,
) -> np.ndarray:
    """Convert mono waveform samples to AV-HuBERT's stacked log-fbank features.

    The released AV-HuBERT checkpoints expect 26-bin logfbank features stacked
    four frames at a time, i.e. ``[T, 104]``. They do not accept raw waveforms.
    """
    from python_speech_features import logfbank

    feats = logfbank(np.asarray(waveform, dtype=np.float32), samplerate=sample_rate).astype(np.float32)
    feat_dim = feats.shape[1]
    remainder = len(feats) % stack_order
    if remainder:
        pad = np.zeros((stack_order - remainder, feat_dim), dtype=feats.dtype)
        feats = np.concatenate([feats, pad], axis=0)
    return feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)


def _prepare_audio_tensor(audio: np.ndarray) -> torch.Tensor:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        arr = waveform_to_avhubert_features(arr)
    if arr.ndim == 2:
        if arr.shape[1] != 104:
            raise ValueError(f"AV-HuBERT audio features must have 104 dims, got {arr.shape}")
        arr = arr.T[None, ...]
    elif arr.ndim == 3:
        if arr.shape[1] == 104:
            pass
        elif arr.shape[2] == 104:
            arr = np.transpose(arr, (0, 2, 1))
        else:
            raise ValueError(f"AV-HuBERT batched audio features must include 104 dims, got {arr.shape}")
    else:
        raise ValueError(f"AV-HuBERT audio input must be waveform or fbank features, got {arr.shape}")
    return torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))


def _prepare_visual_tensor(face_frames: np.ndarray) -> torch.Tensor:
    arr = np.asarray(face_frames, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None, None, ...]
    elif arr.ndim == 4:
        if arr.shape[-1] == 1:
            arr = np.transpose(arr, (3, 0, 1, 2))[None, ...]
        else:
            arr = arr[:, None, ...]
    elif arr.ndim == 5:
        if arr.shape[2] == 1:
            arr = np.transpose(arr, (0, 2, 1, 3, 4))
        elif arr.shape[1] == 1:
            pass
        else:
            raise ValueError(f"AV-HuBERT visual input must be grayscale, got {arr.shape}")
    else:
        raise ValueError(f"AV-HuBERT visual input must be a grayscale crop stack, got {arr.shape}")
    return torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))


def _sha256(path: Path, chunk_size: int = 1_048_576) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
