"""Single-video real/fake scoring CLI (Phase 6 — final demo surface).

Takes one raw .mp4 and one src/train.py checkpoint and emits a single
bonafide/spoof probability. Runs the full feature-extraction pipeline live
(audio demux + optional codec round-trip + frozen backend embedding;
MediaPipe lip landmarks) using the same code paths the training pipeline
already ships. Reads no manifests, writes nothing under data/models/runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import mediapipe as mp
import numpy as np
import torch

from src.data import codec_match_audio
from src.features import audio_backends, audio_io, extract_lips
from src.models.late_fusion import LateFusionClassifier


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="predict",
        description=(
            "Score a single raw .mp4 with a trained src/train.py checkpoint "
            "(audio / visual / fusion). Emits a bonafide/spoof probability."
        ),
    )
    p.add_argument("--video", required=True, type=Path,
                   help="Path to the input .mp4.")
    p.add_argument("--checkpoint", required=True, type=Path,
                   help=".pt produced by src/train.py.")
    p.add_argument("--device", default="auto",
                   choices=("auto", "cuda", "mps", "cpu"),
                   help="auto → cuda > mps > cpu.")
    p.add_argument("--json", action="store_true",
                   help="Emit a flat JSON dict instead of the human line.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="p_spoof >= threshold → label 'spoof', else 'bonafide'.")
    p.add_argument("--no-codec-match", dest="no_codec_match",
                   action="store_true",
                   help="Force-disable the MP3 codec round-trip.")
    return p


_REQUIRED_CKPT_FIELDS = ("state_dict", "modality", "backend",
                         "model_hparams", "norm_stats")
_ALLOWED_MODALITIES = {"audio", "visual", "fusion"}


def _validate_checkpoint(ckpt: dict) -> None:
    missing = [k for k in _REQUIRED_CKPT_FIELDS if k not in ckpt]
    if missing:
        raise ValueError(
            f"checkpoint missing required field(s): {missing}; "
            f"not a src/train.py checkpoint"
        )
    modality = ckpt["modality"]
    if modality not in _ALLOWED_MODALITIES:
        raise ValueError(
            f"checkpoint modality {modality!r} not in {sorted(_ALLOWED_MODALITIES)}"
        )
    if modality != "visual" and ckpt["backend"] is None:
        raise ValueError(
            f"checkpoint modality={modality!r} requires non-None 'backend'; "
            f"got None"
        )


def _should_codec_match(ckpt: dict, codec_match_flag: bool) -> bool:
    if ckpt.get("modality") == "visual":
        return False
    if not codec_match_flag:
        return False
    audio_dir = ckpt.get("audio_dir")
    if audio_dir is None:
        return False
    return Path(audio_dir).name.endswith("_codec")


def _normalize(
    x: np.ndarray,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    eps: float,
) -> np.ndarray:
    if mean is None or std is None:
        return x
    return (x - mean) / (std + eps)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def _render_json(result: dict) -> str:
    return json.dumps(result, default=_json_default)


def _render_human(result: dict) -> str:
    modality = result["modality"]
    parts: list[str] = [
        f"checkpoint={result['checkpoint']}",
        f"modality={modality}",
    ]
    if modality != "visual":
        parts.append(f"backend={result['backend']}")
        parts.append(f"audio_window={result['audio_window_s']}s")
        parts.append(f"codec_matched={result['codec_matched']}")
    if modality != "audio":
        parts.append(
            f"lip_frames_present={result['lip_frames_present']}/{result['lip_frames_total']}"
        )
        parts.append(f"face_detected={result['face_detected']}")
    parts.append(f"p_spoof={result['p_spoof']:.4f}")
    parts.append(f"label={result['label']}")
    parts.append(f"threshold={result['threshold']}")
    return " ".join(parts)


def predict_video(
    video: str | Path,
    checkpoint: str | Path,
    *,
    device: str = "auto",
    threshold: float = 0.5,
    codec_match: bool = True,
) -> dict:
    raise NotImplementedError("predict_video pipeline arrives in later tasks")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = predict_video(
        args.video,
        args.checkpoint,
        device=args.device,
        threshold=args.threshold,
        codec_match=not args.no_codec_match,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
