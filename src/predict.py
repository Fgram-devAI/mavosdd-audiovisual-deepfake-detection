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
