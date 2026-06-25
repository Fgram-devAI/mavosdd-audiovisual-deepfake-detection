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
import pickle
import sys
import tempfile
from pathlib import Path

import mediapipe as mp
import numpy as np
import torch

from src.data import codec_match_audio
from src.features import audio_backends, audio_io, extract_lips
from src.models.late_fusion import LateFusionClassifier

_INFERENCE_CODEC_SR = 44100
_INFERENCE_CODEC_BR = "128k"
_AUDIO_SR = 16000
_AUDIO_SECONDS = 4.0
_LIP_FRAMES = 20


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


def _reconstruct_norm_stats(norm_stats_dict: dict) -> dict:
    def _as_f32(v):
        return np.asarray(v, dtype=np.float32) if v is not None else None
    return {
        "audio_mean": _as_f32(norm_stats_dict.get("audio_mean")),
        "audio_std": _as_f32(norm_stats_dict.get("audio_std")),
        "lips_mean": _as_f32(norm_stats_dict.get("lips_mean")),
        "lips_std": _as_f32(norm_stats_dict.get("lips_std")),
        "eps": float(norm_stats_dict.get("eps", 1e-6)),
    }


def predict_video(
    video: str | Path,
    checkpoint: str | Path,
    *,
    device: str = "auto",
    threshold: float = 0.5,
    codec_match: bool = True,
) -> dict:
    video = Path(video)
    checkpoint = Path(checkpoint)

    # 4a. Resolve device + load checkpoint.
    dev = _resolve_device(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _validate_checkpoint(ckpt)
    modality = ckpt["modality"]
    backend_name = ckpt["backend"]

    if dev.type == "cpu" and modality != "visual":
        print(
            f"warning: running on CPU; first call also downloads the "
            f"{backend_name} weights",
            file=sys.stderr,
        )

    norm = _reconstruct_norm_stats(ckpt["norm_stats"])
    codec_matched = _should_codec_match(ckpt, codec_match)

    audio_tensor: torch.Tensor | None = None
    lips_tensor: torch.Tensor | None = None
    mask_tensor: torch.Tensor | None = None
    lip_frames_present: int | None = None
    face_detected: bool | None = None

    with tempfile.TemporaryDirectory(prefix="predict-") as td:
        tmpdir = Path(td)
        # 4b. Demux audio (skip for visual).
        if modality != "visual":
            tmp_wav = tmpdir / "audio.wav"
            try:
                codec_match_audio.decode_to_wav16k(video, tmp_wav)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "ffmpeg not found on PATH; install ffmpeg "
                    "(e.g. 'brew install ffmpeg' / 'apt-get install ffmpeg')"
                ) from exc
            except RuntimeError as exc:
                raise RuntimeError(
                    f"could not read video/audio from {video}: {exc}"
                ) from exc

            # 4c. Codec-match round-trip (conditional).
            if codec_matched:
                tmp_mp3 = tmpdir / "audio.mp3"
                tmp_wav_cm = tmpdir / "audio_cm.wav"
                codec_match_audio.encode_mp3(
                    tmp_wav, tmp_mp3, _INFERENCE_CODEC_SR, _INFERENCE_CODEC_BR,
                )
                codec_match_audio.decode_to_wav16k(tmp_mp3, tmp_wav_cm)
                window_src = tmp_wav_cm
            else:
                window_src = tmp_wav

            wave = audio_io.load_audio_window(
                window_src, sr=_AUDIO_SR, seconds=_AUDIO_SECONDS,
            )

            # 4d. Frozen backend embedding.
            backend = audio_backends.load_backend(backend_name, dev)
            emb = backend.encode(wave)  # (T, 768)
            emb = _normalize(emb, norm["audio_mean"], norm["audio_std"], norm["eps"])
            audio_tensor = torch.from_numpy(np.asarray(emb, dtype=np.float32)) \
                .unsqueeze(0).to(dev)

        # 4e. Lip landmarks (skip for audio).
        if modality != "audio":
            mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=0.5,
            )
            try:
                feats, mask = extract_lips.extract_one(str(video), mesh)
            finally:
                mesh.close()
            feats = _normalize(feats, norm["lips_mean"], norm["lips_std"], norm["eps"])
            lips_tensor = torch.from_numpy(np.asarray(feats, dtype=np.float32)) \
                .unsqueeze(0).to(dev)
            mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32)) \
                .unsqueeze(0).to(dev)
            lip_frames_present = int(mask.sum())
            face_detected = lip_frames_present > 0

    # 4g. Build the model.
    hp = ckpt["model_hparams"]
    model = LateFusionClassifier(
        modality=modality, emb=hp.get("emb", 128), p=hp.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()

    # 4h. Forward → logit → prob.
    with torch.no_grad():
        if modality == "audio":
            logits = model(audio_tensor)
        elif modality == "visual":
            logits = model(None, lips_tensor, mask_tensor)
        else:  # fusion
            logits = model(audio_tensor, lips_tensor, mask_tensor)
    prob = float(torch.sigmoid(logits).cpu().item())

    label = "spoof" if prob >= threshold else "bonafide"

    return {
        "video": str(video),
        "checkpoint": checkpoint.name,
        "modality": modality,
        "backend": backend_name if modality != "visual" else None,
        "audio_window_s": _AUDIO_SECONDS if modality != "visual" else None,
        "codec_matched": codec_matched if modality != "visual" else None,
        "lip_frames_present": lip_frames_present,
        "lip_frames_total": _LIP_FRAMES if modality != "audio" else None,
        "face_detected": face_detected,
        "p_spoof": prob,
        "threshold": float(threshold),
        "label": label,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = predict_video(
            args.video,
            args.checkpoint,
            device=args.device,
            threshold=args.threshold,
            codec_match=not args.no_codec_match,
        )
    except (ValueError, RuntimeError, OSError, pickle.UnpicklingError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(_render_json(result) if args.json else _render_human(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
