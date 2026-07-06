"""Score external generated videos with all final-fusion heads.

For every ``.mp4`` under ``--input-dir``, extract audio + SyncNet + AV-HuBERT
features once, run the audio anti-spoof head, the sync-consistency head, the
video-level AV head, and the logistic fusion, then write the per-video CSV and
a human-readable summary. External batches are positive-only stress tests
unless paired real controls are supplied — the summary spells that out.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src import common
from src.data.build_final_fusion_scores import (
    _load_audio_head, _load_sync_head, _load_video_av_head,
    _resolve_audio_ckpt, DEFAULT_SYNC_CKPT, DEFAULT_VIDEO_AV_CKPT,
)
from src.data.lipsync_pretrained_dataset import compute_sync_features
from src.data.video_av_dataset import fixed_window
from src.features.audio_io import load_audio_window
from src.features.audio_backends import load_backend as _load_audio_ssl
from src.features.avhubert_backend import AVHubertBackend
from src.features.extract_avhubert_embeddings import _load_waveform as _load_avhubert_waveform
from src.features.extract_syncnet_embeddings import _compute_mel
from src.features.mouth_crop_extract import AVHUBERT_SPEC, SYNCNET_SPEC, extract_mouth_crops
from src.features.syncnet_backend import SyncNetBackend
from src.models.final_fusion import FinalFusionLogReg


def _audio_backend_label(audio: dict) -> str:
    """Return a short label for the audio anti-spoof backend.

    External head dicts carry ``backend`` directly (e.g. ``"wavlm"``).
    The real head dicts from ``_load_audio_head`` also have ``audio_dir``; we
    prefer ``backend`` here so the external batch script works with both real
    and monkeypatched dicts.
    """
    return str(audio.get("backend", ""))


def _video_av_backend_label(video_av: dict) -> str:
    """Return a short label for the video-AV backend, including window count."""
    name = str(video_av.get("backend", ""))
    wc = video_av.get("window_count")
    if wc:
        return f"{name}_fixed{wc}"
    return name


FIELDS = (
    "video_path", "batch_name",
    "audio_backend", "video_av_backend", "sync_backend",
    "audio_fake_score", "sync_inconsistent_score", "video_av_fake_score",
    "final_fusion_score", "label_at_threshold", "threshold",
    "status", "error",
)


def load_all(
    *, audio_ckpt: Path | None, sync_ckpt: Path, video_av_ckpt: Path,
    fusion_ckpt: Path, device: str,
):
    resolved_audio_ckpt = _resolve_audio_ckpt(audio_ckpt)
    audio = _load_audio_head(resolved_audio_ckpt, device)
    audio["encoder"] = _load_audio_ssl(audio["backend"], torch.device(device))
    sync = _load_sync_head(sync_ckpt, device)
    video_av = _load_video_av_head(video_av_ckpt, device)
    fusion = FinalFusionLogReg.load(fusion_ckpt)
    return audio, sync, video_av, fusion


def _sync_backend_object(name: str):
    if name == "syncnet":
        return SyncNetBackend.from_checkpoint(common.SYNCNET_CKPT_PATH)
    if name == "avhubert":
        return AVHubertBackend.from_checkpoint(common.AVHUBERT_CKPT_PATH)
    raise ValueError(f"unknown backend: {name}")


def _extract_backend_pair(video: Path, backend_name: str, backend) -> tuple[np.ndarray, np.ndarray]:
    if backend_name == "syncnet":
        visual = backend.encode_visual(extract_mouth_crops(video, SYNCNET_SPEC).astype(np.float32))
        audio = backend.encode_audio(_compute_mel(video))
        return visual, audio
    visual = backend.encode_visual(extract_mouth_crops(video, AVHUBERT_SPEC).astype(np.float32))
    audio = backend.encode_audio(_load_avhubert_waveform(video))
    return visual, audio


def _audio_head_score(video: Path, audio: dict, device: torch.device) -> float:
    backend_name = audio["backend"]  # plain SSL backend name, e.g. "wavlm"
    wav = load_audio_window(video, sr=common.SR, seconds=common.AUDIO_SECONDS)
    ssl = _load_audio_ssl(backend_name, device)
    feats = ssl.encode(wav).astype(np.float32)
    ns = audio.get("norm_stats") or {}
    mean = ns.get("audio_mean")
    std = ns.get("audio_std")
    if mean is not None and std is not None:
        feats = (feats - mean) / np.maximum(std, float(ns.get("eps", 1e-6)))
    if feats.ndim == 2:
        feats = feats[None]
    with torch.no_grad():
        logit = audio["model"](audio=torch.from_numpy(feats).to(device))
    return float(torch.sigmoid(logit).cpu().item())


def _head_score(head: dict, visual: np.ndarray, audio: np.ndarray, device: torch.device,
                window_count: int | None = None, window_policy: str = "center") -> float:
    v = fixed_window(visual, window_count=window_count, policy=window_policy)
    a = fixed_window(audio, window_count=window_count, policy=window_policy)
    sync_features = compute_sync_features(v, a)
    pooled_v = v.mean(axis=0).astype(np.float32)
    pooled_a = a.mean(axis=0).astype(np.float32)
    with torch.no_grad():
        logit = head["model"](
            torch.from_numpy(sync_features).unsqueeze(0).to(device),
            torch.from_numpy(pooled_v).unsqueeze(0).to(device),
            torch.from_numpy(pooled_a).unsqueeze(0).to(device),
        )
    return float(torch.sigmoid(logit).cpu().item())


def extract_head_scores(
    video: Path, *, audio: dict, sync: dict, video_av: dict,
    sync_backend_obj, video_av_backend_obj, device: torch.device,
) -> dict:
    v_sync, a_sync = _extract_backend_pair(video, sync["backend"], sync_backend_obj)
    if video_av["backend"] == sync["backend"]:
        v_vav, a_vav = v_sync, a_sync
    else:
        v_vav, a_vav = _extract_backend_pair(video, video_av["backend"], video_av_backend_obj)
    return {
        "audio_fake_score": _audio_head_score(video, audio, device),
        "sync_inconsistent_score": _head_score(sync, v_sync, a_sync, device),
        "video_av_fake_score": _head_score(video_av, v_vav, a_vav, device,
                                           window_count=video_av["window_count"],
                                           window_policy=video_av["window_policy"]),
    }


def _score_row(video: Path, batch_name: str, scored: dict, audio: dict, sync: dict,
               video_av: dict, fusion: FinalFusionLogReg, threshold: float) -> dict:
    feature_lookup = {
        "audio_fake_score": scored["audio_fake_score"],
        "video_av_fake_score": scored["video_av_fake_score"],
        "sync_inconsistent_score": scored["sync_inconsistent_score"],
    }
    x = np.asarray([[feature_lookup[c] for c in fusion.feature_columns]], dtype=np.float32)
    fused = float(fusion.predict_proba(x)[0])
    return {
        "video_path": str(video),
        "batch_name": batch_name,
        "audio_backend": _audio_backend_label(audio),
        "video_av_backend": _video_av_backend_label(video_av),
        "sync_backend": sync["backend"],
        "audio_fake_score": f"{scored['audio_fake_score']:.6f}",
        "sync_inconsistent_score": f"{scored['sync_inconsistent_score']:.6f}",
        "video_av_fake_score": f"{scored['video_av_fake_score']:.6f}",
        "final_fusion_score": f"{fused:.6f}",
        "label_at_threshold": "fake" if fused >= threshold else "real",
        "threshold": f"{threshold:.6f}",
        "status": "ok",
        "error": "",
    }


def _failure_row(video: Path, batch_name: str, audio, sync, video_av, threshold: float,
                 exc: Exception) -> dict:
    return {
        "video_path": str(video),
        "batch_name": batch_name,
        "audio_backend": _audio_backend_label(audio) if audio else "",
        "video_av_backend": _video_av_backend_label(video_av) if video_av else "",
        "sync_backend": sync["backend"] if sync else "",
        "audio_fake_score": "",
        "sync_inconsistent_score": "",
        "video_av_fake_score": "",
        "final_fusion_score": "",
        "label_at_threshold": "",
        "threshold": f"{threshold:.6f}",
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
    }


def _write_summary(md_path: Path, batch_name: str, csv_path: Path, rows: list[dict],
                   threshold: float) -> None:
    n = len(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    fakes = [r for r in ok if r["label_at_threshold"] == "fake"]
    hit_rate = (len(fakes) / len(ok)) if ok else float("nan")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        f"# Generated-video batch summary: `{batch_name}`\n\n"
        f"- Videos scored: {n} (ok={len(ok)}, failed={n - len(ok)}).\n"
        f"- Detection hit rate (positive-only): {hit_rate:.4f} at threshold {threshold:.4f}.\n"
        f"- Scores CSV: `{csv_path}`.\n\n"
        f"## Interpretation\n\n"
        f"- This batch is **positive-only** unless a paired real-control folder is supplied. "
        f"Detection rate is a **hit rate**, not accuracy.\n"
        f"- Specificity cannot be measured without real controls.\n"
        f"- The threshold was imported from the MAVOS-DD val EER-selected threshold and may be "
        f"miscalibrated on external generators.\n"
        f"- The `sync_inconsistent_score` column is a **diagnostic** consistency feature, not a "
        f"generated-video decision by itself. **sync-consistency is not deepfake detection.**\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=common.HIGGSFIELD_GEN_VIDEOS_DIR)
    parser.add_argument("--glob", default="*.mp4")
    parser.add_argument("--batch-name", required=True)
    parser.add_argument("--audio-ckpt", type=Path, default=None)
    parser.add_argument("--sync-ckpt", type=Path, default=DEFAULT_SYNC_CKPT)
    parser.add_argument("--video-av-ckpt", type=Path, default=DEFAULT_VIDEO_AV_CKPT)
    parser.add_argument("--fusion-ckpt", type=Path, default=common.CKPT_FINAL_FUSION_LOGREG)
    parser.add_argument("--out", type=Path, default=common.GENERATED_VIDEO_BATCH_SCORES)
    parser.add_argument("--summary", type=Path, default=common.GENERATED_VIDEO_BATCH_SUMMARY)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    audio, sync, video_av, fusion = load_all(
        audio_ckpt=args.audio_ckpt, sync_ckpt=args.sync_ckpt,
        video_av_ckpt=args.video_av_ckpt, fusion_ckpt=args.fusion_ckpt,
        device=args.device,
    )
    threshold = float(fusion.threshold)
    device = torch.device(args.device)

    videos = sorted(Path(args.input_dir).glob(args.glob))
    if args.limit is not None:
        videos = videos[: args.limit]

    sync_backend_obj = _sync_backend_object(sync["backend"])
    if video_av["backend"] == sync["backend"]:
        video_av_backend_obj = sync_backend_obj
    else:
        video_av_backend_obj = _sync_backend_object(video_av["backend"])

    rows: list[dict] = []
    for video in tqdm(videos, desc=f"score {args.batch_name}", unit="video"):
        try:
            scored = extract_head_scores(
                video, audio=audio, sync=sync, video_av=video_av,
                sync_backend_obj=sync_backend_obj,
                video_av_backend_obj=video_av_backend_obj,
                device=device,
            )
            rows.append(_score_row(video, args.batch_name, scored, audio, sync,
                                   video_av, fusion, threshold))
        except Exception as exc:
            rows.append(_failure_row(video, args.batch_name, audio, sync, video_av,
                                     threshold, exc))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    _write_summary(args.summary, args.batch_name, args.out, rows, threshold)
    print(f"wrote {len(rows)} rows to {args.out}; summary at {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
