"""Score external Higgsfield-style videos with video-level AV classifiers.

This is an external stress-test helper, not a training script. It extracts
pretrained SyncNet or AV-HuBERT embeddings directly from each .mp4, applies a
trained video-level AV head, and writes per-video fake scores plus temporal
length diagnostics.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src import common
from src.data.lipsync_pretrained_dataset import SYNC_FEATURE_DIM, compute_sync_features
from src.data.video_av_dataset import fixed_window
from src.features.extract_avhubert_embeddings import _load_waveform
from src.features.extract_syncnet_embeddings import _compute_mel
from src.features.avhubert_backend import AVHubertBackend
from src.features.mouth_crop_extract import AVHUBERT_SPEC, SYNCNET_SPEC, extract_mouth_crops
from src.features.syncnet_backend import SyncNetBackend
from src.models.lipsync_pretrained_head import LipSyncPretrainedHead
from src.train_lipsync_pretrained import resolve_backend


FIELDS = (
    "video_path",
    "backend",
    "score_fake",
    "label_at_threshold",
    "threshold",
    "visual_windows",
    "audio_windows",
    "shared_windows",
    "visual_audio_window_ratio",
    "status",
    "error",
)


def _load_head(checkpoint: Path, backend: str, device: torch.device) -> LipSyncPretrainedHead:
    state = torch.load(checkpoint, map_location=device)
    _visual_dir, _audio_dir, _failures_csv, embed_dim = resolve_backend(backend)
    cfg = state.get("config", {})
    head = LipSyncPretrainedHead(
        sync_feature_dim=SYNC_FEATURE_DIM,
        embed_dim=embed_dim,
        hidden=cfg.get("hidden", 128),
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    head.load_state_dict(state["state_dict"])
    head.eval()
    return head


def _load_backend(backend: str):
    if backend == "syncnet":
        return SyncNetBackend.from_checkpoint(common.SYNCNET_CKPT_PATH)
    if backend == "avhubert":
        return AVHubertBackend.from_checkpoint(common.AVHUBERT_CKPT_PATH)
    raise ValueError(f"unknown backend: {backend}")


def _extract_embeddings(video: Path, backend_name: str, backend) -> tuple[np.ndarray, np.ndarray]:
    if backend_name == "syncnet":
        visual = backend.encode_visual(extract_mouth_crops(video, SYNCNET_SPEC).astype(np.float32))
        audio = backend.encode_audio(_compute_mel(video))
        return visual, audio
    visual = backend.encode_visual(extract_mouth_crops(video, AVHUBERT_SPEC).astype(np.float32))
    audio = backend.encode_audio(_load_waveform(video))
    return visual, audio


def score_one(
    video: Path,
    *,
    backend_name: str,
    backend,
    head: LipSyncPretrainedHead,
    device: torch.device,
    threshold: float,
    window_count: int | None,
    window_policy: str,
) -> dict:
    visual, audio = _extract_embeddings(video, backend_name, backend)
    visual_raw_windows = int(visual.shape[0])
    audio_raw_windows = int(audio.shape[0])
    visual = fixed_window(visual, window_count=window_count, policy=window_policy)
    audio = fixed_window(audio, window_count=window_count, policy=window_policy)
    sync_features = compute_sync_features(visual, audio)
    pooled_visual = visual.mean(axis=0).astype(np.float32)
    pooled_audio = audio.mean(axis=0).astype(np.float32)
    with torch.no_grad():
        logit = head(
            torch.from_numpy(sync_features).unsqueeze(0).to(device),
            torch.from_numpy(pooled_visual).unsqueeze(0).to(device),
            torch.from_numpy(pooled_audio).unsqueeze(0).to(device),
        )
    score = float(torch.sigmoid(logit).cpu().item())
    return {
        "video_path": str(video),
        "backend": backend_name,
        "score_fake": f"{score:.6f}",
        "label_at_threshold": "fake" if score >= threshold else "real",
        "threshold": f"{threshold:.6f}",
        "visual_windows": visual_raw_windows,
        "audio_windows": audio_raw_windows,
        "shared_windows": min(visual_raw_windows, audio_raw_windows),
        "visual_audio_window_ratio": f"{visual_raw_windows / max(audio_raw_windows, 1):.6f}",
        "status": "ok",
        "error": "",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=Path("data/higgsfield_gen_videos"))
    parser.add_argument("--glob", default="*.mp4")
    parser.add_argument("--backend", choices=("syncnet", "avhubert"), default="syncnet")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("report/val_eval/higgsfield_video_av_scores.csv"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--window-count", type=int, default=None,
                        help="crop/pad each visual/audio embedding sequence before scoring")
    parser.add_argument("--window-policy", choices=("center", "first"), default="center")
    args = parser.parse_args(argv)

    videos = sorted(args.video_dir.glob(args.glob))
    if args.limit is not None:
        videos = videos[: args.limit]

    device = torch.device(args.device)
    head = _load_head(args.checkpoint, args.backend, device)
    backend = _load_backend(args.backend)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for video in tqdm(videos, desc=f"score {args.backend}", unit="video"):
        try:
            rows.append(score_one(
                video,
                backend_name=args.backend,
                backend=backend,
                head=head,
                device=device,
                threshold=args.threshold,
                window_count=args.window_count,
                window_policy=args.window_policy,
            ))
        except Exception as exc:
            rows.append({
                "video_path": str(video),
                "backend": args.backend,
                "score_fake": "",
                "label_at_threshold": "",
                "threshold": f"{args.threshold:.6f}",
                "visual_windows": "",
                "audio_windows": "",
                "shared_windows": "",
                "visual_audio_window_ratio": "",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        scores = np.array([float(r["score_fake"]) for r in ok])
        print(
            f"wrote {len(rows)} rows to {args.out}; ok={len(ok)} "
            f"mean_score={scores.mean():.4f} fake@threshold={(scores >= args.threshold).mean():.4f}"
        )
    else:
        print(f"wrote {len(rows)} rows to {args.out}; ok=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
