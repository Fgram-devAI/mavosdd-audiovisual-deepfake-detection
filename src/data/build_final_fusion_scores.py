"""Build the per-video final-fusion score table for MAVOS-DD train/val rows.

Iterates a video-level manifest, loads three pretrained heads (audio anti-spoof,
sync consistency, video-level AV fake), and writes one score row per sample.
Test split access is refused. Visual-frame scores are notebook-only in this
branch and left blank.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src import common
from src.data.feature_store import AUDIO_BACKEND_DIRS
from src.data.lipsync_pretrained_dataset import SYNC_FEATURE_DIM, compute_sync_features
from src.data.video_av_dataset import fixed_window
from src.models.late_fusion import LateFusionClassifier
from src.models.lipsync_pretrained_head import LipSyncPretrainedHead
from src.train_lipsync_pretrained import resolve_backend as resolve_sync_backend


FIELDS = (
    "sample_id",
    "source_video_id",
    "split",
    "source_folder",
    "final_label_binary",
    "audio_fake_score",
    "audio_backend",
    "video_av_fake_score",
    "video_av_backend",
    "sync_inconsistent_score",
    "sync_backend",
    "visual_fake_score",
    "missing_features",
)


AUDIO_CKPT_FALLBACKS = (
    common.CKPT_DIR / "best_audio_wavlm_normalized.pt",
    common.CKPT_DIR / "best_audio_hubert_normalized.pt",
    common.CKPT_DIR / "best_audio_wav2vec2_normalized.pt",
    common.CKPT_DIR / "best_audio_wavlm.pt",
    common.CKPT_DIR / "best_audio_hubert.pt",
    common.CKPT_DIR / "best_audio_wav2vec2.pt",
)
DEFAULT_SYNC_CKPT = common.CKPT_DIR / "best_lipsync_syncnet.pt"
DEFAULT_VIDEO_AV_CKPT = common.CKPT_DIR / "best_video_av_avhubert_fixed25.pt"


def _final_label_binary(source_folder: str) -> str:
    return "0" if source_folder == "real" else "1"


def _fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.6f}"


def _load_audio_head(ckpt: Path, device: str):
    state = torch.load(ckpt, map_location=device)
    hp = state["model_hparams"]
    modality = hp.get("modality", "audio")
    backend = state["backend"]
    audio_dir = Path(state["audio_dir"])
    model = LateFusionClassifier(
        modality=modality, emb=hp.get("emb", 128), p=hp.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    ns = state.get("norm_stats", {}) or {}
    return {
        "model": model,
        "backend": backend,
        "audio_dir": audio_dir,
        "norm_stats": {
            "audio_mean": (np.asarray(ns["audio_mean"], dtype=np.float32)
                           if "audio_mean" in ns else None),
            "audio_std": (np.asarray(ns["audio_std"], dtype=np.float32)
                          if "audio_std" in ns else None),
            "eps": float(ns.get("eps", 1e-6)),
        },
    }


def _load_sync_head(ckpt: Path, device: str):
    state = torch.load(ckpt, map_location=device)
    cfg = state.get("config", {})
    backend = cfg.get("backend", "syncnet")
    visual_dir, audio_dir, failures_csv, embed_dim = resolve_sync_backend(backend)
    model = LipSyncPretrainedHead(
        sync_feature_dim=SYNC_FEATURE_DIM,
        embed_dim=embed_dim,
        hidden=cfg.get("hidden", 128),
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return {"model": model, "backend": backend, "visual_dir": visual_dir,
            "audio_dir": audio_dir, "embed_dim": embed_dim}


def _load_video_av_head(ckpt: Path, device: str):
    state = torch.load(ckpt, map_location=device)
    cfg = state.get("config", {})
    backend = cfg.get("backend", "avhubert")
    visual_dir, audio_dir, failures_csv, embed_dim = resolve_sync_backend(backend)
    model = LipSyncPretrainedHead(
        sync_feature_dim=SYNC_FEATURE_DIM,
        embed_dim=embed_dim,
        hidden=cfg.get("hidden", 128),
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return {"model": model, "backend": backend,
            "visual_dir": visual_dir, "audio_dir": audio_dir,
            "window_count": cfg.get("window_count", 25),
            "window_policy": cfg.get("window_policy", "center"),
            "embed_dim": embed_dim}


def load_heads(*, audio_ckpt: Path, sync_ckpt: Path, video_av_ckpt: Path, device: str):
    return (
        _load_audio_head(audio_ckpt, device),
        _load_sync_head(sync_ckpt, device),
        _load_video_av_head(video_av_ckpt, device),
    )


def backend_names(*, audio, sync, video_av) -> dict:
    return {
        "audio": _audio_backend_label(audio),
        "sync": sync["backend"],
        "video_av": _video_av_backend_label(video_av),
    }


def _audio_backend_label(audio) -> str:
    root_name = audio["audio_dir"].name       # e.g. audio_wavlm_codec
    return root_name.replace("audio_", "")


def _video_av_backend_label(video_av) -> str:
    if video_av["window_count"]:
        return f"{video_av['backend']}_fixed{video_av['window_count']}"
    return video_av["backend"]


def _score_audio(row: dict, audio) -> float | None:
    npy = audio["audio_dir"] / f"{row['audio_sample_id']}.npy"
    if not npy.exists():
        return None
    feats = np.load(npy).astype(np.float32)
    ns = audio.get("norm_stats") or {}
    audio_mean = ns.get("audio_mean") if isinstance(ns, dict) else None
    audio_std = ns.get("audio_std") if isinstance(ns, dict) else None
    eps = float(ns.get("eps", 1e-6)) if isinstance(ns, dict) else 1e-6
    if audio_mean is not None and audio_std is not None:
        feats = (feats - audio_mean) / np.maximum(audio_std, eps)
    if feats.ndim == 2:
        feats = feats[None]
    x = torch.from_numpy(feats)
    with torch.no_grad():
        logit = audio["model"](audio=x)
    return float(torch.sigmoid(logit).cpu().item())


def _score_pretrained_head(row: dict, head, *, window_count=None, window_policy="center") -> float | None:
    vid = row["source_video_id"]
    aid = row["audio_sample_id"]
    v_path = head["visual_dir"] / f"{vid}.npy"
    a_path = head["audio_dir"] / f"{aid}.npy"
    if not v_path.exists() or not a_path.exists():
        return None
    v = np.load(v_path).astype(np.float32)
    a = np.load(a_path).astype(np.float32)
    if v.ndim == 1:
        v = v[None]
    if a.ndim == 1:
        a = a[None]
    v = fixed_window(v, window_count=window_count, policy=window_policy)
    a = fixed_window(a, window_count=window_count, policy=window_policy)
    sync_features = compute_sync_features(v, a)
    pooled_v = v.mean(axis=0).astype(np.float32)
    pooled_a = a.mean(axis=0).astype(np.float32)
    params = getattr(head["model"], "parameters", lambda: iter(()))()
    device = next(params, torch.empty(0)).device
    with torch.no_grad():
        logit = head["model"](
            torch.from_numpy(sync_features).unsqueeze(0).to(device),
            torch.from_numpy(pooled_v).unsqueeze(0).to(device),
            torch.from_numpy(pooled_a).unsqueeze(0).to(device),
        )
    return float(torch.sigmoid(logit).cpu().item())


def score_row(row: dict, *, audio, sync, video_av) -> dict:
    a_score = _score_audio(row, audio)
    s_score = _score_pretrained_head(row, sync)
    v_score = _score_pretrained_head(row, video_av,
                                     window_count=video_av["window_count"],
                                     window_policy=video_av["window_policy"])
    missing = [name for name, val in (
        ("audio_fake_score", a_score),
        ("sync_inconsistent_score", s_score),
        ("video_av_fake_score", v_score),
    ) if val is None]
    return {
        "audio_fake_score": a_score,
        "sync_inconsistent_score": s_score,
        "video_av_fake_score": v_score,
        "missing_features": "|".join(missing),
    }


def build_score_table(
    *,
    manifest: Path,
    split: str,
    audio_ckpt: Path,
    sync_ckpt: Path,
    video_av_ckpt: Path,
    out: Path,
    device: str,
) -> dict:
    if split == "test":
        raise ValueError("test split is locked; refuse to build final-fusion score rows")

    audio, sync, video_av = load_heads(
        audio_ckpt=audio_ckpt, sync_ckpt=sync_ckpt,
        video_av_ckpt=video_av_ckpt, device=device,
    )
    backends = backend_names(audio=audio, sync=sync, video_av=video_av)

    with manifest.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]

    out_rows: list[dict] = []
    for row in tqdm(rows, desc=f"score {split}", unit="row"):
        scored = score_row(row, audio=audio, sync=sync, video_av=video_av)
        out_rows.append({
            "sample_id": row["sample_id"],
            "source_video_id": row["source_video_id"],
            "split": row["split"],
            "source_folder": row["source_folder"],
            "final_label_binary": _final_label_binary(row["source_folder"]),
            "audio_fake_score": _fmt(scored["audio_fake_score"]),
            "audio_backend": backends["audio"],
            "video_av_fake_score": _fmt(scored["video_av_fake_score"]),
            "video_av_backend": backends["video_av"],
            "sync_inconsistent_score": _fmt(scored["sync_inconsistent_score"]),
            "sync_backend": backends["sync"],
            "visual_fake_score": "",
            "missing_features": scored["missing_features"],
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    return {
        "manifest": str(manifest),
        "split": split,
        "n_rows": len(out_rows),
        "audio_ckpt": str(audio_ckpt),
        "audio_backend": backends["audio"],
        "sync_ckpt": str(sync_ckpt),
        "sync_backend": backends["sync"],
        "video_av_ckpt": str(video_av_ckpt),
        "video_av_backend": backends["video_av"],
        "visual_fake_score_available": False,
    }


def _resolve_audio_ckpt(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"audio checkpoint not found: {explicit}")
        return explicit
    for candidate in AUDIO_CKPT_FALLBACKS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "no audio anti-spoof checkpoint found under models/checkpoints/"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=common.VIDEO_AV_MANIFEST)
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=["train", "val"])
    parser.add_argument("--audio-ckpt", type=Path, default=None)
    parser.add_argument("--sync-ckpt", type=Path, default=DEFAULT_SYNC_CKPT)
    parser.add_argument("--video-av-ckpt", type=Path, default=DEFAULT_VIDEO_AV_CKPT)
    parser.add_argument("--out-train", type=Path, default=common.FINAL_FUSION_SCORES_TRAIN)
    parser.add_argument("--out-val", type=Path, default=common.FINAL_FUSION_SCORES_VAL)
    parser.add_argument("--provenance", type=Path, default=common.FINAL_FUSION_SCORE_PROVENANCE)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if "test" in args.splits:
        parser.error("test split is locked in this branch")

    audio_ckpt = _resolve_audio_ckpt(args.audio_ckpt)
    provenance: dict = {"splits": {}}
    for split in args.splits:
        out = args.out_train if split == "train" else args.out_val
        provenance["splits"][split] = build_score_table(
            manifest=args.manifest, split=split,
            audio_ckpt=audio_ckpt, sync_ckpt=args.sync_ckpt,
            video_av_ckpt=args.video_av_ckpt, out=out, device=args.device,
        )
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2))
    print(f"wrote provenance to {args.provenance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
