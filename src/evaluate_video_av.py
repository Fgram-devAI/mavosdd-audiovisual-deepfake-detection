"""Evaluate video-level real-vs-fake classifier over pretrained AV embeddings."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src import common
from src.data.video_av_dataset import VideoAVDataset, make_dataloader
from src.evaluate_lipsync_pretrained import EvaluationRefusedError, _eer
from src.models.lipsync_pretrained_head import LipSyncPretrainedHead
from src.train_lipsync_pretrained import resolve_backend
from src.data.lipsync_pretrained_dataset import SYNC_FEATURE_DIM


def compute_metrics(scores: np.ndarray, labels: np.ndarray, source_folders: list[str]) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

    roc = float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) > 1 else float("nan")
    eer, threshold = _eer(scores, labels)
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    per_source: dict[str, float] = {}
    for source in sorted(set(source_folders)):
        mask = np.array([s == source for s in source_folders])
        if not mask.any():
            continue
        target = 0 if source == "real" else 1
        per_source[source] = float((preds[mask] == target).mean())

    return {
        "roc_auc": roc,
        "eer": eer,
        "threshold_used": threshold,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_source_recall_or_specificity": per_source,
    }


def format_report(
    metrics: dict,
    *,
    split: str,
    n_manifest: int,
    n_evaluated: int,
    n_excluded: int,
    excluded_by_reason: dict,
    partial: bool,
) -> str:
    return (
        f"split={split} n_manifest={n_manifest} n_evaluated={n_evaluated} "
        f"n_excluded={n_excluded} excluded_by_reason={excluded_by_reason} "
        f"partial_evaluation={'true' if partial else 'false'} "
        f"positive_class=fake_video roc_auc={metrics['roc_auc']:.4f} "
        f"eer={metrics['eer']:.4f} threshold_used={metrics['threshold_used']:.4f} "
        f"f1={metrics['f1']:.4f} precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f}\n"
        f"confusion={metrics['confusion']}\n"
        f"per_source_recall_or_specificity={metrics['per_source_recall_or_specificity']}\n"
    )


def evaluate(
    *,
    checkpoint: Path,
    backend: str,
    manifest: Path,
    split: str,
    allow_partial: bool,
    out: Path,
    device: str,
    window_count: int | None,
    window_policy: str,
) -> int:
    if split == "test":
        raise ValueError("test split is locked; refuse to evaluate on test")

    visual_dir, audio_dir, failures_csv, embed_dim = resolve_backend(backend)
    fcsv = failures_csv if failures_csv.exists() else None
    ds = VideoAVDataset(
        manifest=manifest,
        split=split,
        visual_dir=visual_dir,
        audio_dir=audio_dir,
        failures_csv=fcsv,
        window_count=window_count,
        window_policy=window_policy,
    )

    with manifest.open(newline="") as f:
        all_split_rows = [r for r in csv.DictReader(f) if r["split"] == split]
    n_manifest = len(all_split_rows)
    n_evaluated = len(ds)
    n_excluded = n_manifest - n_evaluated
    reasons: Counter = Counter()
    if n_excluded:
        for row in all_split_rows:
            if row["sample_id"] not in ds.excluded_sample_ids:
                continue
            vid = row["source_video_id"]
            aid = row["audio_sample_id"]
            if not (visual_dir / f"{vid}.npy").exists():
                reasons["missing_visual"] += 1
            elif not (audio_dir / f"{aid}.npy").exists():
                reasons["missing_audio"] += 1
            else:
                reasons["extraction_failure"] += 1
        if not allow_partial:
            raise EvaluationRefusedError(
                f"partial evaluation refused: n_manifest={n_manifest} "
                f"n_evaluated={n_evaluated} excluded_by_reason={dict(reasons)}"
            )

    if n_evaluated == 0:
        raise EvaluationRefusedError("n_evaluated=0; refuse to write metrics")

    state = torch.load(checkpoint, map_location=device)
    cfg = state.get("config", {})
    head = LipSyncPretrainedHead(
        sync_feature_dim=SYNC_FEATURE_DIM,
        embed_dim=embed_dim,
        hidden=cfg.get("hidden", 128),
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    head.load_state_dict(state["state_dict"])
    head.eval()

    dl = make_dataloader(ds, batch_size=64, shuffle=False, num_workers=0, seed=42)
    scores: list[float] = []
    labels: list[float] = []
    sources: list[str] = []
    with torch.no_grad():
        for batch in dl:
            logits = head(
                batch["sync_features"].to(device),
                batch["pooled_visual"].to(device),
                batch["pooled_audio"].to(device),
            )
            scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(batch["label"].cpu().numpy().tolist())
            sources.extend(batch["source_folder"])

    unique_labels = set(int(x) for x in labels)
    if unique_labels != {0, 1}:
        raise EvaluationRefusedError(f"missing label class in evaluated rows: {sorted(unique_labels)}")

    metrics = compute_metrics(np.asarray(scores), np.asarray(labels), sources)
    text = format_report(
        metrics,
        split=split,
        n_manifest=n_manifest,
        n_evaluated=n_evaluated,
        n_excluded=n_excluded,
        excluded_by_reason=dict(reasons),
        partial=n_excluded > 0,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    prefix = "# WARNING: partial_evaluation=true -- see excluded_by_reason\n" if n_excluded else ""
    out.write_text(prefix + text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("syncnet", "avhubert"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=common.VIDEO_AV_MANIFEST)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window-count", type=int, default=None,
                        help="crop/pad each visual/audio embedding sequence to this many windows")
    parser.add_argument("--window-policy", choices=("center", "first"), default="center")
    args = parser.parse_args(argv)
    return evaluate(
        checkpoint=args.checkpoint,
        backend=args.backend,
        manifest=args.manifest,
        split=args.split,
        allow_partial=args.allow_partial,
        out=args.out,
        device=args.device,
        window_count=args.window_count,
        window_policy=args.window_policy,
    )


if __name__ == "__main__":
    raise SystemExit(main())
