"""Evaluate final-fusion baselines on val and emit the comparison table."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

from src import common
from src.data.final_fusion_dataset import DEFAULT_FEATURE_COLUMNS, FinalFusionDataset
from src.models.final_fusion import (
    FinalFusionLogReg, FinalFusionMLP,
    rule_score_audio_only, rule_score_max_audio_video_av,
    rule_score_max_available, rule_score_sync_only, rule_score_video_av_only,
)


PER_SOURCE_CANDIDATES = ("real", "echomimic", "memo", "liveportrait", "sonic")

BASELINES = (
    ("audio_only", rule_score_audio_only, ("audio_fake_score",)),
    ("video_av_only", rule_score_video_av_only, ("video_av_fake_score",)),
    ("sync_only", rule_score_sync_only, ("sync_inconsistent_score",)),
    ("max_audio_video_av", rule_score_max_audio_video_av,
     ("audio_fake_score", "video_av_fake_score")),
    ("max_available", rule_score_max_available,
     ("audio_fake_score", "video_av_fake_score", "sync_inconsistent_score")),
)


def _eer_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thr[idx])


def compute_metrics(
    scores: np.ndarray, labels: np.ndarray,
    source_folders: list[str], threshold: float | None = None,
) -> dict:
    if len(set(labels.tolist())) < 2:
        roc = float("nan")
        eer = float("nan")
        used = threshold if threshold is not None else 0.5
    else:
        roc = float(roc_auc_score(labels, scores))
        eer, thr = _eer_threshold(scores, labels)
        used = threshold if threshold is not None else thr
    preds = (scores >= used).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    per_source: dict[str, float] = {}
    for src in PER_SOURCE_CANDIDATES:
        mask = np.array([s == src for s in source_folders])
        if not mask.any():
            continue
        target = 0 if src == "real" else 1
        per_source[src] = float((preds[mask] == target).mean())
    return {
        "roc_auc": roc,
        "eer": eer,
        "threshold": float(used),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_source_recall_or_specificity": per_source,
    }


def _rows_from_csv(path: Path, split: str) -> list[dict]:
    with path.open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def _rule_scores(rows: list[dict], rule_fn, required: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    scores: list[float] = []
    labels: list[int] = []
    sources: list[str] = []
    excluded = 0
    for r in rows:
        if any(r.get(c, "") == "" for c in required):
            excluded += 1
            continue
        scores.append(rule_fn(r))
        labels.append(int(r["final_label_binary"]))
        sources.append(r["source_folder"])
    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int64), sources, excluded


def _load_mlp(path: Path) -> tuple[FinalFusionMLP, dict]:
    state = torch.load(path, map_location="cpu")
    mlp = FinalFusionMLP(
        input_dim=state["input_dim"], hidden=state["hidden"], dropout=state["dropout"],
    )
    mlp.load_state_dict(state["state_dict"])
    mlp.eval()
    return mlp, state


def format_comparison_md(rows: list[dict]) -> str:
    header = (
        "| Row | roc_auc | EER | F1 | Real specificity | Fake recall | LivePortrait recall | Notes |\n"
        "|---|---:|---:|---:|---:|---:|---:|---|\n"
    )
    body = []
    for r in rows:
        ps = r.get("per_source_recall_or_specificity", {})
        real_spec = ps.get("real", float("nan"))
        lp_recall = ps.get("liveportrait", float("nan"))
        body.append(
            f"| {r['name']} | {r['roc_auc']:.4f} | {r['eer']:.4f} | {r['f1']:.4f} | "
            f"{real_spec:.4f} | {r['recall']:.4f} | {lp_recall:.4f} | {r.get('notes', '')} |"
        )
    footnote = (
        "\n\n> `visual_frame_baseline_notebook_only`: the EfficientNet-B0 sampled-frame baseline "
        "lives in `notebooks/03_visual_frame_baseline_extended_data.ipynb` and has no stable CLI. "
        "See `report/val_eval/final_fusion_visual_frame_unavailable.md`.\n"
    )
    return header + "\n".join(body) + footnote


def _format_txt_line(name: str, metrics: dict, notes: str) -> str:
    ps = metrics["per_source_recall_or_specificity"]
    return (
        f"row={name} roc_auc={metrics['roc_auc']:.4f} eer={metrics['eer']:.4f} "
        f"threshold={metrics['threshold']:.4f} f1={metrics['f1']:.4f} "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"confusion={metrics['confusion']} per_source={ps} notes={notes!r}\n"
    )


def evaluate(
    *,
    val_scores: Path,
    logreg_ckpt: Path | None,
    mlp_ckpt: Path | None,
    split: str,
    out: Path,
    comparison: Path,
) -> None:
    if split == "test":
        raise ValueError("test split is locked; refuse to evaluate final fusion on test")

    rows = _rows_from_csv(val_scores, split)
    if not rows:
        raise RuntimeError(f"no rows for split={split!r} in {val_scores}")

    lines: list[str] = []
    md_rows: list[dict] = []

    for name, fn, required in BASELINES:
        s, y, src, excluded = _rule_scores(rows, fn, required)
        if s.size == 0:
            note = "no rows had required features"
            md_rows.append({"name": name, "roc_auc": float("nan"), "eer": float("nan"),
                            "f1": float("nan"), "recall": float("nan"),
                            "per_source_recall_or_specificity": {}, "notes": note})
            lines.append(_format_txt_line(name, {
                "roc_auc": float("nan"), "eer": float("nan"), "threshold": 0.5,
                "f1": float("nan"), "precision": float("nan"), "recall": float("nan"),
                "confusion": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
                "per_source_recall_or_specificity": {},
            }, note))
            continue
        m = compute_metrics(s, y, src)
        notes = f"n={len(s)} excluded_missing={excluded}"
        md_rows.append({"name": name, "notes": notes, **m})
        lines.append(_format_txt_line(name, m, notes))

    trainable_baselines: list[tuple[str, Path | None, str]] = [
        ("logistic_fusion", logreg_ckpt, "preferred if stable"),
        ("mlp_fusion", mlp_ckpt, "only if it genuinely improves"),
    ]
    for name, ckpt, note in trainable_baselines:
        if ckpt is None or not ckpt.exists():
            md_rows.append({"name": name, "roc_auc": float("nan"), "eer": float("nan"),
                            "f1": float("nan"), "recall": float("nan"),
                            "per_source_recall_or_specificity": {},
                            "notes": f"checkpoint missing at {ckpt}"})
            continue
        if name == "logistic_fusion":
            model = FinalFusionLogReg.load(ckpt)
            feat = model.feature_columns
        else:
            mlp, state = _load_mlp(ckpt)
            feat = tuple(state["feature_columns"])
        ds = FinalFusionDataset(score_csv=val_scores, split=split, feature_columns=feat)
        if len(ds) == 0:
            md_rows.append({"name": name, "roc_auc": float("nan"), "eer": float("nan"),
                            "f1": float("nan"), "recall": float("nan"),
                            "per_source_recall_or_specificity": {},
                            "notes": "no rows had all required features"})
            continue
        if name == "logistic_fusion":
            scores = model.predict_proba(ds.X).astype(np.float32)
            threshold = float(model.threshold)
        else:
            mean = np.asarray(state["scaler_mean"], dtype=np.float64)
            scale = np.asarray(state["scaler_scale"], dtype=np.float64)
            X = ((ds.X.astype(np.float64) - mean) / np.where(scale == 0, 1.0, scale)).astype(np.float32)
            with torch.no_grad():
                logits = mlp(torch.from_numpy(X))
                scores = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            threshold = float(state["threshold"])
        m = compute_metrics(scores, ds.y, ds.source_folders, threshold=threshold)
        notes = f"n={len(ds)} excluded_missing={ds.excluded_missing}; {note}"
        md_rows.append({"name": name, "notes": notes, **m})
        lines.append(_format_txt_line(name, m, notes))

    md_rows.append({"name": "visual_frame_baseline_notebook_only",
                    "roc_auc": float("nan"), "eer": float("nan"),
                    "f1": float("nan"), "recall": float("nan"),
                    "per_source_recall_or_specificity": {},
                    "notes": "notebook-only, not part of trainable fusion"})

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines))
    comparison.parent.mkdir(parents=True, exist_ok=True)
    comparison.write_text(format_comparison_md(md_rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-scores", type=Path, default=common.FINAL_FUSION_SCORES_VAL)
    parser.add_argument("--logreg-ckpt", type=Path, default=common.CKPT_FINAL_FUSION_LOGREG)
    parser.add_argument("--mlp-ckpt", type=Path, default=common.CKPT_FINAL_FUSION_MLP)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--out", type=Path, default=common.FINAL_FUSION_VAL_REPORT)
    parser.add_argument("--comparison", type=Path, default=common.FINAL_FUSION_COMPARISON_REPORT)
    args = parser.parse_args(argv)

    evaluate(
        val_scores=args.val_scores,
        logreg_ckpt=args.logreg_ckpt if args.logreg_ckpt.exists() else None,
        mlp_ckpt=args.mlp_ckpt if args.mlp_ckpt.exists() else None,
        split=args.split, out=args.out, comparison=args.comparison,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
