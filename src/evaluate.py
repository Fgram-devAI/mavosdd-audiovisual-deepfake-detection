"""Evaluation: metric functions, checkpoint evaluator, and CLI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.data.feature_store import (
    AudioFeatureDataset,
    FusionFeatureDataset,
    NormalizationStats,
    VisualFeatureDataset,
    make_dataloader,
)
from src.models.late_fusion import LateFusionClassifier


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, score))


def equal_error_rate(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """EER from the ROC curve: argmin |fnr - fpr|."""
    fpr, tpr, thresholds = roc_curve(y_true, score)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def f1_at_threshold(
    y_true: np.ndarray, score: np.ndarray, threshold: float = 0.5
) -> tuple[float, float, float]:
    pred = (score >= threshold).astype(int)
    f1 = float(f1_score(y_true, pred, zero_division=0))
    prec = float(precision_score(y_true, pred, zero_division=0))
    rec = float(recall_score(y_true, pred, zero_division=0))
    return f1, prec, rec


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def per_provider_recall(
    y_true: np.ndarray, y_pred: np.ndarray, providers: np.ndarray
) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in sorted(set(providers.tolist())):
        m = providers == p
        y_p = y_true[m]
        if y_p.sum() == 0:
            continue
        out[p] = float(recall_score(y_p, y_pred[m], zero_division=0))
    return out


def metric_battery(
    y_true: np.ndarray,
    score: np.ndarray,
    providers: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    eer, eer_thr = equal_error_rate(y_true, score)
    f1, prec, rec = f1_at_threshold(y_true, score, threshold=threshold)
    pred = (score >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "roc_auc": roc_auc(y_true, score),
        "eer": eer,
        "eer_threshold": eer_thr,
        "threshold": float(threshold),
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "confusion": confusion(y_true, pred),
        "per_provider_recall": per_provider_recall(y_true, pred, providers),
    }


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def evaluate_checkpoint(
    ckpt_path: str | Path,
    *,
    split: str,
    allow_test: bool = False,
    device: str = "auto",
    manifest: str | Path | None = None,
    lips_dir: Path | None = None,
    batch_size: int = 32,
) -> dict:
    if split == "test" and not allow_test:
        raise SystemExit(
            "Refusing to evaluate on test split without --allow-test. "
            "Use --split val for model selection."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dev = _resolve_device(device)

    hp = ckpt["model_hparams"]
    modality = hp["modality"]
    model = LateFusionClassifier(
        modality=modality, emb=hp.get("emb", 128), p=hp.get("dropout", 0.3)
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()

    ns = ckpt["norm_stats"]
    stats = NormalizationStats(
        audio_mean=np.asarray(ns["audio_mean"], dtype=np.float32) if "audio_mean" in ns else None,
        audio_std=np.asarray(ns["audio_std"], dtype=np.float32) if "audio_std" in ns else None,
        lips_mean=np.asarray(ns["lips_mean"], dtype=np.float32) if "lips_mean" in ns else None,
        lips_std=np.asarray(ns["lips_std"], dtype=np.float32) if "lips_std" in ns else None,
        eps=ns.get("eps", 1e-6),
    )

    manifest_path = str(manifest) if manifest is not None else ckpt["manifest"]
    if modality == "audio":
        ds = AudioFeatureDataset(
            manifest_path=manifest_path, split=split,
            backend=ckpt["backend"], audio_dir=Path(ckpt["audio_dir"]),
            normalization=stats,
        )
    elif modality == "visual":
        ds = VisualFeatureDataset(
            manifest_path=manifest_path, split=split,
            lips_dir=lips_dir, normalization=stats,
        )
    elif modality == "fusion":
        ds = FusionFeatureDataset(
            manifest_path=manifest_path, split=split,
            backend=ckpt["backend"], audio_dir=Path(ckpt["audio_dir"]),
            lips_dir=lips_dir, normalization=stats,
        )
    else:
        raise ValueError(f"unknown checkpoint modality: {modality!r}")

    loader = make_dataloader(ds, batch_size=batch_size, shuffle=False)

    ys: list[int] = []
    scores: list[float] = []
    providers: list[str] = []
    with torch.no_grad():
        for batch in loader:
            if modality == "audio":
                logits = model(batch["audio"].to(dev))
            elif modality == "visual":
                logits = model(None, batch["lips"].to(dev), batch["lips_mask"].to(dev))
            else:
                logits = model(
                    batch["audio"].to(dev),
                    batch["lips"].to(dev),
                    batch["lips_mask"].to(dev),
                )
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            scores.extend(probs)
            ys.extend(batch["label"].cpu().numpy().astype(int).tolist())
            providers.extend(m.get("provider", "") for m in batch["metadata"])

    return metric_battery(
        np.asarray(ys, dtype=int),
        np.asarray(scores, dtype=float),
        np.asarray(providers, dtype=object),
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate an audio anti-spoof checkpoint.")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--split", required=True, choices=("train", "val", "test"))
    p.add_argument("--allow-test", action="store_true",
                   help="Required to evaluate on the test split.")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    p.add_argument("--manifest", default=None, type=Path,
                   help="Override manifest path (default: ckpt['manifest']).")
    p.add_argument("--lips-dir", default=None, type=Path,
                   help="Override lip feature dir for visual/fusion checkpoints.")
    p.add_argument("--batch-size", type=int, default=32)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = evaluate_checkpoint(
            args.checkpoint,
            split=args.split,
            allow_test=args.allow_test,
            device=args.device,
            manifest=args.manifest,
            lips_dir=args.lips_dir,
            batch_size=args.batch_size,
        )
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        f"split={args.split} n={result['n']} "
        f"roc_auc={result['roc_auc']:.4f} eer={result['eer']:.4f} "
        f"f1={result['f1']:.4f} prec={result['precision']:.4f} rec={result['recall']:.4f}"
    )
    print(f"confusion={result['confusion']}")
    if result["per_provider_recall"]:
        print(f"per_provider_recall={result['per_provider_recall']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
