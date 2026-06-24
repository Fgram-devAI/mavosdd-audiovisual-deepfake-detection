"""Evaluation: metric functions, checkpoint evaluator, and CLI."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


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


def main() -> None:
    raise NotImplementedError("CLI lands in Task 4")


if __name__ == "__main__":
    main()
