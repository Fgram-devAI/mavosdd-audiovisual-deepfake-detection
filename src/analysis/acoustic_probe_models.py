"""Cheap classical probes and metric helpers for the acoustic confound probe.

All probes are deterministic given a single integer ``seed``: sklearn's
``random_state`` is set everywhere, and the toy data fed to them comes from a
seeded ``numpy.random.default_rng``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_logistic(X_train: np.ndarray, y_train: np.ndarray, *, seed: int) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def fit_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, *, seed: int
) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        n_jobs=1,
    )
    rf.fit(X_train, y_train)
    return rf


def _eer_from_scores(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def evaluate(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    providers_val: np.ndarray | None = None,
) -> dict:
    scores = model.predict_proba(X_val)[:, 1]
    preds = (scores >= 0.5).astype(int)
    roc_auc = float(roc_auc_score(y_val, scores))
    eer, eer_thresh = _eer_from_scores(y_val, scores)
    f1 = float(f1_score(y_val, preds, zero_division=0))
    precision = float(precision_score(y_val, preds, zero_division=0))
    recall = float(recall_score(y_val, preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()

    per_provider: dict[str, float] = {}
    if providers_val is not None:
        providers_val = np.asarray(providers_val)
        for p in np.unique(providers_val):
            mask = providers_val == p
            if mask.sum() == 0:
                continue
            # Recall for the binary positive class within this provider slice.
            y_slice = y_val[mask]
            pred_slice = preds[mask]
            if y_slice.sum() == 0:
                # No positives in this slice — recall undefined; record None.
                per_provider[str(p)] = float("nan")
                continue
            per_provider[str(p)] = float(
                recall_score(y_slice, pred_slice, zero_division=0)
            )

    return {
        "roc_auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_thresh,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_provider_recall": per_provider,
        "scores": scores.tolist(),
    }


def per_feature_lr_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    *,
    seed: int,
) -> list[dict]:
    results: list[dict] = []
    for i, name in enumerate(feature_names):
        Xt = X_train[:, [i]]
        Xv = X_val[:, [i]]
        model = fit_logistic(Xt, y_train, seed=seed)
        scores = model.predict_proba(Xv)[:, 1]
        try:
            auc = float(roc_auc_score(y_val, scores))
        except ValueError:
            auc = float("nan")
        results.append({"feature": name, "val_roc_auc": auc})
    results.sort(key=lambda r: (np.isnan(r["val_roc_auc"]), -r["val_roc_auc"]))
    return results


def loeo_matrix(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    seed: int,
) -> list[dict]:
    spoof = features_df[features_df["audio_label_binary"] == 1]
    bonafide = features_df[features_df["audio_label_binary"] == 0]
    engines = sorted(spoof["provider"].unique().tolist())

    results: list[dict] = []
    if len(engines) == 0:
        return results

    if len(engines) == 1:
        # Degenerate: no "leave one out" possible. Emit a single skip row so the
        # CLI can still report something useful.
        return [{"engine": engines[0], "skipped": "degenerate_one_engine"}]

    for held in engines:
        train_spoof = spoof[(spoof["provider"] != held) & (spoof["split"] == "train")]
        train_bona = bonafide[bonafide["split"] == "train"]
        val_spoof = spoof[(spoof["provider"] == held) & (spoof["split"] == "val")]
        val_bona = bonafide[bonafide["split"] == "val"]

        if len(val_spoof) == 0 or len(val_bona) == 0:
            results.append({"engine": held, "skipped": "empty_val"})
            continue

        train = pd.concat([train_spoof, train_bona], ignore_index=True)
        val = pd.concat([val_spoof, val_bona], ignore_index=True)

        X_train = train[feature_cols].to_numpy()
        y_train = train["audio_label_binary"].to_numpy().astype(int)
        X_val = val[feature_cols].to_numpy()
        y_val = val["audio_label_binary"].to_numpy().astype(int)

        model = fit_logistic(X_train, y_train, seed=seed)
        scores = model.predict_proba(X_val)[:, 1]
        try:
            auc = float(roc_auc_score(y_val, scores))
        except ValueError:
            auc = float("nan")
        eer, _ = _eer_from_scores(y_val, scores)

        results.append({
            "engine": held,
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "val_roc_auc": auc,
            "val_eer": eer,
        })

    return results
