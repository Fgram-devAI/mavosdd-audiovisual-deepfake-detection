"""Unit tests for src/analysis/acoustic_probe_models.py."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _toy_separable(n: int = 60, seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    pos = rng.normal(loc=+1.0, size=(n // 2, 3))
    neg = rng.normal(loc=-1.0, size=(n // 2, 3))
    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(n // 2), np.zeros(n // 2)]).astype(int)
    return X, y, ["f0", "f1", "f2"]


def test_fit_logistic_is_deterministic():
    from src.analysis.acoustic_probe_models import fit_logistic

    X, y, _ = _toy_separable()
    m1 = fit_logistic(X, y, seed=42)
    m2 = fit_logistic(X, y, seed=42)
    p1 = m1.predict_proba(X)[:, 1]
    p2 = m2.predict_proba(X)[:, 1]
    assert np.allclose(p1, p2)


def test_fit_random_forest_is_deterministic():
    from src.analysis.acoustic_probe_models import fit_random_forest

    X, y, _ = _toy_separable()
    m1 = fit_random_forest(X, y, seed=42)
    m2 = fit_random_forest(X, y, seed=42)
    p1 = m1.predict_proba(X)[:, 1]
    p2 = m2.predict_proba(X)[:, 1]
    assert np.allclose(p1, p2)


def test_evaluate_separable_returns_high_auc():
    from src.analysis.acoustic_probe_models import evaluate, fit_logistic

    X, y, _ = _toy_separable()
    model = fit_logistic(X, y, seed=42)
    res = evaluate(model, X, y, providers_val=None)
    assert res["roc_auc"] > 0.95
    assert 0.0 <= res["eer"] <= 0.5
    assert {"tn", "fp", "fn", "tp"} <= set(res["confusion"].keys())
    assert "scores" in res and len(res["scores"]) == len(y)


def test_evaluate_with_providers_breaks_down_recall():
    from src.analysis.acoustic_probe_models import evaluate, fit_logistic

    X, y, _ = _toy_separable()
    model = fit_logistic(X, y, seed=42)
    providers = np.array(["A"] * (len(y) // 2) + ["B"] * (len(y) - len(y) // 2))
    res = evaluate(model, X, y, providers_val=providers)
    assert set(res["per_provider_recall"].keys()) == {"A", "B"}


def test_per_feature_lr_sweep_returns_sorted_list():
    from src.analysis.acoustic_probe_models import per_feature_lr_sweep

    X, y, names = _toy_separable()
    out = per_feature_lr_sweep(X, y, X, y, names, seed=42)
    assert len(out) == 3
    assert all("feature" in r and "val_roc_auc" in r for r in out)
    aucs = [r["val_roc_auc"] for r in out]
    assert aucs == sorted(aucs, reverse=True)


def test_loeo_matrix_discovers_engines_and_skips_degenerate():
    from src.analysis.acoustic_probe_models import loeo_matrix

    # Two engines E1 and E2, plus bonafide rows.
    rng = np.random.default_rng(42)
    rows = []
    for split, n in [("train", 40), ("val", 20)]:
        for engine, label in [("bonafide", 0), ("E1", 1), ("E2", 1)]:
            for _ in range(n):
                rows.append({
                    "split": split,
                    "provider": "original" if label == 0 else engine,
                    "audio_label_binary": label,
                    "f0": float(rng.normal(+1.0 if label == 1 else -1.0)),
                    "f1": float(rng.normal(+1.0 if label == 1 else -1.0)),
                })
    df = pd.DataFrame(rows)
    out = loeo_matrix(df, feature_cols=["f0", "f1"], seed=42)
    engines = {row["engine"] for row in out}
    assert engines == {"E1", "E2"}
    for row in out:
        assert row.get("skipped") in (None, "empty_val", "degenerate_one_engine")


def test_loeo_matrix_one_engine_emits_degenerate_row():
    from src.analysis.acoustic_probe_models import loeo_matrix

    rng = np.random.default_rng(42)
    rows = []
    for split, n in [("train", 20), ("val", 10)]:
        for engine, label in [("bonafide", 0), ("E1", 1)]:
            for _ in range(n):
                rows.append({
                    "split": split,
                    "provider": "original" if label == 0 else engine,
                    "audio_label_binary": label,
                    "f0": float(rng.normal(+1.0 if label == 1 else -1.0)),
                })
    df = pd.DataFrame(rows)
    out = loeo_matrix(df, feature_cols=["f0"], seed=42)
    assert len(out) == 1
    assert out[0]["engine"] == "E1"
    assert out[0].get("skipped") == "degenerate_one_engine"


def test_loeo_matrix_empty_train_emits_skip_row():
    """Engine whose held-out train spoof set is empty must emit {skipped: empty_train}.

    We build a 2-engine fixture where E1 rows only appear in val and E2 rows
    only appear in train.  When the loop holds E1 out:
      train_spoof = spoof[provider != E1 AND split == train] = E2-train  → non-empty
      val_spoof   = spoof[provider == E1 AND split == val]   = E1-val    → non-empty
    so E1 does NOT hit empty_train here.

    The correct trigger is: when E2 is held out, the remaining training spoof is
      train_spoof = spoof[provider != E2 AND split == train] = empty  (E1 has no train rows)
    So we assert E2's result has skipped == "empty_train".
    """
    from src.analysis.acoustic_probe_models import loeo_matrix

    rng = np.random.default_rng(42)
    rows = []
    # E1 spoof: val split only (no train rows for E1).
    # E2 spoof: train split only (no val rows for E2).
    # When E2 is held out: train_spoof = E1-train = empty → skipped: empty_train.
    for split, engine, label, n in [
        ("val",   "E1", 1, 10),   # E1 exists only in val
        ("train", "E2", 1, 20),   # E2 exists only in train
        ("train", "bonafide", 0, 20),
        ("val",   "bonafide", 0, 10),
    ]:
        for _ in range(n):
            rows.append({
                "split": split,
                "provider": "original" if label == 0 else engine,
                "audio_label_binary": label,
                "f0": float(rng.normal(+1.0 if label == 1 else -1.0)),
            })
    df = pd.DataFrame(rows)
    out = loeo_matrix(df, feature_cols=["f0"], seed=42)
    by_engine = {r["engine"]: r for r in out}
    # E2 held out → train_spoof is empty (only E1 is left, and E1 has no train rows)
    assert "E2" in by_engine
    assert by_engine["E2"].get("skipped") == "empty_train"
