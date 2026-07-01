"""CLI: acoustic confound probe over data/derived/audio_spoof_manifest.csv.

Usage:
    python -m src.analysis.acoustic_probe \\
        --manifest data/derived/audio_spoof_manifest.csv \\
        --out-dir  report/acoustic_probe \\
        [--force] [--no-plots] [--f0] [--loeo] \\
        [--max-rows N] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.analysis.acoustic_features import (
    AcousticFeatureError,
    compute_features,
    feature_columns,
)

REQUIRED_MANIFEST_COLUMNS = [
    "sample_id",
    "split",
    "source_folder",
    "provider",
    "audio_path",
    "audio_label_binary",
]

CACHE_METADATA_COLUMNS = [
    "sample_id",
    "split",
    "source_folder",
    "provider",
    "audio_label_binary",
]


@dataclass(frozen=True)
class CLIConfig:
    manifest: str
    out_dir: str
    force: bool
    no_plots: bool
    with_f0: bool
    loeo: bool
    max_rows: int | None
    seed: int


def _parse_args(argv: list[str] | None) -> CLIConfig:
    p = argparse.ArgumentParser(prog="acoustic_probe", description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--f0", action="store_true", dest="with_f0")
    p.add_argument("--loeo", action="store_true")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    return CLIConfig(
        manifest=args.manifest,
        out_dir=args.out_dir,
        force=args.force,
        no_plots=args.no_plots,
        with_f0=args.with_f0,
        loeo=args.loeo,
        max_rows=args.max_rows,
        seed=args.seed,
    )


def _validate_manifest(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    os.close(fd)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    os.close(fd)
    Path(tmp).write_text(json.dumps(_json_safe(obj), indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def _json_safe(obj: Any) -> Any:
    """Convert numpy scalars and non-finite floats into strict JSON values."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _concat_nonempty(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    # Skip empty frames so pd.concat doesn't emit a FutureWarning about
    # all-NA / empty column dtype inference.
    parts = [df for df in frames if len(df) > 0]
    if not parts:
        return pd.DataFrame(columns=columns)
    if len(parts) == 1:
        return parts[0].reset_index(drop=True)
    return pd.concat(parts, ignore_index=True)


class CacheSchemaError(Exception):
    """Raised when the cached features.csv schema doesn't match the requested flags."""


def _load_or_init_cache(
    out_dir: Path,
    expected_columns: list[str],
    force: bool,
) -> pd.DataFrame:
    cache_path = out_dir / "acoustic_features.csv"
    if force or not cache_path.exists():
        return pd.DataFrame(columns=expected_columns)
    cached = pd.read_csv(cache_path)
    cached_cols = set(cached.columns)
    if cached_cols != set(expected_columns):
        only_cached = cached_cols - set(expected_columns)
        only_expected = set(expected_columns) - cached_cols
        raise CacheSchemaError(
            "cache schema does not match the requested feature schema "
            f"(only_in_cache={sorted(only_cached)}, "
            f"only_in_request={sorted(only_expected)}). "
            "Pass --force to recompute, or point --out-dir at a clean directory."
        )
    # Convert feature columns to float using coerce so bad data surfaces as NaN
    # rather than silently keeping untyped object columns.
    feat_cols_in_cache = [c for c in cached.columns if c not in CACHE_METADATA_COLUMNS]
    if feat_cols_in_cache:
        cached[feat_cols_in_cache] = cached[feat_cols_in_cache].apply(
            pd.to_numeric, errors="coerce"
        )
    return cached


def _extract_features_resumable(
    manifest: pd.DataFrame,
    cache: pd.DataFrame,
    out_dir: Path,
    *,
    with_f0: bool,
    force: bool = False,
    flush_every: int = 100,
) -> tuple[pd.DataFrame, list[dict]]:
    cached_ids = set(cache["sample_id"].astype(str)) if len(cache) else set()
    feat_cols = feature_columns(with_f0=with_f0)
    expected_columns = CACHE_METADATA_COLUMNS + feat_cols

    new_rows: list[dict] = []
    bad_rows: list[dict] = []
    failures_path = out_dir / "acoustic_failures.csv"
    # On --force we start the failures log fresh; otherwise we carry forward
    # any failures from previous resumable runs so the log stays cumulative.
    existing_failures = (
        pd.read_csv(failures_path).to_dict("records")
        if failures_path.exists() and not force else []
    )

    processed = 0
    for _, row in manifest.iterrows():
        sample_id = str(row["sample_id"])
        if sample_id in cached_ids:
            continue
        try:
            feats = compute_features(Path(row["audio_path"]), with_f0=with_f0)
        except AcousticFeatureError as exc:
            bad_rows.append({
                "sample_id": sample_id,
                "audio_path": str(row["audio_path"]),
                "reason": str(exc),
            })
        else:
            new_rows.append({
                **{c: row[c] for c in CACHE_METADATA_COLUMNS},
                **feats,
            })
        processed += 1
        if processed % flush_every == 0:
            _flush_cache_and_failures(
                cache, new_rows, existing_failures + bad_rows,
                expected_columns, out_dir,
            )

    if new_rows or not (out_dir / "acoustic_features.csv").exists():
        new_df = pd.DataFrame(new_rows, columns=expected_columns)
        # Convert feature columns to float to preserve dtype
        for col in feat_cols:
            if col in new_df.columns:
                new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
        cache = _concat_nonempty([cache, new_df], expected_columns)
        # Ensure cache has correct dtypes before writing
        for col in feat_cols:
            if col in cache.columns:
                cache[col] = pd.to_numeric(cache[col], errors='coerce')
        _atomic_write_csv(cache[expected_columns], out_dir / "acoustic_features.csv")
    # Always rewrite failures CSV so a clean run with zero failures produces an
    # empty-but-present CSV.
    failures_df = pd.DataFrame(
        existing_failures + bad_rows,
        columns=["sample_id", "audio_path", "reason"],
    )
    _atomic_write_csv(failures_df, failures_path)
    return cache, existing_failures + bad_rows


def _flush_cache_and_failures(
    cache: pd.DataFrame,
    new_rows: list[dict],
    all_failures: list[dict],
    expected_columns: list[str],
    out_dir: Path,
) -> None:
    new_df = pd.DataFrame(new_rows, columns=expected_columns)
    # Infer feature columns by removing metadata columns
    feat_cols = [c for c in expected_columns if c not in CACHE_METADATA_COLUMNS]
    # Convert feature columns to float to preserve dtype
    for col in feat_cols:
        if col in new_df.columns:
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
    snapshot = _concat_nonempty([cache, new_df], expected_columns)
    # Ensure snapshot has correct dtypes before writing
    for col in feat_cols:
        if col in snapshot.columns:
            snapshot[col] = pd.to_numeric(snapshot[col], errors='coerce')
    _atomic_write_csv(snapshot[expected_columns], out_dir / "acoustic_features.csv")
    _atomic_write_csv(
        pd.DataFrame(all_failures, columns=["sample_id", "audio_path", "reason"]),
        out_dir / "acoustic_failures.csv",
    )


def _run_summaries(features: pd.DataFrame, out_dir: Path) -> None:
    """Compute and write summary statistics grouped by label, provider, and source folder."""
    if len(features) == 0:
        return None
    numeric_cols = features.select_dtypes(include="number").columns.tolist()
    # We summarize the numeric feature columns only, not the label column itself.
    numeric_cols = [c for c in numeric_cols if c != "audio_label_binary"]
    agg = ["mean", "std", "median", lambda s: s.quantile(0.05), lambda s: s.quantile(0.95)]
    rename = {"<lambda_0>": "p5", "<lambda_1>": "p95"}

    def _do(group_cols: list[str], filename: str) -> None:
        if any(c not in features.columns for c in group_cols):
            return
        summary = features.groupby(group_cols)[numeric_cols].agg(agg)
        # collapse multiindex columns to flat names like "rms_mean", "rms_p5", ...
        summary.columns = [
            f"{col}_{rename.get(stat, stat)}" for col, stat in summary.columns
        ]
        summary = summary.reset_index()
        _atomic_write_csv(summary, out_dir / filename)

    _do(["audio_label_binary"], "summary_by_label.csv")
    _do(["audio_label_binary", "provider"], "summary_by_label_provider.csv")
    _do(["source_folder"], "summary_by_source_folder.csv")


def _split_arrays(
    features: pd.DataFrame, feature_cols: list[str], split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract X, y, providers for a given split."""
    import numpy as np
    sub = features[features["split"] == split]
    X = sub[feature_cols].to_numpy()
    y = sub["audio_label_binary"].to_numpy().astype(int)
    providers = sub["provider"].to_numpy()
    return X, y, providers


def _run_default_probes(
    features: pd.DataFrame, feature_cols: list[str], *, seed: int,
) -> dict:
    """Train default probes (LR, RF, per-feature LR sweep) and return eval dicts."""
    import numpy as np
    from src.analysis.acoustic_probe_models import (
        evaluate,
        fit_logistic,
        fit_random_forest,
        per_feature_lr_sweep,
    )

    if len(features) == 0:
        return {}
    X_train, y_train, _ = _split_arrays(features, feature_cols, "train")
    X_val, y_val, providers_val = _split_arrays(features, feature_cols, "val")
    if len(X_train) == 0 or len(X_val) == 0:
        return {"skipped": "empty_split"}
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return {"skipped": "single_class"}

    lr = fit_logistic(X_train, y_train, seed=seed)
    rf = fit_random_forest(X_train, y_train, seed=seed)

    lr_eval = evaluate(lr, X_val, y_val, providers_val=providers_val)
    rf_eval = evaluate(rf, X_val, y_val, providers_val=providers_val)
    rf_eval["feature_importances"] = {
        name: float(score)
        for name, score in zip(feature_cols, rf.feature_importances_)
    }

    sweep = per_feature_lr_sweep(
        X_train, y_train, X_val, y_val, feature_cols, seed=seed,
    )

    return {"lr": lr_eval, "rf": rf_eval, "per_feature_lr": sweep}


def _run_loeo(
    features: pd.DataFrame, feature_cols: list[str], *, seed: int,
) -> list[dict]:
    """Run leave-one-engine-out (LOEO) evaluation."""
    from src.analysis.acoustic_probe_models import loeo_matrix

    if len(features) == 0:
        return []
    return loeo_matrix(features, feature_cols, seed=seed)


def _make_plots(
    features: pd.DataFrame,
    default_results: dict,
    feature_cols: list[str],
    out_dir: Path,
) -> None:
    """Generate matplotlib figures to out_dir/figures/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if len(features) == 0 or not default_results:
        return None

    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    def _hist_by_label(col: str, filename: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, label_name in [(0, "bonafide"), (1, "spoof")]:
            slice_ = features.loc[features["audio_label_binary"] == label, col].dropna()
            if len(slice_) == 0:
                continue
            ax.hist(slice_, bins=30, alpha=0.5, label=label_name)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figs_dir / filename, dpi=120)
        plt.close(fig)

    _hist_by_label("rms", "rms_by_label.png", "RMS by label")
    _hist_by_label("silence_ratio", "silence_ratio_by_label.png", "Silence ratio by label")
    _hist_by_label(
        "spectral_centroid_mean",
        "spectral_centroid_by_label.png",
        "Spectral centroid mean by label",
    )

    # Correlation heatmap
    corr = features[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=90, fontsize=6)
    ax.set_yticklabels(feature_cols, fontsize=6)
    ax.set_title("Feature correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figs_dir / "feature_correlation_heatmap.png", dpi=120)
    plt.close(fig)

    # ROC curves
    val = features[features["split"] == "val"]
    y_val = val["audio_label_binary"].to_numpy().astype(int)
    for tag, key, filename in (("LR", "lr", "roc_lr.png"), ("RF", "rf", "roc_rf.png")):
        eval_ = default_results.get(key, {})
        scores = eval_.get("scores")
        if scores is None or len(scores) != len(y_val):
            continue
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_val, np.asarray(scores))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, label=f"{tag} AUC={eval_['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{tag} ROC (val)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figs_dir / filename, dpi=120)
        plt.close(fig)

    # Per-feature LR bar chart
    sweep = default_results.get("per_feature_lr", [])
    if sweep:
        names = [r["feature"] for r in sweep]
        aucs = [r["val_roc_auc"] for r in sweep]
        fig, ax = plt.subplots(figsize=(7, max(3, 0.25 * len(names))))
        ax.barh(range(len(names)), aucs)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("val ROC-AUC")
        ax.set_title("Per-feature LR sweep")
        fig.tight_layout()
        fig.savefig(figs_dir / "per_feature_lr_auc_bar.png", dpi=120)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    cfg = _parse_args(argv)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(cfg.manifest)
    _validate_manifest(manifest)
    if cfg.max_rows is not None:
        manifest = manifest.head(cfg.max_rows).copy()

    feat_cols = feature_columns(with_f0=cfg.with_f0)
    expected_columns = CACHE_METADATA_COLUMNS + feat_cols

    try:
        cache = _load_or_init_cache(out_dir, expected_columns, cfg.force)
    except CacheSchemaError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    features, bad_rows = _extract_features_resumable(
        manifest, cache, out_dir, with_f0=cfg.with_f0, force=cfg.force,
    )

    _run_summaries(features, out_dir)
    default_results = _run_default_probes(features, feat_cols, seed=cfg.seed)
    loeo_results = _run_loeo(features, feat_cols, seed=cfg.seed) if cfg.loeo else []

    if not cfg.no_plots:
        _make_plots(features, default_results, feat_cols, out_dir)

    metrics = {
        "config": asdict(cfg),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_manifest_rows": int(len(manifest)),
        "n_features_cached": int(len(features)),
        "feature_columns": feat_cols,
        "default_probes": default_results,
        "loeo": loeo_results,
        "bad_rows": bad_rows,
    }
    _atomic_write_json(metrics, out_dir / "probe_metrics.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
