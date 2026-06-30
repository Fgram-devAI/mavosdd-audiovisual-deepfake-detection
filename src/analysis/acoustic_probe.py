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
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

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
    Path(tmp).write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


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
        cache = pd.concat(
            [cache, pd.DataFrame(new_rows, columns=expected_columns)],
            ignore_index=True,
        )
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
    snapshot = pd.concat(
        [cache, pd.DataFrame(new_rows, columns=expected_columns)],
        ignore_index=True,
    )
    _atomic_write_csv(snapshot[expected_columns], out_dir / "acoustic_features.csv")
    _atomic_write_csv(
        pd.DataFrame(all_failures, columns=["sample_id", "audio_path", "reason"]),
        out_dir / "acoustic_failures.csv",
    )


def _run_summaries(features: pd.DataFrame, out_dir: Path) -> None:
    """Filled in by Task 6."""
    # Placeholder so Task 5 tests still produce a usable JSON. No-op here.
    return None


def _run_default_probes(
    features: pd.DataFrame, feature_cols: list[str], *, seed: int,
) -> dict:
    """Filled in by Task 6."""
    return {}


def _run_loeo(
    features: pd.DataFrame, feature_cols: list[str], *, seed: int,
) -> list[dict]:
    """Filled in by Task 7."""
    return []


def _make_plots(
    features: pd.DataFrame,
    default_results: dict,
    feature_cols: list[str],
    out_dir: Path,
) -> None:
    """Filled in by Task 8."""
    return None


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
