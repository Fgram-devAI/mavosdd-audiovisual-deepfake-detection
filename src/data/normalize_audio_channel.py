"""CLI: deterministic audio-channel normalization.

Reads a derived audio manifest, writes 16 kHz mono PCM-16 WAV copies with
consistent loudness / trimming / low-pass, and emits a matching manifest
alongside a failures CSV, a fallback log, and a run-provenance JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.data import audio_normalize as an

logger = logging.getLogger(__name__)

REQUIRED_INPUT_COLUMNS = (
    "sample_id",
    "source_video_id",
    "split",
    "source_folder",
    "provider",
    "audio_path",
    "audio_label_binary",
)

SCHEMA_ORDER = (
    "sample_id", "source_video_id", "split", "media_type", "source_folder",
    "provider", "voice_id_or_name", "audio_path", "video_path",
    "audio_feature_path", "lip_feature_path", "audio_label",
    "audio_label_binary", "video_label", "video_label_binary",
    "pair_label", "pair_label_binary",
)


@dataclass(frozen=True)
class CLIConfig:
    manifest: str
    out_manifest: str
    out_dir: str
    failures_csv: str
    target_sr: int
    lufs: float
    lowpass_hz: float
    trim_top_db: float
    max_failure_rate: float
    max_group_failure_rate: float
    overwrite: bool
    limit: int | None
    dry_run: bool
    no_trim: bool
    no_loudness: bool
    no_lowpass: bool
    seed: int


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="normalize_audio_channel",
        description="Deterministic audio-channel normalization pipeline.",
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-manifest",
                   default="data/derived/audio_spoof_manifest_normalized.csv")
    p.add_argument("--out-dir", default="data/derived/audio_normalized")
    p.add_argument("--failures-csv",
                   default="data/derived/audio_normalize_failures.csv")
    p.add_argument("--target-sr", type=int, default=16000)
    p.add_argument("--lufs", type=float, default=-23.0)
    p.add_argument("--lowpass-hz", type=float, default=7000.0)
    p.add_argument("--trim-top-db", type=float, default=30.0)
    p.add_argument("--max-failure-rate", type=float, default=0.02)
    p.add_argument("--max-group-failure-rate", type=float, default=0.05)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-trim", action="store_true")
    p.add_argument("--no-loudness", action="store_true")
    p.add_argument("--no-lowpass", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


def _parse_args(argv: list[str] | None) -> CLIConfig:
    ns = build_parser().parse_args(argv)
    return CLIConfig(
        manifest=ns.manifest,
        out_manifest=ns.out_manifest,
        out_dir=ns.out_dir,
        failures_csv=ns.failures_csv,
        target_sr=ns.target_sr,
        lufs=ns.lufs,
        lowpass_hz=ns.lowpass_hz,
        trim_top_db=ns.trim_top_db,
        max_failure_rate=ns.max_failure_rate,
        max_group_failure_rate=ns.max_group_failure_rate,
        overwrite=ns.overwrite,
        limit=ns.limit,
        dry_run=ns.dry_run,
        no_trim=ns.no_trim,
        no_loudness=ns.no_loudness,
        no_lowpass=ns.no_lowpass,
        seed=ns.seed,
    )


def _validate_startup(cfg: CLIConfig) -> str | None:
    """Return an error string, or None if startup validation passes."""
    manifest_path = Path(cfg.manifest)
    if not manifest_path.exists():
        return f"missing input manifest: {manifest_path}"
    if cfg.lowpass_hz >= cfg.target_sr / 2.0:
        return (
            f"--lowpass-hz {cfg.lowpass_hz} must be strictly below Nyquist "
            f"(target_sr/2 = {cfg.target_sr / 2})"
        )
    # Read header only.
    with manifest_path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return f"manifest has no header: {manifest_path}"
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in header]
    if missing:
        return f"manifest missing required columns: {missing}"
    return None


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = _parse_args(argv)

    err = _validate_startup(cfg)
    if err is not None:
        print(err, file=sys.stderr)
        return 1

    # The row loop, failure/fallback logging, threshold guards, and provenance
    # JSON are implemented in later tasks. For now the scaffold returns 0
    # after startup validation so smoke tests can lock the exit-1 semantics.
    print(f"normalize_audio_channel: startup ok manifest={cfg.manifest}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
