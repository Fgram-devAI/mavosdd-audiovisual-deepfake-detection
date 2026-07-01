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


def _iter_manifest(path: Path, limit: int | None):
    yielded = 0
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            yield row
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def _load_audio_native(audio_path: str):
    import librosa
    try:
        wave, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as exc:  # noqa: BLE001
        raise an.DecodeError(f"decode_failed: {exc}") from exc
    if wave.size == 0:
        raise an.DecodeError("empty_audio")
    return wave, int(sr)


def _write_wav_atomic(path: Path, wave, sr: int) -> None:
    import numpy as np
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(suffix=".wav", prefix=path.stem + "_", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_str)
    try:
        sf.write(str(tmp), np.asarray(wave, dtype=np.float32), sr, subtype="PCM_16")
    except Exception as exc:  # noqa: BLE001
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise an.WriteError(f"write_failed: {exc}") from exc
    os.replace(tmp, path)


def _write_csv_atomic(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    os.close(fd)
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in fieldnames})
    os.replace(tmp, path)


def _normalize_row(row: dict, cfg: CLIConfig, out_path: Path) -> tuple[list[dict], bool]:
    """Run the transform chain for one row.

    Returns (fallback_records, skipped_existing). Raises AudioNormalizeError
    subclasses on stage failure with the stage name baked into __class__.
    """
    fallbacks: list[dict] = []
    if out_path.exists() and not cfg.overwrite:
        return fallbacks, True

    wave, sr = _load_audio_native(row["audio_path"])

    try:
        wave = an.resample_mono_16k(wave, sr=sr, target_sr=cfg.target_sr)
    except an.ResampleError:
        raise

    if not cfg.no_trim:
        wave, trim_fallback = an.trim_silence(
            wave, sr=cfg.target_sr, top_db=cfg.trim_top_db,
        )
        if trim_fallback:
            fallbacks.append({
                "sample_id": row["sample_id"], "stage": "trim",
                "condition": "trim_fallback=short",
            })

    if not cfg.no_lowpass:
        wave, lp_skipped = an.lowpass(
            wave, sr=cfg.target_sr, cutoff_hz=cfg.lowpass_hz,
        )
        if lp_skipped:
            fallbacks.append({
                "sample_id": row["sample_id"], "stage": "lowpass",
                "condition": "lowpass_skipped=too_short",
            })

    if not cfg.no_loudness:
        wave, silence_skipped = an.loudness_normalize(
            wave, sr=cfg.target_sr, target_lufs=cfg.lufs,
        )
        if silence_skipped:
            fallbacks.append({
                "sample_id": row["sample_id"], "stage": "loudness",
                "condition": "loudness_skipped=silence",
            })
        wave = an.peak_safety(wave)

    if not cfg.dry_run:
        _write_wav_atomic(out_path, wave, cfg.target_sr)

    return fallbacks, False


_STAGE_BY_EXCEPTION = {
    an.DecodeError: "decode",
    an.ResampleError: "resample",
    an.TrimError: "trim",
    an.LowpassError: "lowpass",
    an.LoudnessError: "loudness",
    an.PeakSafetyError: "peak_safety",
    an.WriteError: "write",
    an.PathTokenError: "path",
}


def _stage_from_exc(exc: BaseException) -> str:
    for klass, name in _STAGE_BY_EXCEPTION.items():
        if isinstance(exc, klass):
            return name
    return "unknown"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = _parse_args(argv)

    err = _validate_startup(cfg)
    if err is not None:
        print(err, file=sys.stderr)
        return 1

    out_dir = Path(cfg.out_dir)
    out_manifest = Path(cfg.out_manifest)
    failures_csv = Path(cfg.failures_csv)
    fallbacks_csv = out_dir / "_fallbacks.csv"

    # Truncate failures CSV on --overwrite; otherwise carry forward existing rows.
    prior_failures: list[dict] = []
    if failures_csv.exists() and not cfg.overwrite:
        with failures_csv.open() as f:
            prior_failures = list(csv.DictReader(f))

    written_rows: list[dict] = []
    new_failures: list[dict] = []
    all_fallbacks: list[dict] = []
    n_skipped_existing = 0

    for row in _iter_manifest(Path(cfg.manifest), cfg.limit):
        sid = row.get("sample_id", "")
        provider = row.get("provider", "")
        try:
            safe_provider = an.validate_path_token(provider, field="provider")
            safe_sid = an.validate_path_token(sid, field="sample_id")
        except an.PathTokenError as exc:
            new_failures.append({
                "sample_id": sid, "audio_path": row.get("audio_path", ""),
                "provider": provider, "split": row.get("split", ""),
                "stage": "path", "reason": str(exc),
            })
            continue

        out_path = out_dir / safe_provider / f"{safe_sid}.wav"

        try:
            fallbacks, skipped = _normalize_row(row, cfg, out_path)
        except an.AudioNormalizeError as exc:
            new_failures.append({
                "sample_id": sid, "audio_path": row.get("audio_path", ""),
                "provider": provider, "split": row.get("split", ""),
                "stage": _stage_from_exc(exc), "reason": str(exc),
            })
            continue

        all_fallbacks.extend(fallbacks)
        if skipped:
            n_skipped_existing += 1

        out_row = {c: row.get(c, "") for c in SCHEMA_ORDER}
        out_row["original_audio_path"] = row.get("audio_path", "")
        out_row["audio_path"] = str(out_path)
        written_rows.append(out_row)

    fieldnames_out = list(SCHEMA_ORDER) + ["original_audio_path"]
    if not cfg.dry_run:
        _write_csv_atomic(out_manifest, written_rows, fieldnames_out)
    _write_csv_atomic(
        failures_csv,
        prior_failures + new_failures,
        ["sample_id", "audio_path", "provider", "split", "stage", "reason"],
    )
    if all_fallbacks:
        _write_csv_atomic(
            fallbacks_csv, all_fallbacks,
            ["sample_id", "stage", "condition"],
        )

    n_valid = len(written_rows)
    n_written = 0 if cfg.dry_run else n_valid
    n_failed = len(new_failures)
    print(
        f"normalize_audio_channel: valid={n_valid} written={n_written} "
        f"skipped_existing={n_skipped_existing} failed={n_failed} dry_run={cfg.dry_run}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
