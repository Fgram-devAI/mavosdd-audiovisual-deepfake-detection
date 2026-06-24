"""Split-safe PyTorch datasets and validation over cached feature stores.

This module is the single bridge between derived manifests (under
``data/derived/``) and training code. It never reads raw audio or video.

Public API:
    FeatureStoreValidationError
    AudioFeatureDataset
    VisualFeatureDataset
    FusionFeatureDataset
    fit_normalization_stats
    feature_collate
    make_dataloader
    validate_feature_store
"""
from __future__ import annotations

from pathlib import Path

from src import common


class FeatureStoreValidationError(Exception):
    """Raised when a manifest, cached feature file, or shape contract is invalid."""


AUDIO_FEATURE_DIM: int = 768
SUPPORTED_BACKENDS: tuple[str, ...] = ("wav2vec2", "wavlm", "hubert")

AUDIO_BACKEND_DIRS: dict[str, Path] = {
    "wav2vec2": common.FEAT_AUDIO_WAV2VEC2_DIR,
    "wavlm": common.FEAT_AUDIO_WAVLM_DIR,
    "hubert": common.FEAT_AUDIO_HUBERT_DIR,
}


def resolve_audio_backend_dir(backend: str) -> Path:
    """Return the default audio feature root for a backend short name."""
    try:
        return AUDIO_BACKEND_DIRS[backend]
    except KeyError as exc:
        raise FeatureStoreValidationError(
            f"unknown audio backend: {backend!r}. "
            f"Supported: {SUPPORTED_BACKENDS}"
        ) from exc


import csv as _csv

import numpy as np
import torch
from torch.utils.data import Dataset


_METADATA_KEYS: tuple[str, ...] = (
    "sample_id", "source_video_id", "split", "provider", "source_folder",
)


def _read_manifest_rows(manifest_path: Path | str) -> list[dict]:
    with Path(manifest_path).open(newline="") as f:
        return list(_csv.DictReader(f))


def _filter_split(rows: list[dict], split: str) -> list[dict]:
    return [r for r in rows if r.get("split") == split]


def _row_metadata(row: dict) -> dict:
    return {k: row.get(k, "") for k in _METADATA_KEYS}


def _label_long(row: dict, column: str) -> torch.Tensor:
    raw = row.get(column, "")
    if raw == "" or raw is None:
        raise FeatureStoreValidationError(
            f"manifest row {row.get('sample_id', '?')!r} missing label column {column!r}"
        )
    return torch.tensor(int(raw), dtype=torch.long)


def _load_audio_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FeatureStoreValidationError(f"missing audio feature: {path}")
    arr = np.load(path)
    if arr.ndim != 2:
        raise FeatureStoreValidationError(
            f"audio feature {path} must be rank-2, got shape {arr.shape}"
        )
    if arr.shape[1] != AUDIO_FEATURE_DIM:
        raise FeatureStoreValidationError(
            f"audio feature {path} feature dim {arr.shape[1]} != {AUDIO_FEATURE_DIM}"
        )
    if arr.shape[0] <= 0:
        raise FeatureStoreValidationError(
            f"audio feature {path} time dim must be positive, got {arr.shape[0]}"
        )
    return arr.astype(np.float32, copy=False)


class AudioFeatureDataset(Dataset):
    """Audio-only dataset over a backend-specific feature store."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        backend: str,
        audio_dir: Path | None = None,
    ) -> None:
        self._rows = _filter_split(_read_manifest_rows(manifest_path), split)
        self._audio_dir = Path(audio_dir) if audio_dir is not None else resolve_audio_backend_dir(backend)
        self.backend = backend
        self.split = split

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        audio_path = self._audio_dir / f"{row['sample_id']}.npy"
        arr = _load_audio_array(audio_path)
        item = {
            "audio": torch.from_numpy(arr),
            "label": _label_long(row, "audio_label_binary"),
        }
        item.update(_row_metadata(row))
        return item


def _load_lip_array(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FeatureStoreValidationError(f"missing lip feature: {path}")
    with np.load(path) as npz:
        if "feats" not in npz.files:
            raise FeatureStoreValidationError(
                f"lip feature {path} missing 'feats' key (found {list(npz.files)!r})"
            )
        if "mask" not in npz.files:
            raise FeatureStoreValidationError(
                f"lip feature {path} missing 'mask' key (found {list(npz.files)!r})"
            )
        feats = np.asarray(npz["feats"])
        mask = np.asarray(npz["mask"])
    if not np.issubdtype(feats.dtype, np.floating) and not np.issubdtype(feats.dtype, np.integer):
        raise FeatureStoreValidationError(
            f"lip feature {path} 'feats' must be numeric, got dtype {feats.dtype}"
        )
    if feats.ndim != 2:
        raise FeatureStoreValidationError(
            f"lip feature {path} 'feats' must be rank-2, got shape {feats.shape}"
        )
    if feats.shape[0] <= 0 or feats.shape[1] <= 0:
        raise FeatureStoreValidationError(
            f"lip feature {path} 'feats' has non-positive dim: shape {feats.shape}"
        )
    if mask.ndim != 1:
        raise FeatureStoreValidationError(
            f"lip feature {path} 'mask' must be 1-D (rank-1), got shape {mask.shape}"
        )
    if mask.shape[0] != feats.shape[0]:
        raise FeatureStoreValidationError(
            f"lip feature {path} 'mask' shape {mask.shape} does not match "
            f"feats time dim {feats.shape[0]}"
        )
    if not np.isfinite(feats).all():
        raise FeatureStoreValidationError(
            f"lip feature {path} 'feats' contains NaN or Inf"
        )
    return feats.astype(np.float32, copy=False), mask.astype(np.float32, copy=False)


class VisualFeatureDataset(Dataset):
    """Lip-only dataset over data/features/lips/{source_video_id}.npz."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        lips_dir: Path | None = None,
    ) -> None:
        self._rows = _filter_split(_read_manifest_rows(manifest_path), split)
        self._lips_dir = Path(lips_dir) if lips_dir is not None else common.FEAT_LIPS_DIR
        self.split = split

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        lip_path = self._lips_dir / f"{row['source_video_id']}.npz"
        feats, mask = _load_lip_array(lip_path)
        item = {
            "lips": torch.from_numpy(feats),
            "lips_mask": torch.from_numpy(mask),
            "label": _label_long(row, "pair_label_binary"),
        }
        item.update(_row_metadata(row))
        return item


class FusionFeatureDataset(Dataset):
    """Late-fusion dataset returning audio embeddings + lip features for one row."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        backend: str,
        audio_dir: Path | None = None,
        lips_dir: Path | None = None,
    ) -> None:
        self._rows = _filter_split(_read_manifest_rows(manifest_path), split)
        self._audio_dir = Path(audio_dir) if audio_dir is not None else resolve_audio_backend_dir(backend)
        self._lips_dir = Path(lips_dir) if lips_dir is not None else common.FEAT_LIPS_DIR
        self.backend = backend
        self.split = split

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        audio_arr = _load_audio_array(self._audio_dir / f"{row['sample_id']}.npy")
        feats, mask = _load_lip_array(self._lips_dir / f"{row['source_video_id']}.npz")
        item = {
            "audio": torch.from_numpy(audio_arr),
            "lips": torch.from_numpy(feats),
            "lips_mask": torch.from_numpy(mask),
            "label": _label_long(row, "pair_label_binary"),
        }
        item.update(_row_metadata(row))
        return item


# ---------- Task 5: validate_feature_store + CLI ----------

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field


_MAX_ERR_LIST = 10


@dataclass(frozen=True)
class ValidationReport:
    view: str
    backend: str | None
    manifest_rows: int
    split_counts: dict[str, int]
    label_counts: dict[str, int]
    missing: list[str] = field(default_factory=list)
    bad_shape: list[str] = field(default_factory=list)
    path_mismatches: list[str] = field(default_factory=list)
    truncated_missing: int = 0
    truncated_bad_shape: int = 0
    truncated_path_mismatches: int = 0


_LABEL_COLUMN_BY_VIEW = {
    "audio": "audio_label_binary",
    "visual": "pair_label_binary",
    "fusion": "pair_label_binary",
}


def _check_audio_row(row: dict, audio_dir: Path) -> tuple[str | None, str | None, str | None]:
    sid = row.get("sample_id", "?")
    path = audio_dir / f"{sid}.npy"
    manifest_path = row.get("audio_feature_path", "") or ""
    mismatch = None
    if manifest_path and Path(manifest_path) != path:
        mismatch = f"{sid}: reconstructed={path} manifest={manifest_path}"
    if not path.exists():
        return f"{sid}: {path} not found", None, mismatch
    try:
        _load_audio_array(path)
    except FeatureStoreValidationError as exc:
        return None, f"{sid}: {exc}", mismatch
    return None, None, mismatch


def _check_lip_row(row: dict, lips_dir: Path) -> tuple[str | None, str | None, str | None]:
    vid = row.get("source_video_id", "?")
    sid = row.get("sample_id", vid)
    path = lips_dir / f"{vid}.npz"
    manifest_path = row.get("lip_feature_path", "") or ""
    mismatch = None
    if manifest_path and Path(manifest_path) != path:
        mismatch = f"{sid}: reconstructed={path} manifest={manifest_path}"
    if not path.exists():
        return f"{vid}: {path} not found", None, mismatch
    try:
        _load_lip_array(path)
    except FeatureStoreValidationError as exc:
        return None, f"{vid}: {exc}", mismatch
    return None, None, mismatch


def validate_feature_store(
    view: str,
    manifest_path: str | Path,
    *,
    backend: str | None = None,
    audio_dir: Path | None = None,
    lips_dir: Path | None = None,
) -> ValidationReport:
    if view not in _LABEL_COLUMN_BY_VIEW:
        raise FeatureStoreValidationError(
            f"unknown view: {view!r}. Supported: {sorted(_LABEL_COLUMN_BY_VIEW)}"
        )
    if view in {"audio", "fusion"} and backend is None:
        raise FeatureStoreValidationError(f"backend is required for view={view!r}")

    audio_root = (
        Path(audio_dir) if audio_dir is not None
        else (resolve_audio_backend_dir(backend) if backend else None)
    )
    lips_root = Path(lips_dir) if lips_dir is not None else common.FEAT_LIPS_DIR

    rows = _read_manifest_rows(manifest_path)
    split_counts = Counter(r.get("split", "?") for r in rows)
    label_counts = Counter(r.get(_LABEL_COLUMN_BY_VIEW[view], "") for r in rows)

    missing: list[str] = []
    bad_shape: list[str] = []
    path_mismatches: list[str] = []
    truncated_missing = 0
    truncated_bad_shape = 0
    truncated_path_mismatches = 0

    def _record_missing(msg: str) -> None:
        nonlocal truncated_missing
        if len(missing) < _MAX_ERR_LIST:
            missing.append(msg)
        else:
            truncated_missing += 1

    def _record_bad(msg: str) -> None:
        nonlocal truncated_bad_shape
        if len(bad_shape) < _MAX_ERR_LIST:
            bad_shape.append(msg)
        else:
            truncated_bad_shape += 1

    def _record_mismatch(msg: str) -> None:
        nonlocal truncated_path_mismatches
        if len(path_mismatches) < _MAX_ERR_LIST:
            path_mismatches.append(msg)
        else:
            truncated_path_mismatches += 1

    for row in rows:
        if view in {"audio", "fusion"}:
            miss, bad, mismatch = _check_audio_row(row, audio_root)
            if miss:
                _record_missing(miss)
            if bad:
                _record_bad(bad)
            if mismatch:
                _record_mismatch(mismatch)
        if view in {"visual", "fusion"}:
            miss, bad, mismatch = _check_lip_row(row, lips_root)
            if miss:
                _record_missing(miss)
            if bad:
                _record_bad(bad)
            if mismatch:
                _record_mismatch(mismatch)

    return ValidationReport(
        view=view,
        backend=backend,
        manifest_rows=len(rows),
        split_counts=dict(split_counts),
        label_counts=dict(label_counts),
        missing=missing,
        bad_shape=bad_shape,
        path_mismatches=path_mismatches,
        truncated_missing=truncated_missing,
        truncated_bad_shape=truncated_bad_shape,
        truncated_path_mismatches=truncated_path_mismatches,
    )


_DEFAULT_MANIFEST_BY_VIEW = {
    "audio": common.AUDIO_SPOOF_MANIFEST,
    "visual": common.VISUAL_SPEECH_MANIFEST,
    "fusion": common.FUSION_SPEECH_MANIFEST,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="feature_store",
        description="Validate cached feature stores referenced by a derived manifest.",
    )
    p.add_argument("--validate", action="store_true", required=True,
                   help="Run the validation pipeline. Currently the only mode.")
    p.add_argument("--view", required=True, choices=("audio", "visual", "fusion"),
                   help="Dataset view to validate.")
    p.add_argument("--backend", choices=SUPPORTED_BACKENDS, default=None,
                   help="Audio backend. Required for audio/fusion views.")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Override the default manifest path for the chosen view.")
    p.add_argument("--audio-dir", type=Path, default=None,
                   help="Override the backend audio feature root.")
    p.add_argument("--lips-dir", type=Path, default=None,
                   help="Override the lip feature root.")
    return p


def _print_report(report: ValidationReport) -> None:
    print(f"view={report.view} backend={report.backend} rows={report.manifest_rows}")
    print(f"split_counts={report.split_counts}")
    print(f"label_counts={report.label_counts}")
    print(
        f"missing={len(report.missing) + report.truncated_missing} "
        f"bad_shape={len(report.bad_shape) + report.truncated_bad_shape} "
        f"path_mismatches={len(report.path_mismatches) + report.truncated_path_mismatches}"
    )
    for entry in report.missing[:5]:
        print(f"  MISSING {entry}")
    for entry in report.bad_shape[:5]:
        print(f"  BAD_SHAPE {entry}")
    for entry in report.path_mismatches[:5]:
        print(f"  PATH_MISMATCH {entry}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest if args.manifest is not None else _DEFAULT_MANIFEST_BY_VIEW[args.view]
    report = validate_feature_store(
        args.view,
        manifest_path,
        backend=args.backend,
        audio_dir=args.audio_dir,
        lips_dir=args.lips_dir,
    )
    _print_report(report)
    has_errors = bool(report.missing) or bool(report.bad_shape) \
        or report.truncated_missing or report.truncated_bad_shape
    return 1 if has_errors else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
