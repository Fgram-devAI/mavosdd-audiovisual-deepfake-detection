"""CLI: extract pretrained audio embeddings into backend-specific .npy stores.

Reads rows from data/derived/audio_spoof_manifest.csv (or any compatible CSV),
applies optional split/provider/limit filters, drives one frozen backend, and
writes ``{out_dir}/{sample_id}.npy`` arrays of shape (time, output_dim).

The manifest is never mutated. Existing outputs are skipped unless
``--overwrite`` is passed. Failures log "[FAIL] {sample_id}: {reason}" and
continue with the next row.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from tqdm import tqdm

from src import common
from src.features.audio_backends import (
    AudioEmbeddingBackend,
    list_backends,
    load_backend,
)
from src.features.audio_io import load_audio_window

logger = logging.getLogger(__name__)

DEFAULT_DIR_BY_BACKEND: dict[str, Path] = {
    "wav2vec2": common.FEAT_AUDIO_WAV2VEC2_DIR,
    "wavlm": common.FEAT_AUDIO_WAVLM_DIR,
    "hubert": common.FEAT_AUDIO_HUBERT_DIR,
}

DTYPE_MAP: dict[str, type] = {"float16": np.float16, "float32": np.float32}

DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def pick_device(name: str = "auto") -> torch.device:
    """Resolve a torch device. ``auto`` picks cuda > mps > cpu."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def default_out_dir(backend_name: str) -> Path:
    try:
        return DEFAULT_DIR_BY_BACKEND[backend_name]
    except KeyError as exc:
        raise ValueError(
            f"unknown audio backend: {backend_name!r}. "
            f"Registered: {sorted(DEFAULT_DIR_BY_BACKEND)}"
        ) from exc


def iter_manifest_rows(
    manifest_path: Path,
    *,
    split: str | None,
    source_providers: tuple[str, ...] | None,
    limit: int | None,
) -> Iterator[dict]:
    yielded = 0
    with Path(manifest_path).open(newline="") as f:
        for row in csv.DictReader(f):
            if split is not None and row.get("split") != split:
                continue
            if source_providers and row.get("provider") not in source_providers:
                continue
            yield row
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def extract(
    manifest_path: Path,
    backend: AudioEmbeddingBackend,
    out_dir: Path,
    *,
    split: str | None,
    source_providers: tuple[str, ...] | None,
    limit: int | None,
    overwrite: bool,
    dtype: str,
) -> dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_dtype = DTYPE_MAP[dtype]

    counts = {"written": 0, "skipped": 0, "failed": 0}
    rows = list(iter_manifest_rows(
        manifest_path,
        split=split,
        source_providers=source_providers,
        limit=limit,
    ))
    for row in tqdm(rows, desc=f"{backend.name} embeddings", unit="row"):
        sample_id = row.get("sample_id") or "<missing>"
        out_path = out_dir / f"{sample_id}.npy"
        if out_path.exists() and not overwrite:
            counts["skipped"] += 1
            continue
        try:
            if sample_id == "<missing>":
                raise KeyError("sample_id column missing from manifest row")
            wave = load_audio_window(
                row["audio_path"],
                sr=backend.sample_rate,
                seconds=4.0,
            )
            arr = backend.encode(wave).astype(target_dtype, copy=False)
            np.save(out_path, arr)
            counts["written"] += 1
        except Exception as exc:  # noqa: BLE001 — extractor must continue past per-row errors
            tqdm.write(f"[FAIL] {sample_id}: {exc}")
            counts["failed"] += 1
    return counts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="extract_audio_embeddings",
        description="Extract frozen pretrained audio embeddings into backend-specific .npy stores.",
    )
    p.add_argument("--backend", required=True, choices=list_backends(),
                   help="Encoder name. Determines the default --out-dir.")
    p.add_argument("--manifest", required=True, type=Path,
                   help="Path to a CSV with sample_id, audio_path, split, provider columns.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override the backend-specific default output directory.")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after iterating this many post-filter rows (skipped + written count toward limit).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract even when {out_dir}/{sample_id}.npy already exists.")
    p.add_argument("--source-provider", action="append", default=None,
                   help="Repeat to whitelist providers (e.g. --source-provider elevenlabs).")
    p.add_argument("--split", choices=("train", "val", "test"), default=None,
                   help="Restrict to a single split.")
    p.add_argument("--dtype", choices=tuple(DTYPE_MAP), default="float16",
                   help="Numpy dtype written to disk. Default: float16.")
    p.add_argument("--device", choices=DEVICE_CHOICES, default="auto",
                   help="Compute device. 'auto' picks cuda > mps > cpu.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir if args.out_dir is not None else default_out_dir(args.backend)
    providers = tuple(args.source_provider) if args.source_provider else None
    device = pick_device(args.device)
    print(f"backend={args.backend} device={device}")

    backend = load_backend(args.backend, device)
    counts = extract(
        args.manifest, backend, out_dir,
        split=args.split,
        source_providers=providers,
        limit=args.limit,
        overwrite=args.overwrite,
        dtype=args.dtype,
    )
    print(
        f"backend={args.backend} out_dir={out_dir} "
        f"written={counts['written']} skipped={counts['skipped']} failed={counts['failed']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
