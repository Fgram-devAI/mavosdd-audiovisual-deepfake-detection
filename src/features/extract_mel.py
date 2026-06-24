"""CLI: extract log-power mel-spectrograms into data/features/audio_mel/.

Reads rows from data/derived/audio_spoof_manifest.csv (or any compatible CSV),
applies optional --split / --limit filters, resolves each row's WAV via the
`audio_path` column, runs a deterministic 4.0 s @ 16 kHz window through
torchaudio MelSpectrogram + AmplitudeToDB(power), and writes
``{out_dir}/{sample_id}.npy`` arrays of shape ``(n_mels, T)`` dtype float32.

The manifest is never mutated. Existing outputs are skipped unless
``--overwrite`` is passed. Failures log ``[FAIL] {sample_id}: {reason}`` and
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
import torchaudio
from tqdm import tqdm

from src import common
from src.features.audio_io import load_audio_window

logger = logging.getLogger(__name__)


class MelExtractor:
    """Deterministic log-power mel-spectrogram from a fixed-length mono wave."""

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        n_mels: int,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
        )
        self._to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def extract(self, wave: np.ndarray) -> np.ndarray:
        if wave.ndim != 1:
            raise ValueError(f"MelExtractor expects 1-D mono wave, got shape {wave.shape}")
        t = torch.from_numpy(wave.astype(np.float32, copy=False)).unsqueeze(0)
        with torch.no_grad():
            spec = self._to_db(self._mel(t)).squeeze(0)
        arr = spec.cpu().numpy().astype(np.float32, copy=False)
        return arr


def iter_manifest_rows(
    manifest_path: Path,
    *,
    split: str | None,
    limit: int | None,
) -> Iterator[dict]:
    yielded = 0
    with Path(manifest_path).open(newline="") as f:
        for row in csv.DictReader(f):
            if split is not None and row.get("split") != split:
                continue
            yield row
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def extract(
    manifest_path: Path,
    mel: MelExtractor,
    out_dir: Path,
    *,
    split: str | None,
    limit: int | None,
    overwrite: bool,
    seconds: float,
) -> dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"written": 0, "skipped": 0, "failed": 0}
    rows = list(iter_manifest_rows(manifest_path, split=split, limit=limit))
    for row in tqdm(rows, desc="mel-spec", unit="row"):
        sample_id = row.get("sample_id") or "<missing>"
        out_path = out_dir / f"{sample_id}.npy"
        if out_path.exists() and not overwrite:
            counts["skipped"] += 1
            continue
        try:
            if sample_id == "<missing>":
                raise KeyError("sample_id column missing from manifest row")
            audio_path = row.get("audio_path") or ""
            if not audio_path:
                raise ValueError("audio_path is empty")
            wave = load_audio_window(audio_path, sr=mel.sample_rate, seconds=seconds)
            arr = mel.extract(wave)
            np.save(out_path, arr)
            counts["written"] += 1
        except Exception as exc:  # noqa: BLE001 — extractor must continue past per-row errors
            tqdm.write(f"[FAIL] {sample_id}: {exc}")
            counts["failed"] += 1
    return counts


def _load_mel_config() -> dict:
    """Load the `mel:` block from config/default.yaml as a plain dict."""
    import yaml
    cfg_path = Path(__file__).resolve().parent.parent.parent / "config" / "default.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    return cfg["features"]["mel"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="extract_mel",
        description="Extract log-power mel-spectrograms into data/features/audio_mel/.",
    )
    p.add_argument("--manifest", type=Path, default=common.AUDIO_SPOOF_MANIFEST,
                   help="Path to a CSV with sample_id, audio_path, split columns.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override the default FEAT_AUDIO_MEL_DIR output directory.")
    p.add_argument("--split", choices=("train", "val", "test"), default=None,
                   help="Restrict to a single split.")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after iterating this many post-filter rows "
                        "(skipped + written count toward limit).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract even when {out_dir}/{sample_id}.npy already exists.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir if args.out_dir is not None else common.FEAT_AUDIO_MEL_DIR

    cfg = _load_mel_config()
    mel = MelExtractor(
        sample_rate=cfg["sample_rate"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        win_length=cfg["win_length"],
        n_mels=cfg["n_mels"],
    )
    counts = extract(
        args.manifest, mel, out_dir,
        split=args.split,
        limit=args.limit,
        overwrite=args.overwrite,
        seconds=cfg["seconds"],
    )
    print(
        f"out_dir={out_dir} "
        f"written={counts['written']} skipped={counts['skipped']} failed={counts['failed']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
