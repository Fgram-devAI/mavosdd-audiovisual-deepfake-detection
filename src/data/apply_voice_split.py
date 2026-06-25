"""Rewrite a visual/fusion manifest's `split` column from a source voice-split manifest.

Preserves every non-`split` column byte-identically. Fails loudly if any target
`sample_id` is absent from the source map. CLI:

    python -m src.data.apply_voice_split \
        --source data/derived/audio_spoof_manifest_voice_split.csv \
        --target data/derived/visual_speech_manifest.csv \
        --out    data/derived/visual_speech_manifest_voice_split.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src import common


def _read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "sample_id" not in fieldnames or "split" not in fieldnames:
        raise ValueError(
            f"{path}: manifest must have 'sample_id' and 'split' columns, got {fieldnames}"
        )
    return fieldnames, rows


def apply_voice_split(
    source_manifest: str | Path,
    target_manifest: str | Path,
    out_path: str | Path,
) -> None:
    src_path = Path(source_manifest)
    tgt_path = Path(target_manifest)
    out = Path(out_path)

    _, src_rows = _read_rows(src_path)
    sample_to_split: dict[str, str] = {}
    for r in src_rows:
        sid = r["sample_id"]
        sample_to_split[sid] = r["split"]

    tgt_fields, tgt_rows = _read_rows(tgt_path)

    missing = [r["sample_id"] for r in tgt_rows if r["sample_id"] not in sample_to_split]
    if missing:
        raise ValueError(
            f"{len(missing)} target sample_id(s) missing from source voice-split: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tgt_fields)
        writer.writeheader()
        for r in tgt_rows:
            new_row = dict(r)
            new_row["split"] = sample_to_split[r["sample_id"]]
            writer.writerow(new_row)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="apply_voice_split",
        description="Rewrite a manifest's 'split' column from a source voice-split manifest.",
    )
    p.add_argument("--source", type=Path, default=common.AUDIO_SPOOF_MANIFEST_VOICE_SPLIT,
                   help="Source manifest providing {sample_id -> split} (default: audio voice-split).")
    p.add_argument("--target", required=True, type=Path,
                   help="Manifest whose split column will be remapped.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output manifest path.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    apply_voice_split(args.source, args.target, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
