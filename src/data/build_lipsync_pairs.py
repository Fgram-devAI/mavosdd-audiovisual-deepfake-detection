"""Deterministic pair-manifest builder for the lip-sync consistency branch."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from src import common

logger = logging.getLogger(__name__)

LIPSYNC_MANIFEST_SCHEMA: tuple[str, ...] = (
    "pair_id", "split", "source_video_id", "lip_feature_path",
    "audio_sample_id", "audio_path", "audio_feature_path",
    "audio_provider", "audio_label",
    "sync_label", "sync_label_binary", "negative_type",
    "source_folder", "voice_id_or_name",
)

SYNC_LABEL_BINARY: dict[str, int] = {"sync": 0, "async": 1}

NEGATIVE_TYPES: tuple[str, ...] = (
    "generated_same_transcript",
    "mismatched_original",
    "mismatched_generated",
)


def read_fusion_manifest(path: Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _positive_row(matched: dict) -> dict:
    vid = matched["source_video_id"]
    return {
        "pair_id": f"pos__{vid}",
        "split": matched["split"],
        "source_video_id": vid,
        "lip_feature_path": matched["lip_feature_path"],
        "audio_sample_id": matched["sample_id"],
        "audio_path": matched["audio_path"],
        "audio_feature_path": matched["audio_feature_path"],
        "audio_provider": matched["provider"] or "original",
        "audio_label": matched["audio_label"],
        "sync_label": "sync",
        "sync_label_binary": str(SYNC_LABEL_BINARY["sync"]),
        "negative_type": "",
        "source_folder": matched["source_folder"],
        "voice_id_or_name": matched["voice_id_or_name"],
    }


def _negative_row(
    matched: dict, other_audio: dict, negative_type: str, index: int
) -> dict:
    return {
        "pair_id": f"neg__{negative_type}__{matched['source_video_id']}__{index}__{other_audio['sample_id']}",
        "split": matched["split"],
        "source_video_id": matched["source_video_id"],
        "lip_feature_path": matched["lip_feature_path"],
        "audio_sample_id": other_audio["sample_id"],
        "audio_path": other_audio["audio_path"],
        "audio_feature_path": other_audio["audio_feature_path"],
        "audio_provider": other_audio["provider"] or "original",
        "audio_label": other_audio["audio_label"],
        "sync_label": "async",
        "sync_label_binary": str(SYNC_LABEL_BINARY["async"]),
        "negative_type": negative_type,
        "source_folder": matched["source_folder"],
        "voice_id_or_name": other_audio.get("voice_id_or_name", ""),
    }


def build_pairs(
    rows: list[dict],
    *,
    negatives_per_positive: int,
    splits: tuple[str, ...] = ("train", "val"),
    seed: int = 42,
) -> list[dict]:
    if negatives_per_positive < 0:
        raise ValueError(f"negatives_per_positive must be >= 0, got {negatives_per_positive}")

    rng = np.random.default_rng(seed)
    out: list[dict] = []

    by_split_matched: dict[str, list[dict]] = {}
    by_split_generated: dict[str, list[dict]] = {}
    by_split_generated_same: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        split = r.get("split", "")
        if split not in splits:
            continue
        if r.get("pair_label") == "matched_bonafide":
            by_split_matched.setdefault(split, []).append(r)
        elif r.get("pair_label") == "generated_same_transcript":
            by_split_generated.setdefault(split, []).append(r)
            by_split_generated_same.setdefault(split, {}).setdefault(
                r["source_video_id"], []
            ).append(r)

    for split in splits:
        matched_rows = by_split_matched.get(split, [])
        matched_rows_sorted = sorted(matched_rows, key=lambda r: r["source_video_id"])

        for matched in matched_rows_sorted:
            out.append(_positive_row(matched))

            if negatives_per_positive == 0:
                continue

            same_transcript = by_split_generated_same.get(split, {}).get(
                matched["source_video_id"], []
            )
            same_transcript_sorted = sorted(same_transcript, key=lambda r: r["sample_id"])
            n_gst = min(negatives_per_positive, len(same_transcript_sorted))
            for i, cand in enumerate(same_transcript_sorted[:n_gst]):
                out.append(_negative_row(matched, cand, "generated_same_transcript", i))

            mm_orig_pool = [
                r for r in matched_rows_sorted
                if r["source_video_id"] != matched["source_video_id"]
            ]
            if mm_orig_pool:
                picks = rng.choice(
                    len(mm_orig_pool),
                    size=min(negatives_per_positive, len(mm_orig_pool)),
                    replace=False,
                )
                for i, idx in enumerate(sorted(int(x) for x in picks)):
                    out.append(_negative_row(matched, mm_orig_pool[idx], "mismatched_original", i))
            else:
                logger.warning(
                    "build_pairs: no mismatched_original candidates for %s in split %s",
                    matched["source_video_id"], split,
                )

            mm_gen_pool = [
                r for r in by_split_generated.get(split, [])
                if r["source_video_id"] != matched["source_video_id"]
            ]
            mm_gen_pool_sorted = sorted(mm_gen_pool, key=lambda r: r["sample_id"])
            if mm_gen_pool_sorted:
                picks = rng.choice(
                    len(mm_gen_pool_sorted),
                    size=min(negatives_per_positive, len(mm_gen_pool_sorted)),
                    replace=False,
                )
                for i, idx in enumerate(sorted(int(x) for x in picks)):
                    out.append(_negative_row(
                        matched, mm_gen_pool_sorted[idx], "mismatched_generated", i,
                    ))
            else:
                logger.warning(
                    "build_pairs: no mismatched_generated candidates for %s in split %s",
                    matched["source_video_id"], split,
                )

    return out


def write_pair_manifest(rows: list[dict], out_path: Path) -> None:
    out_path = Path(out_path)
    allowed = set(LIPSYNC_MANIFEST_SCHEMA)
    for row in rows:
        unknown = set(row) - allowed
        if unknown:
            raise ValueError(f"unknown column(s): {sorted(unknown)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(LIPSYNC_MANIFEST_SCHEMA))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in LIPSYNC_MANIFEST_SCHEMA})


def write_provenance(
    rows: list[dict],
    out_path: Path,
    *,
    source_manifest: Path,
    negatives_per_positive: int,
    seed: int,
) -> None:
    payload = {
        "source_manifest": str(source_manifest),
        "seed": seed,
        "negatives_per_positive": negatives_per_positive,
        "total": len(rows),
        "by_split": dict(Counter(r["split"] for r in rows)),
        "by_sync_label": dict(Counter(r["sync_label"] for r in rows)),
        "by_negative_type": dict(Counter(r["negative_type"] for r in rows)),
        "by_audio_provider": dict(Counter(r["audio_provider"] for r in rows)),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the deterministic lip-sync pair manifest.",
    )
    p.add_argument("--source-manifest", type=Path,
                   default=common.FUSION_SPEECH_MANIFEST_VOICE_SPLIT)
    p.add_argument("--out-path", type=Path,
                   default=common.LIPSYNC_PAIRS_MANIFEST)
    p.add_argument("--provenance-path", type=Path,
                   default=common.LIPSYNC_PAIRS_PROVENANCE)
    p.add_argument("--negatives-per-positive", type=int, default=1)
    p.add_argument("--seed", type=int, default=common.SEED)
    p.add_argument("--splits", nargs="+", default=("train", "val"),
                   choices=("train", "val"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rows = read_fusion_manifest(args.source_manifest)
    pairs = build_pairs(
        rows,
        negatives_per_positive=args.negatives_per_positive,
        splits=tuple(args.splits),
        seed=args.seed,
    )
    write_pair_manifest(pairs, args.out_path)
    write_provenance(
        pairs, args.provenance_path,
        source_manifest=args.source_manifest,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
    )
    print(
        f"lipsync_pairs: total={len(pairs)} "
        f"by_split={dict(Counter(r['split'] for r in pairs))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
