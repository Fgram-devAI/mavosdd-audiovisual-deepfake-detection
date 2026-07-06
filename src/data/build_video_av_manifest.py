"""Build video-level AV fake manifest from native own-audio video rows."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src import common


FIELDS = (
    "sample_id",
    "source_video_id",
    "split",
    "source_folder",
    "video_path",
    "audio_path",
    "audio_sample_id",
    "video_label",
    "video_label_binary",
)


def _video_label_for_source(source_folder: str) -> tuple[str, str]:
    if source_folder == "real":
        return "real", "0"
    return "fake", "1"


def build_manifest(*, source: Path, out: Path, splits: set[str]) -> int:
    if "test" in splits:
        raise ValueError("test split is locked; refuse to build video AV test rows")
    with source.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if row["split"] not in splits:
                continue
            if row["provider"] != "original":
                continue
            if row["media_type"] != "pair":
                continue
            video_label, video_label_binary = _video_label_for_source(row["source_folder"])
            rows.append({
                "sample_id": row["sample_id"],
                "source_video_id": row["source_video_id"],
                "split": row["split"],
                "source_folder": row["source_folder"],
                "video_path": row["video_path"],
                "audio_path": row["audio_path"],
                "audio_sample_id": row["sample_id"],
                "video_label": video_label,
                "video_label_binary": video_label_binary,
            })

    rows.sort(key=lambda r: (r["split"], r["source_folder"], r["source_video_id"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=common.VISUAL_SPEECH_MANIFEST_VOICE_SPLIT)
    parser.add_argument("--out", type=Path, default=common.VIDEO_AV_MANIFEST)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args(argv)

    try:
        n = build_manifest(source=args.source, out=args.out, splits=set(args.splits))
    except ValueError as exc:
        print(str(exc))
        return 2
    print(f"wrote {n} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
