"""Export raw MP4 audio tracks to mono 16 kHz WAV files.

Usage:
    python scripts/export_wav.py
    python scripts/export_wav.py --limit 10
    python scripts/export_wav.py --overwrite
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from tqdm import tqdm


def export_one(video_path: Path, output_path: Path, overwrite: bool) -> bool:
    """Return True when a WAV was created, False when skipped."""
    if output_path.exists() and not overwrite:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y" if overwrite else "-n",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_path),
        ],
        check=True,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Export data/raw MP4 audio to WAV.")
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--out-dir", default="data/audio_wav", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="Export only the first N videos.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing WAV files.")
    args = parser.parse_args()

    videos = sorted(args.raw_dir.glob("*/*.mp4"))
    if args.limit is not None:
        videos = videos[: args.limit]

    created = 0
    failed = 0
    for video_path in tqdm(videos, desc="wav export", unit="video"):
        rel = video_path.relative_to(args.raw_dir)
        output_path = args.out_dir / rel.with_suffix(".wav")
        try:
            if export_one(video_path, output_path, overwrite=args.overwrite):
                created += 1
        except subprocess.CalledProcessError as exc:
            failed += 1
            tqdm.write(f"[FAIL] {video_path}: {exc}")

    total = len(videos)
    existing = len(list(args.out_dir.glob("*/*.wav")))
    print(f"WAV export complete: created={created} failed={failed} selected={total} total_wav={existing}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
