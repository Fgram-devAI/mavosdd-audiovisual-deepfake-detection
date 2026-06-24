"""Build derived speech-detection manifests from raw + generated assets."""
from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")


def load_split_map(splits_dir: Path) -> dict[str, str]:
    """Return {video_id: split} from train.csv, val.csv, test.csv.

    Raises ValueError when a video_id appears in more than one split file.
    """
    splits_dir = Path(splits_dir)
    mapping: dict[str, str] = {}
    duplicates: list[str] = []
    for split in SPLIT_NAMES:
        path = splits_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"split file missing: {path}")
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                vid = row["video_id"]
                if vid in mapping and mapping[vid] != split:
                    duplicates.append(f"{vid} (in {mapping[vid]} and {split})")
                mapping[vid] = split
    if duplicates:
        raise ValueError(f"split leakage: {', '.join(duplicates)}")
    return mapping
