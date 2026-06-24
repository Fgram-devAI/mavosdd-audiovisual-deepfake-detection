"""Build derived speech-detection manifests from raw + generated assets."""
from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")

SCHEMA: tuple[str, ...] = (
    "sample_id",
    "source_video_id",
    "split",
    "media_type",
    "source_folder",
    "provider",
    "voice_id_or_name",
    "audio_path",
    "video_path",
    "audio_feature_path",
    "lip_feature_path",
    "audio_label",
    "audio_label_binary",
    "video_label",
    "video_label_binary",
    "pair_label",
    "pair_label_binary",
)

VIDEO_LABEL_BY_SOURCE: dict[str, str] = {
    "real": "real",
    "echomimic": "fake",
    "memo": "fake",
}

VIDEO_LABEL_BINARY_BY_STRING: dict[str, int | str] = {
    "real": 0,
    "fake": 1,
    "na": "",
}

AUDIO_LABEL_BINARY_BY_STRING: dict[str, int] = {
    "bonafide": 0,
    "spoof": 1,
}

PAIR_LABEL_BINARY_BY_STRING: dict[str, int | str] = {
    "matched_bonafide": 0,
    "generated_same_transcript": 1,
    "mismatched_negative": 1,
    "na": "",
}


def _native_video_path(source_folder: str, video_id: str) -> str:
    return f"data/raw/{source_folder}/{video_id}.mp4"


def _native_audio_path(source_folder: str, video_id: str) -> str:
    return f"data/audio_wav/{source_folder}/{video_id}.wav"


def _native_audio_feature_path(video_id: str) -> str:
    return f"data/features/audio/{video_id}.npy"


def _native_lip_feature_path(video_id: str) -> str:
    return f"data/features/lips/{video_id}.npz"


def iter_native_rows(manifest_path: Path, split_map: dict[str, str]) -> list[dict]:
    """Yield one bonafide row dict per native video. Videos with no split are skipped+warned."""
    rows: list[dict] = []
    skipped: list[str] = []
    with Path(manifest_path).open(newline="") as f:
        for raw in csv.DictReader(f):
            vid = raw["video_id"]
            split = split_map.get(vid)
            if split is None:
                skipped.append(vid)
                continue
            src = raw["source_folder"]
            video_label = VIDEO_LABEL_BY_SOURCE.get(src, "na")
            rows.append({
                "sample_id": vid,
                "source_video_id": vid,
                "split": split,
                "media_type": "video",
                "source_folder": src,
                "provider": "original",
                "voice_id_or_name": "",
                "audio_path": _native_audio_path(src, vid),
                "video_path": _native_video_path(src, vid),
                "audio_feature_path": _native_audio_feature_path(vid),
                "lip_feature_path": _native_lip_feature_path(vid),
                "audio_label": "bonafide",
                "audio_label_binary": AUDIO_LABEL_BINARY_BY_STRING["bonafide"],
                "video_label": video_label,
                "video_label_binary": VIDEO_LABEL_BINARY_BY_STRING[video_label],
                "pair_label": "na",
                "pair_label_binary": "",
            })
    if skipped:
        logger.warning("iter_native_rows: %d native rows skipped (no split): %s",
                       len(skipped), skipped[:5])
    return rows


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
