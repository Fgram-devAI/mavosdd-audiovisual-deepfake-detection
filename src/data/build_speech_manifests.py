"""Build derived speech-detection manifests from raw + generated assets."""
from __future__ import annotations

import csv
import json
import logging
import re
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


PROVIDER_DIR: dict[str, str] = {
    "elevenlabs": "elevenlabs",
    "google_tts": "google_tts",
    "elevenlabs_sts": "elevenlabs_sts",
}

PROVIDER_JSONL: dict[str, str] = {
    "elevenlabs": "manifest.jsonl",
    "google_tts": "google_tts_manifest.jsonl",
    "elevenlabs_sts": "sts_manifest.jsonl",
}

_FILENAME_RE = re.compile(r"^(?P<sv>.+)__voice-(?P<voice>.+)\.mp3$")


def parse_tts_filename(name: str) -> tuple[str, str] | None:
    """Extract (source_video_id, voice) from `{id}__voice-{voice}.mp3` or return None."""
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    return m.group("sv"), m.group("voice")


def _normalize_jsonl_record(provider: str, rec: dict) -> dict | None:
    sv = rec.get("video_id")
    if not sv:
        return None
    voice = rec.get("voice_id") or rec.get("voice_name") or ""
    return {
        "provider": provider,
        "source_video_id": sv,
        "voice": voice,
        "synthetic_audio_path": rec.get("synthetic_audio_path", ""),
        "source_folder": rec.get("source_folder", ""),
    }


def _scan_provider_dir(provider: str, provider_dir: Path) -> list[dict]:
    found: list[dict] = []
    if not provider_dir.exists():
        return found
    for mp3 in provider_dir.rglob("*.mp3"):
        parsed = parse_tts_filename(mp3.name)
        if parsed is None:
            logger.warning("unparseable tts filename: %s", mp3)
            continue
        sv, voice = parsed
        source_folder = mp3.parent.name if mp3.parent != provider_dir else ""
        found.append({
            "provider": provider,
            "source_video_id": sv,
            "voice": voice,
            "synthetic_audio_path": str(mp3),
            "source_folder": source_folder,
        })
    return found


def iter_tts_records(tts_dir: Path, providers: list[str]) -> list[dict]:
    """Return normalized TTS records for each requested provider.

    JSONL is preferred per provider when present; otherwise the provider's
    subdirectory is walked for *.mp3 files.
    """
    tts_dir = Path(tts_dir)
    out: list[dict] = []
    for provider in providers:
        jsonl_name = PROVIDER_JSONL.get(provider)
        jsonl_path = tts_dir / jsonl_name if jsonl_name else None
        if jsonl_path and jsonl_path.exists():
            with jsonl_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    norm = _normalize_jsonl_record(provider, rec)
                    if norm is not None:
                        out.append(norm)
            continue
        subdir = PROVIDER_DIR.get(provider)
        if subdir:
            out.extend(_scan_provider_dir(provider, tts_dir / subdir))
    return out
