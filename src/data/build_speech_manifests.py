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


def _generated_sample_id(provider: str, source_video_id: str, voice: str) -> str:
    return f"{provider}__{source_video_id}__voice-{voice}"


def _generated_feature_path(sample_id: str) -> str:
    return f"data/features/audio_generated/{sample_id}.npy"


def iter_generated_rows(
    tts_records: list[dict],
    split_map: dict[str, str],
    source_folder_map: dict[str, str] | None = None,
) -> tuple[list[dict], list[str]]:
    """Build spoof rows. Returns (rows, excluded_source_video_ids).

    Generated rows inherit split from their source_video_id. Records whose
    source_video_id is missing from split_map are excluded with a warning.
    """
    rows: list[dict] = []
    excluded: list[str] = []
    folder_map = source_folder_map or {}
    for rec in tts_records:
        sv = rec["source_video_id"]
        split = split_map.get(sv)
        if split is None:
            excluded.append(sv)
            continue
        provider = rec["provider"]
        voice = rec.get("voice", "")
        sample_id = _generated_sample_id(provider, sv, voice)
        source_folder = rec.get("source_folder") or folder_map.get(sv, "")
        rows.append({
            "sample_id": sample_id,
            "source_video_id": sv,
            "split": split,
            "media_type": "audio",
            "source_folder": source_folder,
            "provider": provider,
            "voice_id_or_name": voice,
            "audio_path": rec.get("synthetic_audio_path", ""),
            "video_path": "",
            "audio_feature_path": _generated_feature_path(sample_id),
            "lip_feature_path": "",
            "audio_label": "spoof",
            "audio_label_binary": AUDIO_LABEL_BINARY_BY_STRING["spoof"],
            "video_label": "na",
            "video_label_binary": VIDEO_LABEL_BINARY_BY_STRING["na"],
            "pair_label": "na",
            "pair_label_binary": "",
        })
    if excluded:
        logger.warning(
            "iter_generated_rows: %d records excluded (unknown source_video_id): %s",
            len(excluded), excluded[:5],
        )
    return rows, excluded


def write_manifest(rows: list[dict], path: Path) -> None:
    """Write rows to `path` with SCHEMA columns. Missing fields become empty strings."""
    path = Path(path)
    schema_set = set(SCHEMA)
    for row in rows:
        unknown = set(row) - schema_set
        if unknown:
            raise ValueError(f"unknown column(s) in row: {sorted(unknown)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SCHEMA))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in SCHEMA})


def _native_source_folder_map(rows: list[dict]) -> dict[str, str]:
    return {r["source_video_id"]: r["source_folder"] for r in rows}


def build_audio_spoof_manifest(
    manifest_path: Path,
    splits_dir: Path,
    tts_dir: Path,
    out_path: Path,
    providers: list[str],
) -> dict:
    split_map = load_split_map(splits_dir)
    native = iter_native_rows(manifest_path, split_map)
    folder_map = _native_source_folder_map(native)
    tts_records = iter_tts_records(tts_dir, providers)
    generated, excluded = iter_generated_rows(tts_records, split_map, folder_map)
    write_manifest(native + generated, out_path)
    return {
        "native_rows": len(native),
        "generated_rows": len(generated),
        "excluded": excluded,
    }


def _matched_pair_row(native: dict) -> dict:
    return {
        "sample_id": f"pos__{native['source_video_id']}",
        "source_video_id": native["source_video_id"],
        "split": native["split"],
        "media_type": "pair",
        "source_folder": native["source_folder"],
        "provider": "original",
        "voice_id_or_name": "",
        "audio_path": native["audio_path"],
        "video_path": native["video_path"],
        "audio_feature_path": native["audio_feature_path"],
        "lip_feature_path": native["lip_feature_path"],
        "audio_label": "bonafide",
        "audio_label_binary": AUDIO_LABEL_BINARY_BY_STRING["bonafide"],
        "video_label": native["video_label"],
        "video_label_binary": VIDEO_LABEL_BINARY_BY_STRING[native["video_label"]],
        "pair_label": "matched_bonafide",
        "pair_label_binary": PAIR_LABEL_BINARY_BY_STRING["matched_bonafide"],
    }


def _generated_pair_row(gen: dict, native: dict) -> dict:
    return {
        "sample_id": gen["sample_id"],
        "source_video_id": gen["source_video_id"],
        "split": gen["split"],
        "media_type": "pair",
        "source_folder": gen["source_folder"] or native["source_folder"],
        "provider": gen["provider"],
        "voice_id_or_name": gen["voice_id_or_name"],
        "audio_path": gen["audio_path"],
        "video_path": native["video_path"],
        "audio_feature_path": gen["audio_feature_path"],
        "lip_feature_path": native["lip_feature_path"],
        "audio_label": "spoof",
        "audio_label_binary": AUDIO_LABEL_BINARY_BY_STRING["spoof"],
        "video_label": native["video_label"],
        "video_label_binary": VIDEO_LABEL_BINARY_BY_STRING[native["video_label"]],
        "pair_label": "generated_same_transcript",
        "pair_label_binary": PAIR_LABEL_BINARY_BY_STRING["generated_same_transcript"],
    }


def build_visual_speech_manifest(
    manifest_path: Path,
    splits_dir: Path,
    tts_dir: Path,
    out_path: Path,
    providers: list[str],
) -> dict:
    split_map = load_split_map(splits_dir)
    native = iter_native_rows(manifest_path, split_map)
    native_by_id = {r["source_video_id"]: r for r in native}
    tts_records = iter_tts_records(tts_dir, providers)
    generated, _ = iter_generated_rows(
        tts_records, split_map, _native_source_folder_map(native)
    )

    matched_rows = [_matched_pair_row(n) for n in native]

    pair_rows: list[dict] = []
    excluded_no_native: list[str] = []
    for g in generated:
        n = native_by_id.get(g["source_video_id"])
        if n is None:
            excluded_no_native.append(g["source_video_id"])
            continue
        pair_rows.append(_generated_pair_row(g, n))

    if excluded_no_native:
        logger.warning(
            "build_visual_speech_manifest: %d generated rows excluded (no native lip source): %s",
            len(excluded_no_native), excluded_no_native[:5],
        )

    write_manifest(matched_rows + pair_rows, out_path)
    return {
        "matched_rows": len(matched_rows),
        "generated_rows": len(pair_rows),
        "excluded_no_native_source": excluded_no_native,
    }


def build_fusion_speech_manifest(
    manifest_path: Path,
    splits_dir: Path,
    tts_dir: Path,
    out_path: Path,
    providers: list[str],
) -> dict:
    split_map = load_split_map(splits_dir)
    native = iter_native_rows(manifest_path, split_map)
    native_by_id = {r["source_video_id"]: r for r in native}
    tts_records = iter_tts_records(tts_dir, providers)
    generated, _ = iter_generated_rows(
        tts_records, split_map, _native_source_folder_map(native)
    )

    bonafide_rows = [_matched_pair_row(n) for n in native]

    spoof_rows: list[dict] = []
    for g in generated:
        n = native_by_id.get(g["source_video_id"])
        if n is None:
            continue
        spoof_rows.append(_generated_pair_row(g, n))

    write_manifest(bonafide_rows + spoof_rows, out_path)
    return {
        "bonafide_rows": len(bonafide_rows),
        "spoof_rows": len(spoof_rows),
    }
