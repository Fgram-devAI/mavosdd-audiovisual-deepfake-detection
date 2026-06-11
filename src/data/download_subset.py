"""Stream MAVOS-DD, keep english/{real,echomimic,memo}, stop at 1,000 videos."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import ffmpeg
from datasets import load_dataset

from src.common import CAPS, LABEL_MAP, MANIFEST, QUARANTINE_DIR, QUARANTINE_LOG, RAW_DIR

DATASET_ID = "unibuc-cs/MAVOS-DD"

QUARANTINE_REASONS = {"unreadable", "no_frames", "zero_fps", "no_audio_stream"}


def classify(record: dict) -> str | None:
    """Return the target source folder, or None when a record is out of scope."""
    lang = str(record.get("language", "")).lower()
    method = str(record.get("generation_method", record.get("method", ""))).lower()
    path = str(record.get("file_name", record.get("path", ""))).lower()

    if lang == "english" and method in CAPS:
        return method
    for cls in CAPS:
        if f"english/{cls}/" in path:
            return cls
    return None


def record_id(record: dict, fallback_idx: int) -> str:
    raw_path = record.get("file_name", record.get("path", f"{fallback_idx}.mp4"))
    return Path(str(raw_path)).stem


def video_payload(record: dict) -> bytes:
    video_obj = record["video"]
    if isinstance(video_obj, dict):
        payload = video_obj.get("bytes")
    else:
        payload = video_obj
    if not isinstance(payload, bytes):
        raise TypeError(f"Unsupported video payload type: {type(payload)!r}")
    return payload


def probe_video(path: Path) -> tuple[str | None, float, float, int]:
    """Return (rejection_reason, duration_s, fps, n_frames).

    rejection_reason is None when the file is readable with positive fps + frames.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return "unreadable", 0.0, 0.0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if n_frames <= 0:
        return "no_frames", 0.0, fps, n_frames
    if fps <= 0:
        return "zero_fps", 0.0, fps, n_frames
    duration = float(n_frames / fps)
    return None, duration, fps, n_frames


def has_audio_stream(path: Path) -> bool:
    """True iff the file carries at least one audio stream (per ffprobe)."""
    try:
        info = ffmpeg.probe(str(path))
    except ffmpeg.Error:
        return False
    except Exception:
        return False
    streams = info.get("streams", []) if isinstance(info, dict) else []
    return any(s.get("codec_type") == "audio" for s in streams)


def quarantine_file(src: Path, video_id: str, source_folder: str, reason: str) -> Path:
    """Move `src` to data/quarantine/<source_folder>/<video_id>.mp4 and log the reason."""
    assert reason in QUARANTINE_REASONS, f"Unknown reason: {reason}"
    target_dir = QUARANTINE_DIR / source_folder
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{video_id}.mp4"
    if src.exists():
        src.replace(target)
    QUARANTINE_LOG.parent.mkdir(parents=True, exist_ok=True)
    new_file = not QUARANTINE_LOG.exists()
    with QUARANTINE_LOG.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["video_id", "source_folder", "reason"])
        writer.writerow([video_id, source_folder, reason])
    return target


def load_existing_state() -> tuple[dict[str, int], set[str], set[str]]:
    """Rebuild (counts, done_ids, quarantined_ids) from the on-disk CSVs."""
    counts = {k: 0 for k in CAPS}
    done_ids: set[str] = set()
    quarantined_ids: set[str] = set()

    if MANIFEST.exists():
        with MANIFEST.open(newline="") as f:
            for row in csv.DictReader(f):
                done_ids.add(row["video_id"])
                counts[row["source_folder"]] += 1

    if QUARANTINE_LOG.exists():
        with QUARANTINE_LOG.open(newline="") as f:
            for row in csv.DictReader(f):
                quarantined_ids.add(row["video_id"])

    return counts, done_ids, quarantined_ids


def main() -> None:
    counts, done_ids = load_existing_counts()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    stream = load_dataset(DATASET_ID, split="train", streaming=True)

    new_file = not MANIFEST.exists()
    with MANIFEST.open("a", newline="") as mf:
        writer = csv.writer(mf)
        if new_file:
            writer.writerow(
                [
                    "video_id",
                    "relative_path",
                    "source_folder",
                    "binary_label",
                    "duration_s",
                    "fps",
                    "n_frames",
                ]
            )

        for i, record in enumerate(stream):
            if all(counts[k] >= CAPS[k] for k in CAPS):
                break

            cls = classify(record)
            if cls is None or counts[cls] >= CAPS[cls]:
                continue

            video_id = record_id(record, i)
            if video_id in done_ids:
                continue

            out_dir = RAW_DIR / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{video_id}.mp4"
            out_path.write_bytes(video_payload(record))

            duration_s, fps, n_frames = probe_video(out_path)
            writer.writerow(
                [
                    video_id,
                    str(out_path),
                    cls,
                    LABEL_MAP[cls],
                    f"{duration_s:.3f}",
                    f"{fps:.3f}",
                    n_frames,
                ]
            )
            counts[cls] += 1
            done_ids.add(video_id)
            print(f"\r{counts}", end="")

    total = sum(counts.values())
    assert total <= sum(CAPS.values()), f"Cap breached: {total}"
    print(f"\nIngestion complete: {counts} | total={total}")


if __name__ == "__main__":
    main()
