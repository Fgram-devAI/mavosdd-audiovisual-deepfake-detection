"""Stream MAVOS-DD, keep english/{real,echomimic,memo}, stop at 1,000 videos."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
from datasets import load_dataset

from src.common import CAPS, LABEL_MAP, MANIFEST, RAW_DIR

DATASET_ID = "unibuc-cs/MAVOS-DD"


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


def load_existing_counts() -> tuple[dict[str, int], set[str]]:
    counts = {k: 0 for k in CAPS}
    done_ids: set[str] = set()
    if not MANIFEST.exists():
        return counts, done_ids

    with MANIFEST.open(newline="") as f:
        for row in csv.DictReader(f):
            done_ids.add(row["video_id"])
            counts[row["source_folder"]] += 1
    return counts, done_ids


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
