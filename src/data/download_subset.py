"""Fetch MAVOS-DD english/{real,echomimic,memo,liveportrait,sonic}.

Per-class caps live in `src.common.CAPS`. Re-running this script after a CAPS
change is safe: `load_existing_state` rebuilds per-class counts from the
manifest and the downloader only fetches whatever is still missing up to the
new caps (no duplicate downloads).
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

import cv2
import ffmpeg
from huggingface_hub import HfApi, hf_hub_download

from src.common import CAPS, LABEL_MAP, MANIFEST, QUARANTINE_DIR, QUARANTINE_LOG, RAW_DIR

DATASET_ID = "unibuc-cs/MAVOS-DD"

QUARANTINE_REASONS = {"unreadable", "no_frames", "zero_fps", "no_audio_stream"}

logger = logging.getLogger(__name__)


def inspect_schema(record: dict) -> None:
    """Warn if expected metadata fields are missing on the first record.

    Hard-fails (KeyError) when no file source exists. Supported sources are a
    streamed `video` payload, a downloaded `local_path`, or deferred Hub download
    metadata (`repo_id` + `path`).
    """
    has_deferred_hub_source = "repo_id" in record and "path" in record
    if "video" not in record and "local_path" not in record and not has_deferred_hub_source:
        raise KeyError("record missing required video source")
    if "language" not in record:
        logger.warning(
            "first record has no 'language' field; falling back to path-prefix filter"
        )
    if "generation_method" not in record and "method" not in record:
        logger.warning(
            "first record has no 'generation_method'/'method' field; falling back to path-prefix filter"
        )


def classify(record: dict) -> str | None:
    """Return the target source folder, or None when a record is out of scope.

    Metadata is authoritative when present: a non-english `language` field, or a
    `generation_method`/`method` outside the cap set, both reject the record
    without falling back to the path. The path-prefix fallback only fires when
    the relevant metadata field is missing.
    """
    lang_raw = record.get("language")
    method_raw = record.get("generation_method", record.get("method"))
    path = str(record.get("file_name", record.get("path", ""))).lower()

    lang = str(lang_raw).lower() if lang_raw is not None else None
    method = str(method_raw).lower() if method_raw is not None else None

    # Language present but not english -> reject outright.
    if lang is not None and lang != "english":
        return None
    # Generator present but out of scope -> reject outright.
    if method is not None and method != "" and method not in CAPS:
        return None

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


def target_repo_files(dataset_id: str = DATASET_ID) -> list[str]:
    """List only the MAVOS-DD files under english/{real,echomimic,memo}/."""
    print(f"Listing files for {dataset_id} ...", flush=True)
    files = HfApi().list_repo_files(dataset_id, repo_type="dataset")
    targets = []
    for path in files:
        lower = path.lower()
        if not lower.endswith(".mp4"):
            continue
        if any(lower.startswith(f"english/{cls}/") for cls in CAPS):
            targets.append(path)
    targets.sort()
    print(f"Found {len(targets)} target mp4 files under english/*", flush=True)
    return targets


def iter_candidate_records(dataset_id: str = DATASET_ID):
    """Yield target english mp4 metadata without downloading yet."""
    for path in target_repo_files(dataset_id):
        cls = path.split("/", 2)[1].lower()
        yield {
            "language": "english",
            "generation_method": cls,
            "file_name": path,
            "path": path,
            "repo_id": dataset_id,
        }


def materialize_video(record: dict, out_path: Path) -> None:
    """Persist a candidate record to the raw-data path expected by the pipeline."""
    if "local_path" in record:
        shutil.copy2(record["local_path"], out_path)
        return
    if "repo_id" in record:
        local_path = hf_hub_download(
            repo_id=record["repo_id"],
            repo_type="dataset",
            filename=record["path"],
        )
        shutil.copy2(local_path, out_path)
        return
    out_path.write_bytes(video_payload(record))


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


MANIFEST_HEADER = [
    "video_id", "relative_path", "source_folder", "binary_label",
    "duration_s", "fps", "n_frames",
]


def _open_manifest_writer(handle, write_header: bool) -> csv.writer:
    writer = csv.writer(handle)
    if write_header:
        writer.writerow(MANIFEST_HEADER)
    return writer


def main() -> None:
    counts, done_ids, quarantined_ids = load_existing_state()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    records = iter(iter_candidate_records(DATASET_ID))

    new_manifest = not MANIFEST.exists()
    inspected = False
    with MANIFEST.open("a", newline="") as mf:
        writer = _open_manifest_writer(mf, write_header=new_manifest)

        i = -1
        while True:
            if all(counts[k] >= CAPS[k] for k in CAPS):
                break
            try:
                record = next(records)
            except StopIteration:
                break
            i += 1

            if not inspected:
                inspect_schema(record)
                inspected = True

            cls = classify(record)
            if cls is None or counts[cls] >= CAPS[cls]:
                continue

            video_id = record_id(record, i)
            if video_id in done_ids or video_id in quarantined_ids:
                continue

            out_dir = RAW_DIR / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{video_id}.mp4"
            materialize_video(record, out_path)

            reason, duration_s, fps, n_frames = probe_video(out_path)
            if reason is None and not has_audio_stream(out_path):
                reason = "no_audio_stream"
            if reason is not None:
                quarantine_file(out_path, video_id, cls, reason)
                quarantined_ids.add(video_id)
                continue

            writer.writerow([
                video_id, str(out_path), cls, LABEL_MAP[cls],
                f"{duration_s:.3f}", f"{fps:.3f}", n_frames,
            ])
            mf.flush()
            counts[cls] += 1
            done_ids.add(video_id)
            print(f"\r{counts}", end="", flush=True)

    total = sum(counts.values())
    for k in CAPS:
        assert counts[k] <= CAPS[k], f"Cap breached for {k}: {counts[k]} > {CAPS[k]}"
    assert total <= sum(CAPS.values()), f"Cap breached: {total}"
    print(f"\nIngestion complete: {counts} | total={total}")


def validate_manifest() -> list[str]:
    """Run all post-ingestion acceptance checks. Returns issue strings; empty == pass."""
    issues: list[str] = []

    if not MANIFEST.exists():
        return [f"manifest missing: {MANIFEST}"]

    with MANIFEST.open(newline="") as f:
        rows = list(csv.DictReader(f))

    expected_total = sum(CAPS.values())
    if len(rows) != expected_total:
        issues.append(f"row count {len(rows)} != {expected_total}")

    per_class = {k: 0 for k in CAPS}
    seen_ids: set[str] = set()
    for row in rows:
        cls = row["source_folder"]
        if cls in per_class:
            per_class[cls] += 1
        vid = row["video_id"]
        if vid in seen_ids:
            issues.append(f"duplicate video_id: {vid}")
        seen_ids.add(vid)
        if not Path(row["relative_path"]).exists():
            issues.append(f"missing file for {vid}: {row['relative_path']}")
        if int(row["binary_label"]) != LABEL_MAP.get(cls, -1):
            issues.append(f"label mismatch for {vid}: {row['binary_label']} vs {LABEL_MAP.get(cls)}")
        try:
            d = float(row["duration_s"]); fps = float(row["fps"]); n = int(row["n_frames"])
        except ValueError:
            issues.append(f"non-numeric probe field for {vid}")
            continue
        if not (d > 0 and fps > 0 and n > 0):
            issues.append(f"non-positive probe field for {vid}: dur={d} fps={fps} n={n}")

    for cls, cap in CAPS.items():
        if per_class[cls] < cap:
            issues.append(f"{cls} under cap: {per_class[cls]} < {cap}")
        if per_class[cls] > cap:
            issues.append(f"{cls} OVER cap: {per_class[cls]} > {cap}")

    if QUARANTINE_LOG.exists():
        with QUARANTINE_LOG.open(newline="") as f:
            for row in csv.DictReader(f):
                if row["reason"] not in QUARANTINE_REASONS:
                    issues.append(
                        f"invalid quarantine reason for {row['video_id']}: {row['reason']}"
                    )
                qpath = QUARANTINE_DIR / row["source_folder"] / f"{row['video_id']}.mp4"
                if not qpath.exists():
                    issues.append(f"quarantine row without file: {row['video_id']} → {qpath}")

    return issues


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MAVOS-DD subset ingestion.")
    parser.add_argument(
        "--validate", action="store_true",
        help="Run post-ingestion acceptance checks (read-only); no streaming, no writes.",
    )
    args = parser.parse_args(argv)

    if args.validate:
        issues = validate_manifest()
        if not issues:
            print("VALIDATION OK")
            return 0
        for s in issues:
            print(s)
        return 1

    main()
    return 0


if __name__ == "__main__":
    sys.exit(cli())
