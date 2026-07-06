"""CLI: extract pretrained SyncNet embeddings for the lipsync pair manifest."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from tqdm import tqdm

from src import common
from src.features.mouth_crop_extract import (
    MouthDetectionError,
    SYNCNET_SPEC,
    extract_mouth_crops,
)
from src.features.syncnet_backend import SyncNetBackend


def iter_pair_manifest_rows(
    manifest: Path,
    *,
    splits: Sequence[str],
    limit: int | None,
) -> Iterable[dict]:
    if "test" in splits:
        raise ValueError("test split is locked; refuse to open test rows")
    with manifest.open() as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["split"] in set(splits)]
    rows.sort(key=lambda r: r["pair_id"])
    if limit is not None:
        rows = rows[:limit]
    return rows


def unique_visual_and_audio_units(
    rows: Iterable[dict],
    *,
    raw_video_root: Path,
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    visuals: dict[str, Path] = {}
    audios: dict[str, Path] = {}
    for r in rows:
        vid = r["source_video_id"]
        if vid not in visuals:
            visuals[vid] = raw_video_root / r["source_folder"] / f"{vid}.mp4"
        aid = r["audio_sample_id"]
        if aid not in audios:
            audios[aid] = Path(r["audio_path"])
    return list(visuals.items()), list(audios.items())


def log_failure(csv_path: Path, *, sample_id: str, stage: str, error: BaseException) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=common.EXTRACTION_FAILURE_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({
            "sample_id": sample_id,
            "stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        })


def _compute_mel(audio_path: Path) -> np.ndarray:
    """Compute log-mel spectrogram windows for the SyncNet audio branch.

    Returns (N_windows, 1, 80, 16) float32. Mono, 16 kHz.
    """
    import librosa

    y, _ = librosa.load(str(audio_path), sr=common.SR, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=common.SR, n_mels=80, n_fft=800, hop_length=200)
    log_mel = np.log10(mel + 1e-6)
    windows = []
    step = 16
    for start in range(0, log_mel.shape[1] - step + 1, step):
        windows.append(log_mel[:, start : start + step])
    if not windows:
        raise RuntimeError(f"audio too short for mel windowing: {audio_path}")
    return np.stack(windows)[:, None, :, :].astype(np.float32)


def _extract_visual_unit(
    video_id: str,
    video_path: Path,
    out_dir: Path,
    adapter: SyncNetBackend,
    failures_csv: Path,
    *,
    overwrite: bool,
) -> None:
    out_path = out_dir / f"{video_id}.npy"
    if out_path.exists() and not overwrite:
        return
    try:
        stacks = extract_mouth_crops(video_path, SYNCNET_SPEC)
        emb = adapter.encode_visual(stacks.astype(np.float32))
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_path, emb.astype(np.float16))
    except MouthDetectionError as e:
        log_failure(failures_csv, sample_id=video_id, stage=e.stage, error=e)
    except Exception as e:
        log_failure(failures_csv, sample_id=video_id, stage="encoder_forward", error=e)


def _extract_audio_unit(
    audio_sample_id: str,
    audio_path: Path,
    out_dir: Path,
    adapter: SyncNetBackend,
    failures_csv: Path,
    *,
    overwrite: bool,
) -> None:
    out_path = out_dir / f"{audio_sample_id}.npy"
    if out_path.exists() and not overwrite:
        return
    try:
        mel = _compute_mel(audio_path)
        emb = adapter.encode_audio(mel)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_path, emb.astype(np.float16))
    except Exception as e:
        stage = "audio_decode" if "load" in str(e).lower() else "encoder_forward"
        log_failure(failures_csv, sample_id=audio_sample_id, stage=stage, error=e)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=common.LIPSYNC_PAIRS_MANIFEST)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--raw-video-root", type=Path, default=common.RAW_DIR)
    parser.add_argument("--out-visual-dir", type=Path, default=common.FEAT_SYNCNET_VISUAL_DIR)
    parser.add_argument("--out-audio-dir", type=Path, default=common.FEAT_SYNCNET_AUDIO_DIR)
    parser.add_argument("--checkpoint", type=Path, default=common.SYNCNET_CKPT_PATH)
    parser.add_argument("--failures-csv", type=Path, default=common.SYNCNET_FAILURES_CSV)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if "test" in args.splits:
        print("[extract_syncnet] refusing test split", file=sys.stderr)
        return 2

    adapter = SyncNetBackend.from_checkpoint(args.checkpoint)
    rows = list(iter_pair_manifest_rows(
        args.manifest, splits=tuple(args.splits), limit=args.limit,
    ))
    visuals, audios = unique_visual_and_audio_units(rows, raw_video_root=args.raw_video_root)
    for vid, path in tqdm(visuals, desc="syncnet visual", unit="video"):
        _extract_visual_unit(vid, path, args.out_visual_dir, adapter, args.failures_csv,
                             overwrite=args.overwrite)
    for aid, path in tqdm(audios, desc="syncnet audio", unit="clip"):
        _extract_audio_unit(aid, path, args.out_audio_dir, adapter, args.failures_csv,
                            overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
