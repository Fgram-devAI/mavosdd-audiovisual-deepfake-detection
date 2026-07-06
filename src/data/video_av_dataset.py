"""Dataset for video-level real-vs-fake classification over cached AV embeddings."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.lipsync_pretrained_dataset import (
    MAX_OFFSET,
    SYNC_FEATURE_DIM,
    compute_sync_features,
)


def fixed_window(
    windows: np.ndarray,
    *,
    window_count: int | None,
    policy: str = "center",
) -> np.ndarray:
    """Crop or right-pad a 2D window sequence to a fixed temporal length."""
    if window_count is None:
        return windows
    if window_count <= 0:
        raise ValueError("window_count must be positive")
    if windows.ndim != 2:
        raise ValueError(f"expected 2D window array, got {windows.shape}")
    n, dim = windows.shape
    if n == window_count:
        return windows
    if n > window_count:
        if policy == "first":
            start = 0
        elif policy == "center":
            start = (n - window_count) // 2
        else:
            raise ValueError(f"unknown window policy: {policy!r}")
        return windows[start:start + window_count]
    pad = np.zeros((window_count - n, dim), dtype=windows.dtype)
    return np.concatenate([windows, pad], axis=0)


class VideoAVDataset(Dataset):
    def __init__(
        self,
        *,
        manifest: Path,
        split: str,
        visual_dir: Path,
        audio_dir: Path,
        failures_csv: Path | None = None,
        max_offset: int = MAX_OFFSET,
        window_count: int | None = None,
        window_policy: str = "center",
    ) -> None:
        if split == "test":
            raise ValueError("test split is locked; refuse to open test rows")
        self.visual_dir = visual_dir
        self.audio_dir = audio_dir
        self.max_offset = max_offset
        self.window_count = window_count
        self.window_policy = window_policy
        self.excluded_sample_ids: set[str] = set()

        failed_sample_ids: set[str] = set()
        if failures_csv is not None and failures_csv.exists():
            with failures_csv.open(newline="") as f:
                reader = csv.DictReader(f)
                failed_sample_ids = {r["sample_id"] for r in reader}

        with manifest.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r["split"] == split]

        kept: list[dict] = []
        for row in rows:
            vid = row["source_video_id"]
            aid = row["audio_sample_id"]
            vpath = visual_dir / f"{vid}.npy"
            apath = audio_dir / f"{aid}.npy"
            if (
                not vpath.exists()
                or not apath.exists()
                or vid in failed_sample_ids
                or aid in failed_sample_ids
            ):
                self.excluded_sample_ids.add(row["sample_id"])
                continue
            kept.append(row)
        kept.sort(key=lambda r: r["sample_id"])
        self._rows = kept

    def __len__(self) -> int:
        return len(self._rows)

    def _load_windows(self, path: Path) -> np.ndarray:
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"expected 2D embedding array at {path}, got {arr.shape}")
        return arr

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        vid = row["source_video_id"]
        aid = row["audio_sample_id"]
        v_windows = self._load_windows(self.visual_dir / f"{vid}.npy")
        a_windows = self._load_windows(self.audio_dir / f"{aid}.npy")
        v_windows = fixed_window(
            v_windows, window_count=self.window_count, policy=self.window_policy,
        )
        a_windows = fixed_window(
            a_windows, window_count=self.window_count, policy=self.window_policy,
        )
        sync_features = compute_sync_features(v_windows, a_windows, max_offset=self.max_offset)
        label = torch.tensor(float(row["video_label_binary"]), dtype=torch.float32)
        return {
            "sync_features": torch.from_numpy(sync_features),
            "pooled_visual": torch.from_numpy(v_windows.mean(axis=0).astype(np.float32)),
            "pooled_audio": torch.from_numpy(a_windows.mean(axis=0).astype(np.float32)),
            "label": label,
            "sample_id": row["sample_id"],
            "source_folder": row["source_folder"],
            "video_label": row["video_label"],
        }


def _collate(batch: list[dict]) -> dict:
    return {
        "sync_features": torch.stack([b["sync_features"] for b in batch]),
        "pooled_visual": torch.stack([b["pooled_visual"] for b in batch]),
        "pooled_audio": torch.stack([b["pooled_audio"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "sample_id": [b["sample_id"] for b in batch],
        "source_folder": [b["source_folder"] for b in batch],
        "video_label": [b["video_label"] for b in batch],
    }


def make_dataloader(
    dataset: VideoAVDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        collate_fn=_collate,
    )
