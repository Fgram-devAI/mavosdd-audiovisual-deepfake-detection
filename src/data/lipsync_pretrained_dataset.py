"""PyTorch Dataset over cached per-window AV embeddings — computes sync features per pair."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


MAX_OFFSET = 5
SYNC_FEATURE_NAMES: tuple[str, ...] = (
    "mean_cos_sim_zero_offset",
    "median_cos_sim_zero_offset",
    "std_cos_sim_zero_offset",
    "best_offset_mean_cos_sim",
    "best_offset_index_normalized",
    "zero_minus_best_offset",
    "n_windows_used_normalized",
)
SYNC_FEATURE_DIM: int = len(SYNC_FEATURE_NAMES)
_REFERENCE_MAX_WINDOWS = 25


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, 1e-8, None)


def compute_sync_features(
    visual_windows: np.ndarray,
    audio_windows: np.ndarray,
    *,
    max_offset: int = MAX_OFFSET,
) -> np.ndarray:
    """Compute the fixed-length pair-level sync feature vector.

    ``visual_windows`` and ``audio_windows`` are ``(N_v, D)`` and ``(N_a, D)``
    per-window embeddings from the pretrained backend. They are truncated to
    the shared window count and l2-normalized before the offset scan.
    """
    n = min(visual_windows.shape[0], audio_windows.shape[0])
    if n == 0:
        return np.zeros(SYNC_FEATURE_DIM, dtype=np.float32)
    v = _l2_normalize(visual_windows[:n].astype(np.float32))
    a = _l2_normalize(audio_windows[:n].astype(np.float32))

    sims_zero = (v * a).sum(axis=-1)
    mean_zero = float(sims_zero.mean())
    median_zero = float(np.median(sims_zero))
    std_zero = float(sims_zero.std())

    offsets = list(range(-max_offset, max_offset + 1))
    offset_means: list[float] = []
    for k in offsets:
        if k == 0:
            offset_means.append(mean_zero)
            continue
        if abs(k) >= n:
            offset_means.append(float("-inf"))
            continue
        if k > 0:
            v_slice = v[:-k]
            a_slice = a[k:]
        else:
            v_slice = v[-k:]
            a_slice = a[:k]
        offset_means.append(float((v_slice * a_slice).sum(axis=-1).mean()))
    best_idx = int(np.argmax(offset_means))
    best_mean = float(offset_means[best_idx])
    best_offset = offsets[best_idx]
    best_offset_norm = float(best_offset) / float(max_offset)
    zero_minus_best = mean_zero - best_mean

    n_norm = min(1.0, n / _REFERENCE_MAX_WINDOWS)

    return np.array([
        mean_zero,
        median_zero,
        std_zero,
        best_mean,
        best_offset_norm,
        zero_minus_best,
        n_norm,
    ], dtype=np.float32)


class LipSyncPretrainedDataset(Dataset):
    def __init__(
        self,
        *,
        manifest: Path,
        split: str,
        backend: str,
        visual_dir: Path,
        audio_dir: Path,
        failures_csv: Path | None = None,
        max_offset: int = MAX_OFFSET,
    ) -> None:
        if split == "test":
            raise ValueError("test split is locked; refuse to open test rows")
        self.backend = backend
        self.visual_dir = visual_dir
        self.audio_dir = audio_dir
        self.max_offset = max_offset
        self.excluded_pair_ids: set[str] = set()

        failed_sample_ids: set[str] = set()
        if failures_csv is not None and failures_csv.exists():
            with failures_csv.open() as f:
                reader = csv.DictReader(f)
                failed_sample_ids = {r["sample_id"] for r in reader}

        with manifest.open() as f:
            reader = csv.DictReader(f)
            all_rows = [r for r in reader if r["split"] == split]

        kept: list[dict] = []
        for r in all_rows:
            vid = r["source_video_id"]
            aid = r["audio_sample_id"]
            vpath = visual_dir / f"{vid}.npy"
            apath = audio_dir / f"{aid}.npy"
            if (
                not vpath.exists()
                or not apath.exists()
                or vid in failed_sample_ids
                or aid in failed_sample_ids
            ):
                self.excluded_pair_ids.add(r["pair_id"])
                continue
            kept.append(r)
        kept.sort(key=lambda r: r["pair_id"])
        self._rows = kept

    def __len__(self) -> int:
        return len(self._rows)

    def _load_windows(self, path: Path) -> np.ndarray:
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    def __getitem__(self, idx: int) -> dict:
        r = self._rows[idx]
        vid = r["source_video_id"]
        aid = r["audio_sample_id"]
        v_windows = self._load_windows(self.visual_dir / f"{vid}.npy")
        a_windows = self._load_windows(self.audio_dir / f"{aid}.npy")
        sync_features = compute_sync_features(v_windows, a_windows, max_offset=self.max_offset)
        pooled_visual = v_windows.mean(axis=0).astype(np.float32)
        pooled_audio = a_windows.mean(axis=0).astype(np.float32)
        label = torch.tensor(float(r["sync_label_binary"]), dtype=torch.float32)
        return {
            "sync_features": torch.from_numpy(sync_features),
            "pooled_visual": torch.from_numpy(pooled_visual),
            "pooled_audio": torch.from_numpy(pooled_audio),
            "sync_label": label,
            "pair_id": r["pair_id"],
            "negative_type": r["negative_type"],
            "audio_provider": r["audio_provider"],
        }


def _collate(batch: list[dict]) -> dict:
    return {
        "sync_features": torch.stack([b["sync_features"] for b in batch]),
        "pooled_visual": torch.stack([b["pooled_visual"] for b in batch]),
        "pooled_audio": torch.stack([b["pooled_audio"] for b in batch]),
        "sync_label": torch.stack([b["sync_label"] for b in batch]),
        "pair_id": [b["pair_id"] for b in batch],
        "negative_type": [b["negative_type"] for b in batch],
        "audio_provider": [b["audio_provider"] for b in batch],
    }


def make_dataloader(
    dataset: LipSyncPretrainedDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=g,
        collate_fn=_collate,
    )
