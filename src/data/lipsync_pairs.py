"""LipSyncPairDataset and DataLoader helper for the lip-sync consistency branch."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src import common
from src.data.feature_store import (
    FeatureStoreValidationError,
    _load_audio_array,  # reused for shape + dim + finite validation
    _load_lip_array,    # reused for shape + mask validation
)


_CODEC_DIRS = {
    "wav2vec2": common.FEAT_AUDIO_WAV2VEC2_CODEC_DIR,
    "wavlm": common.FEAT_AUDIO_WAVLM_CODEC_DIR,
    "hubert": common.FEAT_AUDIO_HUBERT_CODEC_DIR,
}


def _resolve_codec_audio_dir(backend: str) -> Path:
    if backend not in _CODEC_DIRS:
        raise FeatureStoreValidationError(f"unknown backend: {backend!r}")
    return _CODEC_DIRS[backend]


_METADATA_KEYS = (
    "pair_id", "split", "source_video_id",
    "audio_sample_id", "audio_provider",
    "audio_label", "negative_type", "source_folder", "voice_id_or_name",
)


def _read_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _load_audio(path: Path) -> torch.Tensor:
    # _load_audio_array raises FeatureStoreValidationError on missing/bad shape/dim.
    arr = _load_audio_array(path)
    return torch.from_numpy(arr)


def _load_lips(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    # _load_lip_array raises FeatureStoreValidationError on missing/bad shape/mask/nan.
    feats, mask = _load_lip_array(path)
    return torch.from_numpy(feats), torch.from_numpy(mask)


class LipSyncPairDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        backend: str,
        audio_dir: Path | None = None,
        lips_dir: Path | None = None,
    ) -> None:
        rows = [r for r in _read_rows(Path(manifest_path)) if r.get("split") == split]
        self._rows = rows
        self.split = split
        self.backend = backend
        self._audio_dir = Path(audio_dir) if audio_dir is not None else _resolve_codec_audio_dir(backend)
        self._lips_dir = Path(lips_dir) if lips_dir is not None else common.FEAT_LIPS_DIR

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        audio = _load_audio(self._audio_dir / f"{row['audio_sample_id']}.npy")
        feats, mask = _load_lips(self._lips_dir / f"{row['source_video_id']}.npz")
        try:
            label = torch.tensor(int(row["sync_label_binary"]), dtype=torch.long)
        except (KeyError, ValueError) as exc:
            raise FeatureStoreValidationError(
                f"row {row.get('pair_id')!r} has invalid sync_label_binary={row.get('sync_label_binary')!r}"
            ) from exc
        item: dict = {"audio": audio, "lips": feats, "lips_mask": mask, "label": label}
        for k in _METADATA_KEYS:
            item[k] = row.get(k, "")
        return item


def lipsync_collate(items: list[dict]) -> dict:
    if not items:
        raise FeatureStoreValidationError("lipsync_collate received empty batch")

    def _stack(key: str) -> torch.Tensor:
        base = items[0][key].shape
        for i, it in enumerate(items[1:], start=1):
            if it[key].shape != base:
                raise FeatureStoreValidationError(
                    f"{key} shape mismatch in batch: 0={tuple(base)} {i}={tuple(it[key].shape)}"
                )
        return torch.stack([it[key] for it in items], dim=0)

    return {
        "audio": _stack("audio"),
        "lips": _stack("lips"),
        "lips_mask": _stack("lips_mask"),
        "label": torch.stack([it["label"].reshape(()) for it in items], dim=0),
        "metadata": [{k: it[k] for k in _METADATA_KEYS} for it in items],
    }


def make_lipsync_dataloader(
    dataset: LipSyncPairDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=drop_last,
        collate_fn=lipsync_collate,
    )
