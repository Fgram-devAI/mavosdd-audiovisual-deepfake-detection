"""PyTorch dataset over pre-extracted audio .npy and lip .npz feature stores."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.common import FEAT_AUDIO_DIR, FEAT_LIPS_DIR


class MultimodalFeatureDataset(Dataset):
    def __init__(
        self,
        split_csv: str | Path,
        audio_dir: str | Path = FEAT_AUDIO_DIR,
        lips_dir: str | Path = FEAT_LIPS_DIR,
    ):
        self.df = pd.read_csv(split_csv)
        self.audio_dir = Path(audio_dir)
        self.lips_dir = Path(lips_dir)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video_id = row["video_id"]
        audio = np.load(self.audio_dir / f"{video_id}.npy", mmap_mode="r")
        lips = np.load(self.lips_dir / f"{video_id}.npz")
        label = np.float32(row["binary_label"])
        return (
            torch.from_numpy(np.asarray(audio, dtype=np.float32)),
            torch.from_numpy(lips["feats"].astype(np.float32)),
            torch.from_numpy(lips["mask"].astype(np.float32)),
            torch.tensor(label, dtype=torch.float32),
        )
