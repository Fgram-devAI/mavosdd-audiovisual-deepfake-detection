"""Score-vector Dataset over the final-fusion score CSV.

Reads a score table produced by ``build_final_fusion_scores`` and exposes numpy
arrays for scikit-learn or PyTorch consumption. Refuses ``split == "test"`` and
refuses ``source_folder`` (or any manifest metadata) as an input feature.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


DEFAULT_FEATURE_COLUMNS: tuple[str, ...] = (
    "audio_fake_score",
    "video_av_fake_score",
    "sync_inconsistent_score",
)

_FORBIDDEN_FEATURE_COLUMNS: frozenset[str] = frozenset({
    "source_folder", "source_video_id", "sample_id", "split",
    "audio_backend", "video_av_backend", "sync_backend",
    "final_label_binary", "missing_features",
})


class FinalFusionDataset:
    def __init__(
        self,
        *,
        score_csv: Path,
        split: str,
        feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    ) -> None:
        if split == "test":
            raise ValueError("test split is locked; refuse to open final-fusion test rows")
        bad = [c for c in feature_columns if c in _FORBIDDEN_FEATURE_COLUMNS]
        if bad:
            raise ValueError(
                f"forbidden feature column(s): {bad}. "
                "source_folder / metadata columns are reporting-only, not model input."
            )

        with Path(score_csv).open(newline="") as f:
            reader = csv.DictReader(f)
            all_rows = [r for r in reader if r["split"] == split]

        xs: list[list[float]] = []
        ys: list[int] = []
        sample_ids: list[str] = []
        sources: list[str] = []
        excluded = 0
        for row in all_rows:
            values: list[float] = []
            skip = False
            for col in feature_columns:
                cell = row.get(col, "")
                if cell == "":
                    skip = True
                    break
                values.append(float(cell))
            if skip:
                excluded += 1
                continue
            xs.append(values)
            ys.append(int(row["final_label_binary"]))
            sample_ids.append(row["sample_id"])
            sources.append(row["source_folder"])

        self.feature_columns = tuple(feature_columns)
        self.X = np.asarray(xs, dtype=np.float32).reshape(len(xs), len(feature_columns))
        self.y = np.asarray(ys, dtype=np.int64)
        self.sample_ids = sample_ids
        self.source_folders = sources
        self.excluded_missing = excluded

    def __len__(self) -> int:
        return int(self.X.shape[0])
