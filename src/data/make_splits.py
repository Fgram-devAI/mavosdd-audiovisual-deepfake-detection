"""Generate frozen stratified train/val/test split CSV files from data/manifest.csv."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import MANIFEST, SEED, SPLITS_DIR


def main() -> None:
    df = pd.read_csv(MANIFEST)
    stratify_col = df["source_folder"]
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=stratify_col,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["source_folder"],
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")


if __name__ == "__main__":
    main()
