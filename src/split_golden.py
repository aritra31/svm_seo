# split_golden.py — create train/test split from golden dataset
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config import GOLDEN_XLSX, DATA_DIR

def split_golden(test_size=0.3, seed=42):
    """
    Randomly splits the golden dataset into train/test parts.
    Saves train/test Excel files and a merged CSV with 'split' column.
    """
    gold_path = Path(GOLDEN_XLSX)
    df = pd.read_excel(gold_path)
    df = df.fillna("")

    # random but reproducible split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)

    # mark and combine
    train_df["split"] = "train"
    test_df["split"] = "test"
    merged = pd.concat([train_df, test_df], ignore_index=True)

    out_dir = Path(DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_path = out_dir / "golden_split.xlsx"
    merged.to_excel(merged_path, index=False)

    print(f"✅ Split complete — {len(train_df)} train / {len(test_df)} test rows")
    print(f"Saved merged file with 'split' column to: {merged_path}")

if __name__ == "__main__":
    split_golden(test_size=0.3)
