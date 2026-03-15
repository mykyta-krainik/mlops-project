"""
SageMaker ProcessingStep — feature preprocessing.

Applies TextPreprocessor, splits into train/validation, and saves a reference
dataset (pre-split) for Evidently drift detection.

Input:  /opt/ml/processing/input/raw/*.csv
Output:
  /opt/ml/processing/output/train/train.csv
  /opt/ml/processing/output/validation/validation.csv
  /opt/ml/processing/output/reference/reference.parquet   ← Evidently baseline
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.preprocessing import TextPreprocessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input/raw")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--test-size", type=float, default=config.model.test_size)
    parser.add_argument("--random-state", type=int, default=config.model.random_state)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        csv_files = list(input_dir.parent.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {input_dir}")

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(csv_files)} file(s)")

    # ── Preprocess ────────────────────────────────────────────────────────────
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df)
    print(f"After preprocessing: {len(df)} rows")

    # ── Save reference dataset for Evidently (pre-split) ─────────────────────
    ref_dir = output_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ref_dir / "reference.parquet", index=False)
    print(f"Saved reference dataset: {ref_dir / 'reference.parquet'}")

    # ── Train/validation split ────────────────────────────────────────────────
    train_df, val_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}")

    train_dir = output_dir / "train"
    val_dir = output_dir / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_dir / "train.csv", index=False)
    val_df.to_csv(val_dir / "validation.csv", index=False)
    print(f"Saved train → {train_dir / 'train.csv'}")
    print(f"Saved validation → {val_dir / 'validation.csv'}")


if __name__ == "__main__":
    main()
