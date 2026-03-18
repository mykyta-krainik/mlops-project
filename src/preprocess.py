import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.preprocessing import TextPreprocessor


def run_preprocess(raw_s3_uri: str, pipeline_bucket: str, run_prefix: str) -> dict:
    import tempfile
    import boto3

    s3 = boto3.client("s3")
    parts = raw_s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        local_csv = tmp / "train.csv"
        s3.download_file(bucket, key, str(local_csv))

        df = pd.concat([pd.read_csv(local_csv)], ignore_index=True)
        print(f"Loaded {len(df)} rows")

        before = len(df)
        df = df.dropna(subset=["comment_text"])
        df["comment_text"] = df["comment_text"].astype(str)
        if len(df) < before:
            print(f"Dropped {before - len(df)} rows with NaN comment_text")

        preprocessor = TextPreprocessor()
        df = preprocessor.preprocess_dataframe(df)
        print(f"After preprocessing: {len(df)} rows")

        ref_path = tmp / "reference.parquet"
        df.to_parquet(ref_path, index=False)
        ref_key = f"{run_prefix}/reference/reference.parquet"
        s3.upload_file(str(ref_path), pipeline_bucket, ref_key)
        s3.upload_file(str(ref_path), config.aws.processed_bucket, "reference/reference.parquet")

        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            df,
            test_size=config.model.test_size,
            random_state=config.model.random_state,
        )
        print(f"Train: {len(train_df)}, Validation: {len(val_df)}")

        train_path = tmp / "train.csv"
        val_path = tmp / "validation.csv"
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        train_key = f"{run_prefix}/train/train.csv"
        val_key = f"{run_prefix}/validation/validation.csv"
        s3.upload_file(str(train_path), pipeline_bucket, train_key)
        s3.upload_file(str(val_path), pipeline_bucket, val_key)
        print(f"Uploaded train → s3://{pipeline_bucket}/{train_key}")
        print(f"Uploaded validation → s3://{pipeline_bucket}/{val_key}")

    return {
        "train_uri": f"s3://{pipeline_bucket}/{train_key}",
        "val_uri": f"s3://{pipeline_bucket}/{val_key}",
        "reference_uri": f"s3://{pipeline_bucket}/{ref_key}",
    }


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

    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df)
    print(f"After preprocessing: {len(df)} rows")

    ref_dir = output_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ref_dir / "reference.parquet", index=False)
    print(f"Saved reference dataset: {ref_dir / 'reference.parquet'}")

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
