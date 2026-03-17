"""
SageMaker ProcessingStep — ingest.

The raw CSV is mounted by SageMaker from S3 at --input-dir.
This step validates the schema and passes the file through to --output-dir.
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

REQUIRED_COLUMNS = {
    "id", "comment_text",
    "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate",
}


def run_ingest(input_s3_uri: str) -> str:
    """Download CSV from S3, validate schema, return the same URI (validation-only step)."""
    import tempfile
    import boto3

    s3 = boto3.client("s3")
    parts = input_s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / "train.csv"
        s3.download_file(bucket, key, str(local_path))
        df = pd.read_csv(local_path)
        validate(df, input_s3_uri)
        print(f"Schema validated for {input_s3_uri}")

    return input_s3_uri


def validate(df: pd.DataFrame, path: str) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    print(f"Validated {len(df)} rows, {len(df.columns)} columns from {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input/raw")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output/raw")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        # SageMaker may mount a single file rather than a directory
        csv_files = list(input_dir.parent.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {input_dir}")

    for src in csv_files:
        df = pd.read_csv(src)
        validate(df, str(src))
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        print(f"Copied {src} → {dst}")


if __name__ == "__main__":
    main()
