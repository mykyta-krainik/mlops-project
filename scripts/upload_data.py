"""
One-time bootstrap: upload local training data to S3.

Usage:
  python scripts/upload_data.py \
    [--local-path data/train.csv] \
    [--s3-key train.csv]

Environment:
  AWS_REGION, AWS_RAW_BUCKET (or defaults from src/config.py)
"""

import argparse
import sys
from pathlib import Path

import boto3
import botocore

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload local data CSV to S3 raw bucket")
    parser.add_argument(
        "--local-path",
        type=str,
        default="data/train.csv",
        help="Local path to the training CSV",
    )
    parser.add_argument(
        "--s3-key",
        type=str,
        default="train.csv",
        help="S3 object key within the raw bucket",
    )
    parser.add_argument("--bucket", type=str, default=config.aws.raw_bucket)
    args = parser.parse_args()

    local_path = Path(args.local_path)
    if not local_path.exists():
        print(f"ERROR: {local_path} does not exist")
        sys.exit(1)

    s3 = boto3.client("s3", region_name=config.aws.region)
    bucket = args.bucket

    # Verify bucket exists
    try:
        s3.head_bucket(Bucket=bucket)
    except botocore.exceptions.ClientError as e:
        print(f"ERROR: bucket '{bucket}' not accessible: {e}")
        sys.exit(1)

    file_size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"Uploading {local_path} ({file_size_mb:.1f} MB) → s3://{bucket}/{args.s3_key}")

    s3.upload_file(str(local_path), bucket, args.s3_key)
    print("Upload complete.")

    # Verify
    head = s3.head_object(Bucket=bucket, Key=args.s3_key)
    print(f"Confirmed: s3://{bucket}/{args.s3_key} ({head['ContentLength']} bytes)")


if __name__ == "__main__":
    main()
