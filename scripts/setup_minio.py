import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from minio import Minio
from minio.error import S3Error

from src.config import config


def wait_for_minio(client: Minio, max_retries: int = 30, retry_interval: int = 2) -> bool:
    for attempt in range(max_retries):
        try:
            client.list_buckets()
            return True
        except Exception as e:
            print(f"Waiting for Minio... (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_interval)
    return False


def setup_buckets(client: Minio) -> None:
    buckets = [
        config.minio.raw_data_bucket,
        config.minio.processed_data_bucket,
        config.minio.models_bucket,
        config.minio.mlflow_artifacts_bucket,
    ]

    for bucket in buckets:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            print(f"Created bucket: {bucket}")
        else:
            print(f"Bucket already exists: {bucket}")


def main():
    parser = argparse.ArgumentParser(description="Initialize Minio buckets")
    parser.add_argument(
        "--endpoint",
        default=config.minio.endpoint,
        help="Minio endpoint",
    )
    parser.add_argument(
        "--access-key",
        default=config.minio.access_key,
        help="Minio access key",
    )
    parser.add_argument(
        "--secret-key",
        default=config.minio.secret_key,
        help="Minio secret key",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for Minio to be ready",
    )
    args = parser.parse_args()

    client = Minio(
        endpoint=args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        secure=config.minio.secure,
    )

    if args.wait:
        print("Waiting for Minio to be ready...")
        if not wait_for_minio(client):
            print("ERROR: Minio is not available")
            sys.exit(1)
        print("Minio is ready!")

    setup_buckets(client)
    print("Minio setup complete!")


if __name__ == "__main__":
    main()

