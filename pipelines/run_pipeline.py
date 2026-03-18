"""
  python pipelines/run_pipeline.py \
    --role-arn arn:aws:iam::123:role/mlops-toxic-sagemaker-exec \
    --image-uri 123.dkr.ecr.us-east-1.amazonaws.com/mlops-toxic:latest \
    [--input-s3-uri s3://mlops-toxic-raw/train.csv] \
    [--f1-threshold 0.02] \
    [--wait]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.pipeline import get_pipeline
from src.config import config

TERMINAL_STATES = {"Succeeded", "Failed", "Stopped"}
POLL_INTERVAL = 30  # seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Upsert + start SageMaker Pipeline")
    parser.add_argument("--role-arn", type=str, default=config.sagemaker.role_arn)
    parser.add_argument("--image-uri", type=str, default=config.sagemaker.ecr_image_uri)
    parser.add_argument("--pipeline-bucket", type=str, default=config.aws.pipeline_bucket)
    parser.add_argument(
        "--input-s3-uri",
        type=str,
        default=f"s3://{config.aws.raw_bucket}/train.csv",
    )
    parser.add_argument("--f1-threshold", type=float, default=config.sagemaker.f1_threshold)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--model-package-group", type=str, default="mlops-toxic-models")
    parser.add_argument("--wait", action="store_true", help="Block until execution completes")
    parser.add_argument("--no-start", action="store_true", help="Only upsert, do not start")
    args = parser.parse_args()

    if not args.role_arn:
        print("ERROR: --role-arn is required (or set SAGEMAKER_ROLE_ARN env var)")
        sys.exit(1)
    if not args.image_uri:
        print("ERROR: --image-uri is required (or set ECR_IMAGE_URI env var)")
        sys.exit(1)

    print(f"Upserting pipeline '{config.sagemaker.pipeline_name}'…")
    pipeline = get_pipeline(
        role_arn=args.role_arn,
        image_uri=args.image_uri,
        pipeline_bucket=args.pipeline_bucket,
        region=config.aws.region,
    )
    pipeline.upsert(role_arn=args.role_arn)
    print("Pipeline definition upserted.")

    if args.no_start:
        print("--no-start flag set, skipping execution.")
        return

    import datetime

    run_name = args.run_name or f"run-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    execution = pipeline.start(
        parameters={
            "InputS3Uri": args.input_s3_uri,
            "F1Threshold": str(args.f1_threshold),
            "RunName": run_name,
            "ModelPackageGroup": args.model_package_group,
        },
        execution_display_name=run_name,
    )

    print(f"Started execution: {execution.arn}")
    print(f"Run name: {run_name}")

    if not args.wait:
        print("Not waiting for completion (use --wait to block).")
        return

    import botocore.exceptions

    print(f"Polling every {POLL_INTERVAL}s…")

    resp = None
    while True:
        try:
            resp = execution.describe()
        except botocore.exceptions.ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("ResourceNotFound", "ValidationException"):
                print("  Execution not yet visible, retrying…")
                time.sleep(POLL_INTERVAL)
                continue
            raise
        status = resp["PipelineExecutionStatus"]
        print(f"  Status: {status}")

        if status in TERMINAL_STATES:
            break

        time.sleep(POLL_INTERVAL)

    if status == "Succeeded":
        print(f"\nPipeline execution SUCCEEDED: {execution.arn}")
    else:
        failure_reason = resp.get("FailureReason", "unknown") if resp else "unknown"
        print(f"\nPipeline execution {status}: {failure_reason}")

        try:
            steps_resp = execution.list_steps()
            steps = steps_resp if isinstance(steps_resp, list) else steps_resp.get("PipelineExecutionSteps", [])
            failed = [s for s in steps if s.get("StepStatus") == "Failed"]

            for s in failed:
                print(f"\n  Failed step: {s.get('StepName')}")
                print(f"    Failure reason: {s.get('FailureReason', 'n/a')}")
                metadata = s.get("Metadata", {})
                for key, val in metadata.items():
                    print(f"    {key}: {val}")

        except Exception as list_exc:
            print(f"  (Could not retrieve step details: {list_exc})")

        sys.exit(1)


if __name__ == "__main__":
    main()
