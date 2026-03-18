import argparse
import json
import sys
from pathlib import Path

import boto3
import botocore

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config


def get_prod_f1(model_package_group: str) -> float:
    sm = boto3.client("sagemaker", region_name=config.aws.region)
    try:
        paginator = sm.get_paginator("list_model_packages")
        for page in paginator.paginate(
            ModelPackageGroupName=model_package_group,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
        ):
            packages = page.get("ModelPackageSummaryList", [])
            if packages:
                pkg_arn = packages[0]["ModelPackageArn"]
                detail = sm.describe_model_package(ModelPackageName=pkg_arn)
                metrics = detail.get("ModelMetrics", {})
                model_quality = metrics.get("ModelQuality", {})
                statistics = model_quality.get("Statistics", {})
                s3_uri = statistics.get("S3Uri", "")
                if s3_uri:
                    return _read_f1_from_s3(s3_uri)
    except botocore.exceptions.ClientError as e:
        print(f"Could not query model registry: {e} — treating as first run (prod_f1=0.0)")

    return 0.0


def _read_f1_from_s3(s3_uri: str) -> float:
    s3 = boto3.client("s3")
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj["Body"].read())
        return float(data.get("f1_macro", 0.0))
    except Exception as e:
        print(f"Could not read metrics from {s3_uri}: {e}")
        return 0.0


def run_evaluate(baseline: dict, improved: dict, model_package_group: str) -> dict:
    baseline_f1 = float(baseline["f1_macro"])
    improved_f1 = float(improved["f1_macro"])

    print(f"Baseline f1_macro:  {baseline_f1:.4f}")
    print(f"Improved f1_macro:  {improved_f1:.4f}")

    if improved_f1 >= baseline_f1:
        best_model = "improved"
        best_f1 = improved_f1
        best_model_s3_uri = improved["model_s3_uri"]
    else:
        best_model = "baseline"
        best_f1 = baseline_f1
        best_model_s3_uri = baseline["model_s3_uri"]

    print(f"Best candidate:     {best_model} ({best_f1:.4f})")

    prod_f1 = get_prod_f1(model_package_group)
    print(f"Current prod f1:    {prod_f1:.4f}")
    print(f"Threshold:          {config.sagemaker.f1_threshold}")

    return {
        "baseline_f1_macro": baseline_f1,
        "improved_f1_macro": improved_f1,
        "best_model": best_model,
        "best_f1_macro": best_f1,
        "best_model_s3_uri": best_model_s3_uri,
        "prod_f1_macro": prod_f1,
        "f1_threshold": config.sagemaker.f1_threshold,
        "exceeds_threshold": best_f1 >= prod_f1 + config.sagemaker.f1_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    parser.add_argument(
        "--model-package-group",
        type=str,
        default=f"{config.sagemaker.pipeline_name.replace('-pipeline', '')}-models",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics_path = input_dir / "baseline" / "metrics.json"
    improved_metrics_path = input_dir / "improved" / "metrics.json"

    with open(baseline_metrics_path) as f:
        baseline = json.load(f)
    with open(improved_metrics_path) as f:
        improved = json.load(f)

    baseline_f1 = float(baseline["f1_macro"])
    improved_f1 = float(improved["f1_macro"])

    print(f"Baseline f1_macro:  {baseline_f1:.4f}")
    print(f"Improved f1_macro:  {improved_f1:.4f}")

    if improved_f1 >= baseline_f1:
        best_model = "improved"
        best_f1 = improved_f1
    else:
        best_model = "baseline"
        best_f1 = baseline_f1

    print(f"Best candidate:     {best_model} ({best_f1:.4f})")

    prod_f1 = get_prod_f1(args.model_package_group)
    print(f"Current prod f1:    {prod_f1:.4f}")
    print(f"Threshold:          {config.sagemaker.f1_threshold}")

    evaluation = {
        "baseline_f1_macro": baseline_f1,
        "improved_f1_macro": improved_f1,
        "best_model": best_model,
        "best_f1_macro": best_f1,
        "prod_f1_macro": prod_f1,
        "f1_threshold": config.sagemaker.f1_threshold,
        "exceeds_threshold": best_f1 >= prod_f1 + config.sagemaker.f1_threshold,
    }

    output_path = output_dir / "evaluation.json"
    with open(output_path, "w") as f:
        json.dump(evaluation, f, indent=2)

    print(f"\nWrote evaluation report to {output_path}")
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
