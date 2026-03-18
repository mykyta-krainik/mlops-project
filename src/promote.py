"""
SageMaker ProcessingStep entry point — champion/challenger gate and deployment.

Runs only when the ConditionStep decides the new model beats prod by the threshold.

Responsibilities:
  1. Register best model to MLflow (Databricks) registry at stage "Staging"
  2. Register to SageMaker Model Registry
  3. Deploy to STAGING endpoint with canary split (80% blue / 20% green)

When called with --to-prod flag (from deploy.yml after load test passes):
  - Promotes green variant to 100% on the PRODUCTION endpoint

When called with --simulate-failure (dev only):
  - Updates staging to a bad config to test SageMaker auto-rollback

Input layout:
  /opt/ml/processing/input/evaluation/evaluation.json
  /opt/ml/processing/input/model/model.onnx   (the best model artifact)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import boto3
import mlflow
import mlflow.pyfunc

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config


def setup_mlflow() -> bool:
    """Configure MLflow. Returns True if setup succeeded."""
    try:
        if config.mlflow.tracking_uri == "databricks":
            import os
            os.environ["DATABRICKS_HOST"] = config.mlflow.databricks_host
            os.environ["DATABRICKS_TOKEN"] = config.mlflow.databricks_token
            mlflow.set_tracking_uri("databricks")
            # Force legacy Workspace Model Registry (not Unity Catalog)
            mlflow.set_registry_uri("databricks")
        else:
            mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        return True
    except Exception as e:
        print(f"MLflow setup failed (registration will be skipped): {e}")
        return False


def register_to_mlflow(model_s3_uri: str, metrics: dict, run_name: str) -> str:
    """Log model artifact from S3 to MLflow and register to Staging."""
    if not setup_mlflow():
        return ""

    try:
        with mlflow.start_run(run_name=f"register_{run_name}"):
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
            mlflow.log_param("model_s3_uri", model_s3_uri)

            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            registered = mlflow.register_model(model_uri, name="toxic-comment-classifier")
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name="toxic-comment-classifier",
                version=registered.version,
                stage="Staging",
            )
            print(f"Registered to MLflow: version={registered.version}, stage=Staging")
            return registered.version
    except Exception as e:
        print(f"MLflow registration failed (non-fatal): {e}")
        return ""


def register_to_sagemaker(
    model_s3_uri: str,
    ecr_image_uri: str,
    metrics: dict,
    model_package_group: str,
    is_first_run: bool,
) -> str:
    """Create a SageMaker Model Package and return its ARN."""
    sm = boto3.client("sagemaker", region_name=config.aws.region)

    approval_status = "Approved" if is_first_run else "PendingManualApproval"

    # Upload metrics to S3 so evaluate.py can fetch them for future runs
    s3 = boto3.client("s3")
    metrics_key = f"model-registry/{datetime.now().strftime('%Y%m%d_%H%M%S')}/metrics.json"
    s3.put_object(
        Bucket=config.aws.pipeline_bucket,
        Key=metrics_key,
        Body=json.dumps(metrics),
        ContentType="application/json",
    )
    metrics_s3_uri = f"s3://{config.aws.pipeline_bucket}/{metrics_key}"

    response = sm.create_model_package(
        ModelPackageGroupName=model_package_group,
        ModelPackageDescription=f"f1_macro={metrics.get('f1_macro', 0):.4f}",
        InferenceSpecification={
            "Containers": [
                {
                    "Image": ecr_image_uri,
                    "ModelDataUrl": model_s3_uri,
                }
            ],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": metrics_s3_uri,
                }
            }
        },
        ModelApprovalStatus=approval_status,
    )

    pkg_arn = response["ModelPackageArn"]
    print(f"Registered to SageMaker Model Registry: {pkg_arn} ({approval_status})")
    return pkg_arn


def deploy_canary_to_staging(model_s3_uri: str, ecr_image_uri: str, run_name: str) -> None:
    """
    Update staging endpoint with a canary split:
      blue (existing model): 80% traffic
      green (new model):     20% traffic
    SageMaker manages the rollout internally — no downtime.
    """
    sm = boto3.client("sagemaker", region_name=config.aws.region)
    endpoint_name = config.sagemaker.staging_endpoint
    ts = datetime.now().strftime("%Y%m%d%H%M%S")

    # Describe existing endpoint to get current model for blue variant
    import botocore.exceptions
    endpoint_exists = False
    endpoint_failed = False
    current_model_name = None
    try:
        existing = sm.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
        if existing["EndpointStatus"] == "Failed":
            endpoint_failed = True
            print(f"Endpoint '{endpoint_name}' is in Failed state — will delete and recreate")
        else:
            current_config_name = existing["EndpointConfigName"]
            current_config = sm.describe_endpoint_config(EndpointConfigName=current_config_name)
            current_model_name = current_config["ProductionVariants"][0]["ModelName"]
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] not in ("ValidationException", "ResourceNotFoundException"):
            raise

    # Delete failed endpoint so we can create fresh
    if endpoint_failed:
        sm.delete_endpoint(EndpointName=endpoint_name)
        print(f"Deleted failed endpoint '{endpoint_name}', waiting for deletion...")
        import time as _time
        for _ in range(60):
            try:
                sm.describe_endpoint(EndpointName=endpoint_name)
                _time.sleep(10)
            except botocore.exceptions.ClientError as e:
                if e.response["Error"]["Code"] in ("ValidationException", "ResourceNotFoundException"):
                    break
                raise
        endpoint_exists = False

    # Create new model resource
    new_model_name = f"{config.project if hasattr(config, 'project') else 'mlops-toxic'}-{run_name}"
    sm.create_model(
        ModelName=new_model_name,
        ExecutionRoleArn=config.sagemaker.role_arn,
        PrimaryContainer={
            "Image": ecr_image_uri,
            "ModelDataUrl": model_s3_uri,
        },
    )

    if current_model_name:
        variants = [
            {
                "VariantName": "blue",
                "ModelName": current_model_name,
                "InstanceType": config.sagemaker.instance_type,
                "InitialInstanceCount": 1,
                "InitialVariantWeight": 0.8,
            },
            {
                "VariantName": "green",
                "ModelName": new_model_name,
                "InstanceType": config.sagemaker.instance_type,
                "InitialInstanceCount": 1,
                "InitialVariantWeight": 0.2,
            },
        ]
    else:
        variants = [
            {
                "VariantName": "green",
                "ModelName": new_model_name,
                "InstanceType": config.sagemaker.instance_type,
                "InitialInstanceCount": 1,
                "InitialVariantWeight": 1.0,
            }
        ]

    config_name = f"mlops-toxic-staging-canary-{ts}"
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=variants,
    )

    if endpoint_exists:
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        print(f"Updated staging endpoint '{endpoint_name}' with canary config '{config_name}'")
    else:
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        print(f"Created staging endpoint '{endpoint_name}' with config '{config_name}'")

    _wait_for_endpoint(sm, endpoint_name)


def promote_to_prod(run_name: str) -> None:
    """
    Shift green variant to 100% on the PRODUCTION endpoint.
    Called from deploy.yml after the Locust load test passes on staging.
    """
    sm = boto3.client("sagemaker", region_name=config.aws.region)
    staging_endpoint = config.sagemaker.staging_endpoint
    prod_endpoint = config.sagemaker.prod_endpoint
    ts = datetime.now().strftime("%Y%m%d%H%M%S")

    # Get the green (new) model from staging
    staging_config_name = sm.describe_endpoint(EndpointName=staging_endpoint)["EndpointConfigName"]
    staging_config = sm.describe_endpoint_config(EndpointConfigName=staging_config_name)
    variants = staging_config["ProductionVariants"]

    green_variant = next((v for v in variants if v["VariantName"] == "green"), variants[0])
    new_model_name = green_variant["ModelName"]

    # Create a prod config with 100% traffic on the new model
    prod_config_name = f"mlops-toxic-prod-{ts}"
    sm.create_endpoint_config(
        EndpointConfigName=prod_config_name,
        ProductionVariants=[
            {
                "VariantName": "blue",
                "ModelName": new_model_name,
                "InstanceType": config.sagemaker.instance_type,
                "InitialInstanceCount": 1,
                "InitialVariantWeight": 1.0,
            }
        ],
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri": f"s3://{config.aws.pipeline_bucket}/data-capture",
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
            "CaptureContentTypeHeader": {"JsonContentTypes": ["application/json"]},
        },
    )

    try:
        sm.describe_endpoint(EndpointName=prod_endpoint)
        sm.update_endpoint(EndpointName=prod_endpoint, EndpointConfigName=prod_config_name)
        print(f"Updated prod endpoint '{prod_endpoint}' with config '{prod_config_name}'")
    except sm.exceptions.ClientError:
        sm.create_endpoint(EndpointName=prod_endpoint, EndpointConfigName=prod_config_name)
        print(f"Created prod endpoint '{prod_endpoint}' with config '{prod_config_name}'")

    _wait_for_endpoint(sm, prod_endpoint)

    staging_variant_names = [v["VariantName"] for v in variants]
    
    if "green" in staging_variant_names:
        sm.update_endpoint_weights_and_capacities(
            EndpointName=staging_endpoint,
            DesiredWeightsAndCapacities=[{"VariantName": "green", "DesiredWeight": 1.0}],
        )
        print(f"Staging '{staging_endpoint}' shifted to 100% green")
    else:
        print(f"Staging '{staging_endpoint}' has no green variant (first run) — skipping weight shift")


def rollback_staging() -> None:
    sm = boto3.client("sagemaker", region_name=config.aws.region)
    endpoint_name = config.sagemaker.staging_endpoint
    ts = datetime.now().strftime("%Y%m%d%H%M%S")

    staging_config_name = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointConfigName"]
    staging_config = sm.describe_endpoint_config(EndpointConfigName=staging_config_name)
    variants = staging_config["ProductionVariants"]

    blue_variant = next((v for v in variants if v["VariantName"] == "blue"), None)
    if blue_variant is None:
        # First run: only a green variant exists (the new model is the only model).
        # Nothing stable to roll back to — leave the endpoint as-is.
        print("No blue variant found on staging — endpoint is on its first deployment, skipping rollback.")
        return

    blue_model_name = blue_variant["ModelName"]
    print(f"Rolling back staging to 100% blue (model: {blue_model_name})")

    rollback_config_name = f"mlops-toxic-staging-rollback-{ts}"
    sm.create_endpoint_config(
        EndpointConfigName=rollback_config_name,
        ProductionVariants=[
            {
                "VariantName": "blue",
                "ModelName": blue_model_name,
                "InstanceType": config.sagemaker.instance_type,
                "InitialInstanceCount": 1,
                "InitialVariantWeight": 1.0,
            }
        ],
    )

    sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=rollback_config_name)
    print(f"Staging rollback initiated with config '{rollback_config_name}'")
    _wait_for_endpoint(sm, endpoint_name)
    print("Staging rolled back to 100% blue.")


def simulate_failure() -> None:
    """Dev-only: update staging to a bad config to test SageMaker auto-rollback."""
    sm = boto3.client("sagemaker", region_name=config.aws.region)
    endpoint_name = config.sagemaker.staging_endpoint
    ts = datetime.now().strftime("%Y%m%d%H%M%S")

    # Current config (our "rollback target")
    current_config_name = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointConfigName"]
    print(f"Current config: {current_config_name}")

    # Create a broken config (references a non-existent model)
    bad_config_name = f"mlops-toxic-bad-{ts}"
    sm.create_endpoint_config(
        EndpointConfigName=bad_config_name,
        ProductionVariants=[
            {
                "VariantName": "blue",
                "ModelName": "nonexistent-model-for-failure-test",
                "InstanceType": "ml.t3.medium",
                "InitialInstanceCount": 1,
                "InitialVariantWeight": 1.0,
            }
        ],
    )

    print(f"Attempting update to bad config '{bad_config_name}'…")
    sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=bad_config_name)

    start = time.time()
    while True:
        status = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
        elapsed = time.time() - start
        print(f"[{elapsed:.0f}s] Status: {status}")
        if status == "InService":
            print(f"Endpoint recovered in {elapsed:.0f}s")
            break
        if status == "Failed":
            print("Endpoint entered Failed state — rolling back manually")
            sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=current_config_name)
            _wait_for_endpoint(sm, endpoint_name)
            break
        if elapsed > 600:
            raise TimeoutError("Endpoint did not recover within 10 minutes")
        time.sleep(30)


def _wait_for_endpoint(sm_client, endpoint_name: str, timeout: int = 1800) -> None:
    start = time.time()
    while True:
        status = sm_client.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
        elapsed = time.time() - start
        print(f"[{elapsed:.0f}s] {endpoint_name}: {status}")
        if status == "InService":
            print(f"Endpoint '{endpoint_name}' is InService")
            return
        if status in ("Failed", "OutOfService"):
            raise RuntimeError(f"Endpoint '{endpoint_name}' entered {status} state")
        if elapsed > timeout:
            raise TimeoutError(f"Endpoint '{endpoint_name}' did not reach InService within {timeout}s")
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--ecr-image-uri", type=str, default=config.sagemaker.ecr_image_uri)
    parser.add_argument(
        "--model-package-group",
        type=str,
        default="mlops-toxic-models",
    )
    parser.add_argument("--to-prod", action="store_true", help="Promote staging green to production")
    parser.add_argument("--rollback-staging", action="store_true", help="Roll staging back to 100% blue")
    parser.add_argument("--simulate-failure", action="store_true", help="Dev: test auto-rollback")
    args = parser.parse_args()

    if args.simulate_failure:
        simulate_failure()
        return

    if args.rollback_staging:
        rollback_staging()
        return

    ts = datetime.now().strftime("%Y%m%d%H%M%S")

    if args.to_prod:
        promote_to_prod(run_name=ts)
        return

    # ── Normal promotion flow ─────────────────────────────────────────────────
    input_dir = Path(args.input_dir)
    eval_path = input_dir / "evaluation" / "evaluation.json"
    with open(eval_path) as f:
        evaluation = json.load(f)

    best_model = evaluation["best_model"]
    best_f1 = evaluation["best_f1_macro"]
    prod_f1 = evaluation["prod_f1_macro"]
    is_first_run = prod_f1 == 0.0

    print(f"Promoting: {best_model} (f1={best_f1:.4f}) over prod (f1={prod_f1:.4f})")

    model_s3_env = __import__("os").environ.get("BEST_MODEL_S3_URI", "")

    if not model_s3_env:
        raise ValueError(
            "BEST_MODEL_S3_URI environment variable not set. "
            "The pipeline definition must inject this from the training step output."
        )

    register_to_mlflow(model_s3_env, evaluation, run_name=ts)
    register_to_sagemaker(
        model_s3_uri=model_s3_env,
        ecr_image_uri=args.ecr_image_uri,
        metrics=evaluation,
        model_package_group=args.model_package_group,
        is_first_run=is_first_run,
    )
    deploy_canary_to_staging(
        model_s3_uri=model_s3_env,
        ecr_image_uri=args.ecr_image_uri,
        run_name=ts,
    )


if __name__ == "__main__":
    main()
