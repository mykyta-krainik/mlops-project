import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

PROJECT_NAME = os.environ.get("PROJECT_NAME", "mlops-toxic")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")
MODELS_BUCKET = os.environ.get("MODELS_BUCKET", "")
BASELINE_MODEL_GROUP = os.environ.get("BASELINE_MODEL_GROUP", "")
IMPROVED_MODEL_GROUP = os.environ.get("IMPROVED_MODEL_GROUP", "")
PROMOTION_THRESHOLD = float(os.environ.get("PROMOTION_THRESHOLD", "0.02"))
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", f"{PROJECT_NAME}-{ENVIRONMENT}-endpoint")

sagemaker_client = boto3.client("sagemaker")
s3_client = boto3.client("s3")
sns_client = boto3.client("sns")

# Lineage client
lineage_client = LineageClient() if LINEAGE_AVAILABLE else None


def get_training_job_metrics(job_name: str) -> Dict[str, float]:
    """Retrieve metrics from a completed training job."""
    try:
        response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        
        # Get metrics from training job output
        output_path = response.get("ModelArtifacts", {}).get("S3ModelArtifacts", "")
        
        if not output_path:
            logger.warning(f"No model artifacts found for job {job_name}")
            return {}

        # Try to get metrics from the output data directory
        output_data_path = response.get("OutputDataConfig", {}).get("S3OutputPath", "")
        if output_data_path:
            # Metrics are saved to /opt/ml/output/data/metrics.json during training
            metrics_key = f"training-output/{job_name}/output/metrics.json"
            
            try:
                bucket = MODELS_BUCKET
                response = s3_client.get_object(Bucket=bucket, Key=metrics_key)
                metrics = json.loads(response["Body"].read().decode())
                logger.info(f"Retrieved metrics from S3: {metrics}")
                return metrics
            except Exception as e:
                logger.warning(f"Could not retrieve metrics from S3: {e}")

        # Fallback: get final metrics from training job description
        final_metrics = response.get("FinalMetricDataList", [])
        metrics = {}
        for metric in final_metrics:
            metrics[metric["MetricName"]] = metric["Value"]
        
        logger.info(f"Retrieved metrics from training job: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Failed to get training job metrics: {e}")
        return {}


def get_current_production_metrics(model_package_group: str) -> Tuple[Optional[str], Dict[str, float]]:
    """Get metrics from the current approved production model."""
    try:
        # List model packages in the group
        response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_package_group,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )

        packages = response.get("ModelPackageSummaryList", [])
        
        if not packages:
            logger.info(f"No approved models found in {model_package_group}")
            return None, {}

        current_package_arn = packages[0]["ModelPackageArn"]
        
        # Get model package details
        package_details = sagemaker_client.describe_model_package(
            ModelPackageName=current_package_arn
        )

        # Get metrics from customer metadata
        metadata = package_details.get("CustomerMetadataProperties", {})
        metrics = {
            "roc_auc_macro": float(metadata.get("roc_auc_macro", 0)),
            "f1_macro": float(metadata.get("f1_macro", 0)),
            "accuracy": float(metadata.get("accuracy", 0)),
        }

        logger.info(f"Current production model: {current_package_arn}")
        logger.info(f"Current production metrics: {metrics}")

        return current_package_arn, metrics

    except Exception as e:
        logger.error(f"Failed to get current production metrics: {e}")
        return None, {}


def should_promote_model(
    new_metrics: Dict[str, float],
    current_metrics: Dict[str, float],
    threshold: float,
) -> Tuple[bool, str]:
    """
    Determine if the new model should be promoted.
    
    Primary metric: roc_auc_macro
    The new model is promoted if it beats the current model by the threshold.
    """
    new_roc_auc = new_metrics.get("roc_auc_macro", 0)
    current_roc_auc = current_metrics.get("roc_auc_macro", 0)

    improvement = new_roc_auc - current_roc_auc

    if current_roc_auc == 0:
        # No current model, promote the new one
        reason = "No current production model exists. Promoting new model."
        return True, reason

    if improvement >= threshold:
        reason = (
            f"New model ROC-AUC ({new_roc_auc:.4f}) is {improvement:.4f} better than "
            f"current ({current_roc_auc:.4f}), exceeding threshold ({threshold})."
        )
        return True, reason
    else:
        reason = (
            f"New model ROC-AUC ({new_roc_auc:.4f}) improvement ({improvement:.4f}) "
            f"does not exceed threshold ({threshold}). Current: {current_roc_auc:.4f}"
        )
        return False, reason


def register_and_approve_model(
    job_name: str,
    model_package_group: str,
    metrics: Dict[str, float],
    dataset_version: Optional[str] = None,
    dataset_metadata_uri: Optional[str] = None,
) -> str:
    job_details = sagemaker_client.describe_training_job(TrainingJobName=job_name)
    model_data_url = job_details["ModelArtifacts"]["S3ModelArtifacts"]
    training_image = job_details["AlgorithmSpecification"]["TrainingImage"]
    training_job_arn = job_details["TrainingJobArn"]

    inference_image = training_image.replace("-training:", "-inference:")

    customer_metadata = {
        "roc_auc_macro": str(metrics.get("roc_auc_macro", 0)),
        "f1_macro": str(metrics.get("f1_macro", 0)),
        "accuracy": str(metrics.get("accuracy", 0)),
        "training_job": job_name,
        "training_job_arn": training_job_arn,
        "promoted_at": datetime.now().isoformat(),
    }
    
    if dataset_version:
        customer_metadata["dataset_version"] = dataset_version
    if dataset_metadata_uri:
        customer_metadata["dataset_metadata_uri"] = dataset_metadata_uri

    # Create model package
    response = sagemaker_client.create_model_package(
        ModelPackageGroupName=model_package_group,
        ModelPackageDescription=f"Model from training job {job_name}",
        InferenceSpecification={
            "Containers": [
                {
                    "Image": inference_image,
                    "ModelDataUrl": model_data_url,
                }
            ],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        ModelApprovalStatus="Approved",
        CustomerMetadataProperties=customer_metadata,
    )

    model_package_arn = response["ModelPackageArn"]
    logger.info(f"Registered and approved model package: {model_package_arn}")

    return model_package_arn


def update_endpoint(model_package_arn: str, job_name: str) -> bool:
    """Update the SageMaker endpoint with the new model."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_name = f"{ENDPOINT_NAME}-model-{timestamp}"
        config_name = f"{ENDPOINT_NAME}-config-{timestamp}"

        # Get model package details
        package_details = sagemaker_client.describe_model_package(
            ModelPackageName=model_package_arn
        )

        container = package_details["InferenceSpecification"]["Containers"][0]

        # Create model
        sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": container["Image"],
                "ModelDataUrl": container["ModelDataUrl"],
            },
            ExecutionRoleArn=os.environ.get(
                "SAGEMAKER_ROLE_ARN",
                f"arn:aws:iam::{boto3.client('sts').get_caller_identity()['Account']}:role/{PROJECT_NAME}-{ENVIRONMENT}-sagemaker-execution-role",
            ),
        )

        # Create endpoint config
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": "primary",
                    "ModelName": model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": "ml.t3.medium",
                    "InitialVariantWeight": 1.0,
                }
            ],
            DataCaptureConfig={
                "EnableCapture": True,
                "InitialSamplingPercentage": 100,
                "DestinationS3Uri": f"s3://{MODELS_BUCKET}/data-capture",
                "CaptureOptions": [
                    {"CaptureMode": "Input"},
                    {"CaptureMode": "Output"},
                ],
            },
        )

        # Update endpoint
        sagemaker_client.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name,
        )

        logger.info(f"Updated endpoint {ENDPOINT_NAME} with new config {config_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to update endpoint: {e}")
        return False


def send_notification(
    job_name: str,
    promoted: bool,
    reason: str,
    metrics: Dict[str, float],
    model_type: str,
) -> None:
    """Send SNS notification about the promotion decision."""
    if not SNS_TOPIC_ARN:
        logger.info("No SNS topic configured, skipping notification")
        return

    status = "PROMOTED" if promoted else "NOT PROMOTED"
    
    message = {
        "default": f"Model Promotion: {status}",
        "email": f"""
Model Promotion Decision: {status}

Training Job: {job_name}
Model Type: {model_type}
Decision: {reason}

Metrics:
- ROC-AUC (macro): {metrics.get('roc_auc_macro', 'N/A'):.4f}
- F1 (macro): {metrics.get('f1_macro', 'N/A'):.4f}
- Accuracy: {metrics.get('accuracy', 'N/A'):.4f}

Time: {datetime.now().isoformat()}
Environment: {ENVIRONMENT}
""",
    }

    try:
        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=json.dumps(message),
            MessageStructure="json",
            Subject=f"[{ENVIRONMENT.upper()}] Model {status}: {job_name}",
        )
        logger.info("Sent SNS notification")
    except Exception as e:
        logger.error(f"Failed to send SNS notification: {e}")


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for model promotion.
    
    Triggered by EventBridge when a SageMaker training job completes.
    """
    logger.info(f"Event: {json.dumps(event)}")

    try:
        # Extract training job information from EventBridge event
        detail = event.get("detail", {})
        job_name = detail.get("TrainingJobName", "")
        job_status = detail.get("TrainingJobStatus", "")

        if not job_name:
            # Check if job name is in a different format
            job_name = event.get("TrainingJobName", "")

        if not job_name:
            raise ValueError("No training job name found in event")

        logger.info(f"Processing training job: {job_name}, status: {job_status}")

        if job_status != "Completed":
            logger.info(f"Training job {job_name} is not completed (status: {job_status})")
            return {
                "statusCode": 200,
                "body": {"message": f"Skipping non-completed job: {job_status}"},
            }

        # Determine model type from job name
        if "improved" in job_name.lower():
            model_type = "improved"
            model_package_group = IMPROVED_MODEL_GROUP
        else:
            model_type = "baseline"
            model_package_group = BASELINE_MODEL_GROUP

        # Step 1: Get new model metrics
        new_metrics = get_training_job_metrics(job_name)
        
        if not new_metrics or new_metrics.get("roc_auc_macro", 0) == 0:
            logger.warning(f"No valid metrics found for job {job_name}")
            return {
                "statusCode": 400,
                "body": {"message": "No valid metrics found for training job"},
            }

        # Step 2: Get current production model metrics
        current_package_arn, current_metrics = get_current_production_metrics(
            model_package_group
        )

        # Step 3: Decide whether to promote
        should_promote, reason = should_promote_model(
            new_metrics, current_metrics, PROMOTION_THRESHOLD
        )

        logger.info(f"Promotion decision: {should_promote}, reason: {reason}")

        endpoint_updated = False
        model_package_arn = None

        dataset_version = None
        dataset_metadata_uri = None

        try:
            job_details = sagemaker_client.describe_training_job(TrainingJobName=job_name)
            tags = job_details.get("Tags", [])
            
            for tag in tags:
                if tag["Key"] == "DatasetVersion":
                    dataset_version = tag["Value"]
                elif tag["Key"] == "DatasetMetadataURI":
                    dataset_metadata_uri = tag["Value"]
        except Exception as e:
            logger.warning(f"Could not extract dataset version from tags: {e}")

        if should_promote:
            model_package_arn = register_and_approve_model(
                job_name, model_package_group, new_metrics
            )

            # Step 5: Update the endpoint
            endpoint_updated = update_endpoint(model_package_arn, job_name)

        # Step 6: Send notification
        send_notification(job_name, should_promote, reason, new_metrics, model_type)

        response = {
            "statusCode": 200,
            "body": {
                "training_job": job_name,
                "model_type": model_type,
                "promoted": should_promote,
                "reason": reason,
                "new_metrics": new_metrics,
                "current_metrics": current_metrics,
                "model_package_arn": model_package_arn,
                "endpoint_updated": endpoint_updated,
            },
        }

        logger.info(f"Response: {json.dumps(response)}")
        return response

    except Exception as e:
        logger.error(f"Error in model promotion: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": {"error": str(e)},
        }


# For local testing
if __name__ == "__main__":
    # Mock EventBridge event for testing
    test_event = {
        "detail": {
            "TrainingJobName": "mlops-toxic-dev-baseline-20240115120000",
            "TrainingJobStatus": "Completed",
        }
    }
    result = handler(test_event, None)
    print(json.dumps(result, indent=2))
