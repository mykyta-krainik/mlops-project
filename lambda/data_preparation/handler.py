import io
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd
from sqlalchemy import select

from db_manager import DatabaseManager
from db_models import ReviewedComment

logger = logging.getLogger()
logger.setLevel(logging.INFO)

PROJECT_NAME = os.environ.get("PROJECT_NAME", "mlops-toxic")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")
RAW_DATA_BUCKET = os.environ.get("RAW_DATA_BUCKET", "")
PROCESSED_DATA_BUCKET = os.environ.get("PROCESSED_DATA_BUCKET", "")
MODELS_BUCKET = os.environ.get("MODELS_BUCKET", "")
DB_SECRET_ARN = os.environ.get("DB_SECRET_ARN", "")
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "")
TRAINING_IMAGE_URI = os.environ.get("TRAINING_IMAGE_URI", "")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")

s3_client = boto3.client("s3")
sagemaker_client = boto3.client("sagemaker")

db_manager = DatabaseManager(secret_arn=DB_SECRET_ARN)


def list_raw_data_files() -> List[str]:
    files = []
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=RAW_DATA_BUCKET):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".csv"):
                files.append(key)

    logger.info(f"Found {len(files)} CSV files in raw data bucket")
    return files


def load_raw_data(files: List[str]) -> pd.DataFrame:
    all_dfs = []

    for file_key in files:
        logger.info(f"Loading {file_key}")
        response = s3_client.get_object(Bucket=RAW_DATA_BUCKET, Key=file_key)
        df = pd.read_csv(io.BytesIO(response["Body"].read()))
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No raw data files found")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} samples from raw data")

    return combined_df


def load_reviewed_comments(last_training_date: Optional[datetime] = None) -> pd.DataFrame:
    with db_manager.get_session() as session:
        stmt = select(ReviewedComment).where(
            ReviewedComment.reviewed_labels.is_not(None)
        )
        
        if last_training_date:
            stmt = stmt.where(ReviewedComment.reviewed_at > last_training_date)
        
        results = session.execute(stmt).scalars().all()
        logger.info(f"Loaded {len(results)} reviewed comments from database")
        
        if len(results) == 0:
            return pd.DataFrame()
        
        records = []
        for comment in results:
            labels = comment.reviewed_labels or {}
            records.append({
                "id": f"{comment.id}_reviewed",
                "comment_text": comment.comment_text,
                "toxic": labels.get("toxic", 0),
                "severe_toxic": labels.get("severe_toxic", 0),
                "obscene": labels.get("obscene", 0),
                "threat": labels.get("threat", 0),
                "insult": labels.get("insult", 0),
                "identity_hate": labels.get("identity_hate", 0),
            })
        
        result_df = pd.DataFrame(records)
        return result_df


def merge_datasets(raw_df: pd.DataFrame, reviewed_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "id",
        "comment_text",
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    for col in required_columns:
        if col not in raw_df.columns:
            raise ValueError(f"Missing required column in raw data: {col}")

    raw_df = raw_df[required_columns].copy()

    if len(reviewed_df) > 0:
        combined_df = pd.concat([raw_df, reviewed_df], ignore_index=True)
        
        logger.info(
            f"Combined dataset: {len(raw_df)} raw + {len(reviewed_df)} reviewed = {len(combined_df)} total"
        )
    else:
        combined_df = raw_df
        logger.info(f"No reviewed comments to add, using {len(raw_df)} raw samples")

    combined_df = combined_df.drop_duplicates(subset=["comment_text"], keep="last")
    logger.info(f"After deduplication: {len(combined_df)} samples")

    return combined_df


def upload_training_data(df: pd.DataFrame) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    key = f"training/train_{timestamp}.csv"

    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    s3_client.put_object(
        Bucket=PROCESSED_DATA_BUCKET,
        Key=key,
        Body=buffer.getvalue(),
        ContentType="text/csv",
    )

    logger.info(f"Uploaded training data to s3://{PROCESSED_DATA_BUCKET}/{key}")

    return f"s3://{PROCESSED_DATA_BUCKET}/{key}"


def start_training_job(
    model_type: str, training_data_uri: str, timestamp: str
) -> str:
    job_name = f"{PROJECT_NAME}-{ENVIRONMENT}-{model_type}-{timestamp}"

    hyperparameters = {
        "model-type": model_type,
        "max-features": "10000" if model_type == "baseline" else "15000",
        "lr-c": "1.0" if model_type == "baseline" else "0.5",
        "max-iter": "1000" if model_type == "baseline" else "2000",
        "test-size": "0.2",
        "random-state": "42",
    }

    if MLFLOW_TRACKING_URI:
        hyperparameters["mlflow-tracking-uri"] = MLFLOW_TRACKING_URI

    training_data_uri_parts = training_data_uri.replace("s3://", "").split("/", 1)
    training_bucket = training_data_uri_parts[0]
    training_key = training_data_uri_parts[1]

    training_job_config = {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": TRAINING_IMAGE_URI,
            "TrainingInputMode": "File",
        },
        "RoleArn": SAGEMAKER_ROLE_ARN,
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{training_bucket}/{os.path.dirname(training_key)}/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "text/csv",
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{MODELS_BUCKET}/training-output/{model_type}/",
        },
        "ResourceConfig": {
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1,
            "VolumeSizeInGB": 5,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 3600
        },
        "HyperParameters": hyperparameters,
        "EnableManagedSpotTraining": False,
        "Tags": [
            {"Key": "Project", "Value": PROJECT_NAME},
            {"Key": "Environment", "Value": ENVIRONMENT},
            {"Key": "ModelType", "Value": model_type},
        ],
    }

    response = sagemaker_client.create_training_job(**training_job_config)
    logger.info(f"Started training job: {job_name}")

    return job_name


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for data preparation and training trigger.
    
    This function is triggered by EventBridge on a weekly schedule.
    """
    logger.info(f"Event: {json.dumps(event)}")
    logger.info("Starting data preparation pipeline")

    try:
        # Step 1: List and load raw data from S3
        raw_files = list_raw_data_files()
        raw_df = load_raw_data(raw_files)

        # Step 2: Load reviewed comments from RDS
        try:
            reviewed_df = load_reviewed_comments()
        except Exception as e:
            logger.warning(f"Failed to load reviewed comments: {e}")
            reviewed_df = pd.DataFrame()

        # Step 3: Merge datasets
        combined_df = merge_datasets(raw_df, reviewed_df)

        # Step 4: Upload to processed data bucket
        training_data_uri = upload_training_data(combined_df)

        # Step 5: Start training jobs for both model types
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        baseline_job = start_training_job("baseline", training_data_uri, timestamp)
        improved_job = start_training_job("improved", training_data_uri, timestamp)

        response = {
            "statusCode": 200,
            "body": {
                "message": "Data preparation and training jobs started successfully",
                "training_data_uri": training_data_uri,
                "samples_count": len(combined_df),
                "raw_samples": len(raw_df),
                "reviewed_samples": len(reviewed_df),
                "training_jobs": {
                    "baseline": baseline_job,
                    "improved": improved_job,
                },
            },
        }

        logger.info(f"Response: {json.dumps(response)}")
        return response

    except Exception as e:
        logger.error(f"Error in data preparation: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": {"error": str(e)},
        }


# For local testing
if __name__ == "__main__":
    # Mock event for testing
    test_event = {"source": "local-test"}
    result = handler(test_event, None)
    print(json.dumps(result, indent=2))
