"""
SageMaker v3 @step-decorated pipeline functions.

Each function represents one pipeline step. They are created inside `build_steps()`
so that infrastructure config (image_uri, role_arn, instance_type) can be bound
at pipeline-definition time via the @step decorator parameters.

Data flows between steps via function return values (serialized to S3 by SageMaker).
Large data (CSVs, model artifacts) is passed as S3 URIs (strings); small data
(metrics, evaluation results) is passed as Python dicts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sagemaker.mlops.workflow.function_step import step

from src.config import config


def build_steps(
    image_uri: str,
    role_arn: str,
    instance_type: str,
    pipeline_bucket: str,
):
    """Return the six @step-decorated pipeline functions bound to the given infra config.

    Returns:
        (ingest, preprocess, train_baseline, train_improved, evaluate, promote)
    """
    _env = {
        "PYTHONPATH": "/app",
        "MLFLOW_TRACKING_URI": config.mlflow.tracking_uri,
        "MLFLOW_EXPERIMENT_NAME": config.mlflow.experiment_name,
        "DATABRICKS_HOST": config.mlflow.databricks_host or "",
        "DATABRICKS_TOKEN": config.mlflow.databricks_token or "",
        "AWS_REGION": config.aws.region,
        "SAGEMAKER_ROLE_ARN": role_arn,
        "AWS_RAW_BUCKET": config.aws.raw_bucket,
        "AWS_PROCESSED_BUCKET": config.aws.processed_bucket,
        "AWS_MODELS_BUCKET": config.aws.models_bucket,
        "AWS_PIPELINE_BUCKET": pipeline_bucket,
        "SM_INSTANCE_TYPE": instance_type,
        "ECR_IMAGE_URI": image_uri,
        "STAGING_ENDPOINT": config.sagemaker.staging_endpoint,
        "PROD_ENDPOINT": config.sagemaker.prod_endpoint,
    }

    def sm_step(name: str, **extra_kwargs):
        """Partial @step decorator with shared infra config."""
        return step(
            name=name,
            image_uri=image_uri,
            role=role_arn,
            instance_type=instance_type,
            environment_variables=_env,
            **extra_kwargs,
        )

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    @sm_step(name="IngestStep")
    def ingest(input_s3_uri: str) -> str:
        """Validate raw CSV schema. Returns the same S3 URI after validation."""
        import sys
        sys.path.insert(0, "/app")
        from src.ingest import run_ingest
        return run_ingest(input_s3_uri)

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    @sm_step(name="PreprocessStep")
    def preprocess(raw_s3_uri: str, run_prefix: str) -> dict:
        """Preprocess CSV, split train/val, upload to S3. Returns URIs dict."""
        import os
        import sys
        sys.path.insert(0, "/app")
        from src.preprocess import run_preprocess
        return run_preprocess(raw_s3_uri, os.environ["AWS_PIPELINE_BUCKET"], run_prefix)

    # ── Steps 3 & 4: Train (baseline and improved run in parallel) ────────────
    @sm_step(name="TrainBaselineStep")
    def train_baseline(train_uri: str, val_uri: str, run_prefix: str) -> dict:
        """Train baseline model (TF-IDF + LogReg). Returns metrics + model S3 URI."""
        import os
        import sys
        sys.path.insert(0, "/app")
        from src.train import run_train
        return run_train(
            train_uri, val_uri,
            model_name="baseline",
            models_bucket=os.environ["AWS_MODELS_BUCKET"],
            run_prefix=run_prefix,
            max_features=10000,
            ngram_min=1,
            ngram_max=2,
            C=1.0,
            max_iter=1000,
        )

    @sm_step(name="TrainImprovedStep")
    def train_improved(train_uri: str, val_uri: str, run_prefix: str) -> dict:
        """Train improved model (tuned hyperparams). Returns metrics + model S3 URI."""
        import os
        import sys
        sys.path.insert(0, "/app")
        from src.train import run_train
        return run_train(
            train_uri, val_uri,
            model_name="improved",
            models_bucket=os.environ["AWS_MODELS_BUCKET"],
            run_prefix=run_prefix,
            max_features=20000,
            ngram_min=1,
            ngram_max=3,
            C=5.0,
            max_iter=2000,
        )

    # ── Step 5: Evaluate ──────────────────────────────────────────────────────
    @sm_step(name="EvaluateStep")
    def evaluate(baseline: dict, improved: dict, model_package_group: str) -> dict:
        """Compare candidates vs each other and production. Returns evaluation dict."""
        import sys
        sys.path.insert(0, "/app")
        from src.evaluate import run_evaluate
        return run_evaluate(baseline, improved, model_package_group)

    # ── Step 6a: Promote (runs only when condition passes) ────────────────────
    @sm_step(name="PromoteStep")
    def promote(eval_result: dict, ecr_image_uri: str, model_package_group: str) -> None:
        """Register best model to MLflow + SageMaker registry; deploy canary to staging."""
        import sys
        sys.path.insert(0, "/app")
        from datetime import datetime
        from src.promote import (
            register_to_mlflow,
            register_to_sagemaker,
            deploy_canary_to_staging,
        )
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        is_first_run = eval_result["prod_f1_macro"] == 0.0

        register_to_mlflow(
            model_s3_uri=eval_result["best_model_s3_uri"],
            metrics=eval_result,
            run_name=ts,
        )
        register_to_sagemaker(
            model_s3_uri=eval_result["best_model_s3_uri"],
            ecr_image_uri=ecr_image_uri,
            metrics=eval_result,
            model_package_group=model_package_group,
            is_first_run=is_first_run,
        )
        deploy_canary_to_staging(
            model_s3_uri=eval_result["best_model_s3_uri"],
            ecr_image_uri=ecr_image_uri,
            run_name=ts,
        )

    return ingest, preprocess, train_baseline, train_improved, evaluate, promote
