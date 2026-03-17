"""
SageMaker v3 Pipeline definition for the toxic comment classifier.

Pipeline steps:
  1. ingest        — @step: validate raw CSV from S3
  2. preprocess    — @step: TextPreprocessor + train/val split
  3. train_baseline — @step: baseline model (TF-IDF + LogReg)          ┐ parallel
  4. train_improved — @step: improved model (tuned hyperparams)         ┘
  5. evaluate      — @step: compare both models vs current prod
  6. condition     — ConditionStep: best_f1 >= prod_f1?
       if True  → promote_step (@step: register + deploy to staging)
       if False → FailStep("Challenger did not beat champion")

Usage:
  from pipelines.pipeline import get_pipeline
  pipeline = get_pipeline(role_arn=..., image_uri=..., pipeline_bucket=...)
  pipeline.upsert(role_arn=role_arn)
  pipeline.start(parameters={"InputS3Uri": "s3://..."})
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── SageMaker v3 imports ──────────────────────────────────────────────────────
from sagemaker.core.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.core.workflow.execution_variables import ExecutionVariables
from sagemaker.core.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core.workflow.step_outputs import get_step
from sagemaker.mlops.workflow.condition_step import ConditionStep
from sagemaker.mlops.workflow.fail_step import FailStep
from sagemaker.mlops.workflow.pipeline import Pipeline

from pipelines.steps import build_steps
from src.config import config


def get_pipeline(
    role_arn: str,
    image_uri: str,
    pipeline_bucket: str,
    region: str = None,
) -> Pipeline:
    import boto3
    region = region or config.aws.region
    boto_session = boto3.Session(region_name=region)
    session = PipelineSession(boto_session=boto_session, default_bucket=pipeline_bucket)

    # ── Pipeline parameters ────────────────────────────────────────────────────
    p_input_s3_uri = ParameterString(
        name="InputS3Uri",
        default_value=f"s3://{config.aws.raw_bucket}/train.csv",
    )
    p_f1_threshold = ParameterFloat(
        name="F1Threshold",
        default_value=config.sagemaker.f1_threshold,
    )
    p_run_name = ParameterString(
        name="RunName",
        default_value="pipeline-run",
    )
    p_model_package_group = ParameterString(
        name="ModelPackageGroup",
        default_value="mlops-toxic-models",
    )

    # ── Build @step-decorated functions ───────────────────────────────────────
    ingest, preprocess, train_baseline, train_improved, evaluate, promote = build_steps(
        image_uri=image_uri,
        role_arn=role_arn,
        instance_type=config.sagemaker.instance_type,
        pipeline_bucket=pipeline_bucket,
    )

    # ── Compose pipeline DAG ──────────────────────────────────────────────────
    # Each call returns a DelayedReturn; SageMaker resolves the DAG from dependencies.
    run_prefix = ExecutionVariables.PIPELINE_EXECUTION_ID

    raw_uri = ingest(p_input_s3_uri)
    splits = preprocess(raw_uri, run_prefix)

    baseline_result = train_baseline(splits["train_uri"], splits["val_uri"], run_prefix)
    improved_result = train_improved(splits["train_uri"], splits["val_uri"], run_prefix)

    eval_result = evaluate(baseline_result, improved_result, p_model_package_group)

    # ── Condition: best_f1 >= prod_f1 (threshold baked into evaluate step) ───
    condition = ConditionGreaterThanOrEqualTo(
        left=eval_result["best_f1_macro"],
        right=eval_result["prod_f1_macro"],
    )

    promote_delayed = promote(eval_result, image_uri, p_model_package_group)
    fail_step = FailStep(
        name="FailThreshold",
        error_message="Challenger did not beat champion",
    )

    condition_step = ConditionStep(
        name="CheckF1Threshold",
        conditions=[condition],
        if_steps=[get_step(promote_delayed)],
        else_steps=[fail_step],
    )

    # ── Assemble pipeline ──────────────────────────────────────────────────────
    # All @step DelayedReturn objects must be listed explicitly — the StepsCompiler
    # does NOT auto-discover upstream @step steps from condition expressions because
    # StepOutput._referenced_steps is not yet implemented in v3.5.0.
    # Note: promote_delayed is omitted here because it is already registered in
    # condition_step.if_steps and adding it again would raise a duplicate-name error.
    pipeline = Pipeline(
        name=config.sagemaker.pipeline_name,
        parameters=[p_input_s3_uri, p_f1_threshold, p_run_name, p_model_package_group],
        steps=[raw_uri, splits, baseline_result, improved_result, eval_result, condition_step],
        sagemaker_session=session,
    )

    return pipeline
