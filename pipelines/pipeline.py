import sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    region = region or config.aws.region
    boto_session = boto3.Session(region_name=region)
    session = PipelineSession(boto_session=boto_session, default_bucket=pipeline_bucket)

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

    ingest, preprocess, train_baseline, train_improved, evaluate, promote = build_steps(
        image_uri=image_uri,
        role_arn=role_arn,
        instance_type=config.sagemaker.instance_type,
        pipeline_bucket=pipeline_bucket,
    )

    run_prefix = ExecutionVariables.PIPELINE_EXECUTION_ID

    raw_uri = ingest(p_input_s3_uri)
    splits = preprocess(raw_uri, run_prefix)

    baseline_result = train_baseline(splits["train_uri"], splits["val_uri"], run_prefix)
    improved_result = train_improved(splits["train_uri"], splits["val_uri"], run_prefix)

    eval_result = evaluate(baseline_result, improved_result, p_model_package_group)

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

    pipeline = Pipeline(
        name=config.sagemaker.pipeline_name,
        parameters=[p_input_s3_uri, p_f1_threshold, p_run_name, p_model_package_group],
        steps=[raw_uri, splits, baseline_result, improved_result, eval_result, condition_step],
        sagemaker_session=session,
    )

    return pipeline
