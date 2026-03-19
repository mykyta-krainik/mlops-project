import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
CAPTURE_LOOKBACK_DAYS = int(os.getenv("CAPTURE_LOOKBACK_DAYS", "7"))


def download_reference(s3_client, bucket: str, prefix: str = "reference/reference.parquet") -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        s3_client.download_file(bucket, prefix, tmp.name)
        return pd.read_parquet(tmp.name)


def download_capture_data(s3_client, pipeline_bucket: str, endpoint_name: str) -> pd.DataFrame:
    import base64
    import json as jsonlib

    prefix = f"data-capture/{endpoint_name}/"
    cutoff = datetime.now(timezone.utc) - timedelta(days=CAPTURE_LOOKBACK_DAYS)

    paginator = s3_client.get_paginator("list_objects_v2")
    rows = []

    for page in paginator.paginate(Bucket=pipeline_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["LastModified"] < cutoff:
                continue
            response = s3_client.get_object(Bucket=pipeline_bucket, Key=obj["Key"])
            for line in response["Body"].iter_lines():
                if not line:
                    continue
                record = jsonlib.loads(line)
                try:
                    input_data = record["captureData"]["endpointInput"]["data"]
                    # Data Capture encodes payload as base64
                    try:
                        decoded = base64.b64decode(input_data).decode("utf-8")
                    except Exception:
                        decoded = input_data
                    payload = jsonlib.loads(decoded)
                    comment = payload.get("comment") or (payload.get("comments") or [None])[0]
                    if comment:
                        rows.append({"comment_text": comment})
                except (KeyError, IndexError, ValueError):
                    pass

    if not rows:
        print("No captured data found in the lookback window.")
        return pd.DataFrame(columns=["comment_text"])

    return pd.DataFrame(rows)


def run_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> tuple[dict, str]:
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    text_cols = ["comment_text"]
    column_mapping = ColumnMapping(text_features=text_cols)

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df[text_cols] if text_cols[0] in reference_df.columns else reference_df,
        current_data=current_df[text_cols] if text_cols[0] in current_df.columns else current_df,
        column_mapping=column_mapping,
    )

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as html_file:
        report.save_html(html_file.name)
        html_path = html_file.name

    summary = report.as_dict()
    drift_score = summary.get("metrics", [{}])[0].get("result", {}).get("share_of_drifted_columns", 0.0)

    return {"drift_score": drift_score, "summary": summary}, html_path


def upload_reports(s3_client, pipeline_bucket: str, json_summary: dict, html_path: str) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    prefix = f"drift-reports/{date_str}"

    html_key = f"{prefix}/drift_report.html"
    json_key = f"{prefix}/drift_summary.json"

    s3_client.upload_file(html_path, pipeline_bucket, html_key)
    s3_client.put_object(
        Bucket=pipeline_bucket,
        Key=json_key,
        Body=json.dumps(json_summary, default=str),
        ContentType="application/json",
    )

    print(f"Drift report: s3://{pipeline_bucket}/{html_key}")
    print(f"Drift summary: s3://{pipeline_bucket}/{json_key}")

    import shutil
    shutil.copy2(html_path, f"/tmp/drift_report_{date_str}.html")

    return f"s3://{pipeline_bucket}/{html_key}"


def publish_cloudwatch_metric(drift_score: float) -> None:
    cw = boto3.client("cloudwatch", region_name=config.aws.region)
    cw.put_metric_data(
        Namespace="mlops-toxic/monitoring",
        MetricData=[
            {
                "MetricName": "DriftScore",
                "Value": drift_score,
                "Unit": "None",
                "Timestamp": datetime.now(timezone.utc),
                "Dimensions": [{"Name": "Endpoint", "Value": "mlops-toxic-prod"}],
            }
        ],
    )
    print(f"Published DriftScore={drift_score:.4f} to CloudWatch")


def upload_status(s3_client, pipeline_bucket: str, status: dict) -> None:
    date_str = datetime.now().strftime("%Y-%m-%d")
    s3_client.put_object(
        Bucket=pipeline_bucket,
        Key=f"drift-reports/{date_str}/drift_summary.json",
        Body=json.dumps(status, default=str),
        ContentType="application/json",
    )
    print(f"Status written to s3://{pipeline_bucket}/drift-reports/{date_str}/drift_summary.json")


def main() -> None:
    s3 = boto3.client("s3", region_name=config.aws.region)
    drift_score = 0.0
    drift_detected = False

    try:
        print("Downloading reference dataset…")
        try:
            reference_df = download_reference(s3, config.aws.processed_bucket)
            print(f"Reference: {len(reference_df)} rows")
        except Exception as e:
            print(f"Could not download reference dataset: {e}")
            print("Skipping drift check (reference not available — run pipeline first).")
            upload_status(s3, config.aws.pipeline_bucket, {"status": "skipped", "reason": "reference_unavailable", "error": str(e)})
            return

        print("Downloading recent captured data…")
        current_df = download_capture_data(s3, config.aws.pipeline_bucket, config.sagemaker.prod_endpoint)
        print(f"Current window: {len(current_df)} rows")

        if len(current_df) < 300:
            print(f"Not enough captured data for drift analysis ({len(current_df)} rows, need >= 300). Skipping.")
            upload_status(s3, config.aws.pipeline_bucket, {"status": "skipped", "reason": "insufficient_data", "captured_rows": len(current_df), "required_rows": 300})
            return

        print("Running Evidently drift report…")
        json_summary, html_path = run_drift_report(reference_df, current_df)
        drift_score = json_summary["drift_score"]

        print(f"Drift score: {drift_score:.4f} (threshold: {DRIFT_THRESHOLD})")

        upload_reports(s3, config.aws.pipeline_bucket, json_summary, html_path)

        if drift_score > DRIFT_THRESHOLD:
            drift_detected = True

    finally:
        publish_cloudwatch_metric(drift_score)

    if drift_detected:
        print(f"DRIFT DETECTED: {drift_score:.4f} > {DRIFT_THRESHOLD}")
        sys.exit(1)

    print("No significant drift detected.")


if __name__ == "__main__":
    main()
