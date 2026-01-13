import io
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd

logger = logging.getLogger(__name__)


class EvidentlyMonitor:
    def __init__(
        self,
        reference_data_path: Optional[str] = None,
        models_bucket: Optional[str] = None,
        evidently_bucket: Optional[str] = None,
        cloudwatch_namespace: str = "ToxicCommentAPI",
        region: Optional[str] = None,
    ):
        self.reference_data_path = reference_data_path
        self.models_bucket = models_bucket or os.environ.get("MODELS_BUCKET", "")
        self.evidently_bucket = evidently_bucket or os.environ.get("EVIDENTLY_BUCKET", "")
        self.cloudwatch_namespace = cloudwatch_namespace
        self.region = region

        self._reference_data: Optional[pd.DataFrame] = None
        self._prediction_buffer: List[Dict[str, Any]] = []
        self._buffer_size = 1000

        self._s3_client = boto3.client("s3", region_name=region)
        self._cloudwatch_client = boto3.client("cloudwatch", region_name=region)

    def load_reference_data(self, s3_path: Optional[str] = None) -> pd.DataFrame:
        if s3_path:
            bucket, key = s3_path.replace("s3://", "").split("/", 1)
            response = self._s3_client.get_object(Bucket=bucket, Key=key)
            self._reference_data = pd.read_csv(io.BytesIO(response["Body"].read()))
        elif self.reference_data_path:
            self._reference_data = pd.read_csv(self.reference_data_path)
        else:
            try:
                response = self._s3_client.get_object(
                    Bucket=self.models_bucket,
                    Key="reference/reference_data.csv",
                )
                self._reference_data = pd.read_csv(io.BytesIO(response["Body"].read()))
            except Exception as e:
                logger.warning(f"Could not load reference data: {e}")
                self._reference_data = None

        if self._reference_data is not None:
            logger.info(f"Loaded {len(self._reference_data)} reference samples")

        return self._reference_data

    def log_prediction(
        self,
        text: str,
        predictions: Dict[str, float],
        is_toxic: bool,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self._prediction_buffer.append({
            "text": text,
            "text_length": len(text),
            "word_count": len(text.split()),
            "predictions": predictions,
            "is_toxic": is_toxic,
            "timestamp": (timestamp or datetime.now()).isoformat(),
            **{f"pred_{k}": v for k, v in predictions.items()},
        })

        if len(self._prediction_buffer) >= self._buffer_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self._prediction_buffer:
            return

        try:
            df = pd.DataFrame(self._prediction_buffer)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            buffer = io.BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)

            key = f"predictions/{timestamp}.csv"
            self._s3_client.put_object(
                Bucket=self.evidently_bucket,
                Key=key,
                Body=buffer.getvalue(),
                ContentType="text/csv",
            )

            logger.info(f"Flushed {len(self._prediction_buffer)} predictions to S3")
            self._prediction_buffer = []

        except Exception as e:
            logger.error(f"Failed to flush prediction buffer: {e}")

    def run_drift_detection(
        self,
        current_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        try:
            from evidently import ColumnMapping
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
        except ImportError:
            logger.error("Evidently is not installed")
            return {"error": "Evidently not available"}

        if self._reference_data is None:
            self.load_reference_data()

        if self._reference_data is None:
            return {"error": "No reference data available"}

        if current_data is None:
            current_data = self._load_recent_predictions()

        if current_data is None or len(current_data) == 0:
            return {"error": "No current data available"}

        numerical_features = ["text_length", "word_count"]
        prediction_columns = [
            "pred_toxic",
            "pred_severe_toxic",
            "pred_obscene",
            "pred_threat",
            "pred_insult",
            "pred_identity_hate",
        ]

        column_mapping = ColumnMapping(
            numerical_features=numerical_features,
            prediction=prediction_columns,
        )

        report = Report(metrics=[
            DataDriftPreset(),
        ])

        common_columns = list(
            set(self._reference_data.columns) & set(current_data.columns)
        )
        common_columns = [c for c in common_columns if c in numerical_features + prediction_columns]

        if not common_columns:
            return {"error": "No common columns for comparison"}

        report.run(
            reference_data=self._reference_data[common_columns],
            current_data=current_data[common_columns],
            column_mapping=column_mapping,
        )

        result = report.as_dict()
        drift_score = self._extract_drift_score(result)

        report_html = report.get_html()
        self._save_report(report_html, drift_score)

        self._publish_drift_metric(drift_score)

        return {
            "drift_score": drift_score,
            "drift_detected": drift_score > 0.15,
            "samples_analyzed": len(current_data),
            "reference_samples": len(self._reference_data),
            "timestamp": datetime.now().isoformat(),
        }

    def _load_recent_predictions(self, days: int = 1) -> Optional[pd.DataFrame]:
        try:
            from datetime import timedelta

            cutoff = datetime.now() - timedelta(days=days)
            cutoff_str = cutoff.strftime("%Y%m%d")

            paginator = self._s3_client.get_paginator("list_objects_v2")
            all_dfs = []

            for page in paginator.paginate(
                Bucket=self.evidently_bucket,
                Prefix="predictions/",
            ):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    file_date = key.split("/")[-1].split("_")[0]
                    if file_date >= cutoff_str:
                        response = self._s3_client.get_object(
                            Bucket=self.evidently_bucket,
                            Key=key,
                        )
                        df = pd.read_csv(io.BytesIO(response["Body"].read()))
                        all_dfs.append(df)

            if all_dfs:
                return pd.concat(all_dfs, ignore_index=True)
            return None

        except Exception as e:
            logger.error(f"Failed to load recent predictions: {e}")
            return None

    def _extract_drift_score(self, result: Dict) -> float:
        try:
            metrics = result.get("metrics", [])
            for metric in metrics:
                metric_result = metric.get("result", {})
                if "share_of_drifted_columns" in metric_result:
                    return metric_result["share_of_drifted_columns"]
            return 0.0
        except Exception:
            return 0.0

    def _save_report(self, html_content: str, drift_score: float) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        key = f"reports/drift_report_{timestamp}.html"

        self._s3_client.put_object(
            Bucket=self.evidently_bucket,
            Key=key,
            Body=html_content.encode("utf-8"),
            ContentType="text/html",
            Metadata={
                "drift_score": str(drift_score),
                "timestamp": timestamp,
            },
        )

        logger.info(f"Saved drift report to s3://{self.evidently_bucket}/{key}")
        return f"s3://{self.evidently_bucket}/{key}"

    def _publish_drift_metric(self, drift_score: float) -> None:
        try:
            self._cloudwatch_client.put_metric_data(
                Namespace=self.cloudwatch_namespace,
                MetricData=[
                    {
                        "MetricName": "DriftScore",
                        "Dimensions": [
                            {"Name": "Environment", "Value": os.environ.get("ENVIRONMENT", "dev")},
                        ],
                        "Value": drift_score,
                        "Unit": "None",
                    }
                ],
            )
            logger.info(f"Published drift score {drift_score} to CloudWatch")
        except Exception as e:
            logger.error(f"Failed to publish drift metric: {e}")

    def create_reference_dataset(
        self,
        training_data_path: str,
        sample_size: int = 10000,
    ) -> pd.DataFrame:
        if training_data_path.startswith("s3://"):
            bucket, key = training_data_path.replace("s3://", "").split("/", 1)
            response = self._s3_client.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(io.BytesIO(response["Body"].read()))
        else:
            df = pd.read_csv(training_data_path)

        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        if "comment_text" in df.columns:
            df["text_length"] = df["comment_text"].str.len()
            df["word_count"] = df["comment_text"].str.split().str.len()

        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        self._s3_client.put_object(
            Bucket=self.models_bucket,
            Key="reference/reference_data.csv",
            Body=buffer.getvalue(),
            ContentType="text/csv",
        )

        logger.info(f"Created reference dataset with {len(df)} samples")
        self._reference_data = df

        return df


def lambda_handler(event: Dict, context: Any) -> Dict:
    monitor = EvidentlyMonitor(
        models_bucket=os.environ.get("MODELS_BUCKET"),
        evidently_bucket=os.environ.get("EVIDENTLY_BUCKET"),
        cloudwatch_namespace=os.environ.get("CLOUDWATCH_NAMESPACE", "ToxicCommentAPI"),
    )

    result = monitor.run_drift_detection()

    if result.get("drift_detected"):
        sns_topic = os.environ.get("SNS_TOPIC_ARN")
        if sns_topic:
            sns = boto3.client("sns")
            sns.publish(
                TopicArn=sns_topic,
                Subject="[ALERT] Data Drift Detected",
                Message=json.dumps(result, indent=2),
            )

    return {
        "statusCode": 200,
        "body": result,
    }
