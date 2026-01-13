import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class CloudWatchMetricsLogger:
    def __init__(
        self,
        namespace: str = "ToxicCommentAPI",
        region: Optional[str] = None,
        environment: Optional[str] = None,
        flush_interval: int = 60,
        max_buffer_size: int = 100,
    ):
        self.namespace = namespace
        self.region = region
        self.environment = environment or os.environ.get("ENVIRONMENT", "dev")
        self.flush_interval = flush_interval
        self.max_buffer_size = max_buffer_size

        self._client = boto3.client("cloudwatch", region_name=region)
        self._metrics_buffer: List[Dict] = []
        self._buffer_lock = threading.Lock()
        self._last_flush = time.time()

        self._counters = defaultdict(int)
        self._latencies: List[float] = []
        self._counter_lock = threading.Lock()

        self._stop_flush = False
        self._flush_thread = threading.Thread(target=self._background_flush, daemon=True)
        self._flush_thread.start()

    def _background_flush(self) -> None:
        while not self._stop_flush:
            time.sleep(self.flush_interval)
            try:
                self._flush_aggregated_metrics()
            except Exception as e:
                logger.error(f"Error in background flush: {e}")

    def _flush_aggregated_metrics(self) -> None:
        with self._counter_lock:
            if not self._counters and not self._latencies:
                return

            metrics_data = []
            timestamp = datetime.now(datetime.timezone.utc)
            dimensions = [{"Name": "Environment", "Value": self.environment}]

            if self._counters["predictions"] > 0:
                metrics_data.append({
                    "MetricName": "PredictionCount",
                    "Dimensions": dimensions,
                    "Timestamp": timestamp,
                    "Value": self._counters["predictions"],
                    "Unit": "Count",
                })

            if self._counters["toxic_predictions"] > 0:
                metrics_data.append({
                    "MetricName": "ToxicPredictionCount",
                    "Dimensions": dimensions,
                    "Timestamp": timestamp,
                    "Value": self._counters["toxic_predictions"],
                    "Unit": "Count",
                })

            if self._counters["errors"] > 0:
                metrics_data.append({
                    "MetricName": "ErrorCount",
                    "Dimensions": dimensions,
                    "Timestamp": timestamp,
                    "Value": self._counters["errors"],
                    "Unit": "Count",
                })

            total = self._counters["predictions"]
            if total > 0:
                toxicity_rate = (self._counters["toxic_predictions"] / total) * 100
                metrics_data.append({
                    "MetricName": "ToxicityRate",
                    "Dimensions": dimensions,
                    "Timestamp": timestamp,
                    "Value": toxicity_rate,
                    "Unit": "Percent",
                })

                error_rate = (self._counters["errors"] / (total + self._counters["errors"])) * 100
                metrics_data.append({
                    "MetricName": "ErrorRate",
                    "Dimensions": dimensions,
                    "Timestamp": timestamp,
                    "Value": error_rate,
                    "Unit": "Percent",
                })

            if self._latencies:
                avg_latency = sum(self._latencies) / len(self._latencies)
                max_latency = max(self._latencies)

                metrics_data.append({
                    "MetricName": "AverageLatency",
                    "Dimensions": dimensions,
                    "Timestamp": timestamp,
                    "Value": avg_latency,
                    "Unit": "Milliseconds",
                })

                metrics_data.append({
                    "MetricName": "MaxLatency",
                    "Dimensions": dimensions,
                    "Timestamp": timestamp,
                    "Value": max_latency,
                    "Unit": "Milliseconds",
                })

                sorted_latencies = sorted(self._latencies)
                p95_index = int(len(sorted_latencies) * 0.95)
                p95_latency = sorted_latencies[min(p95_index, len(sorted_latencies) - 1)]

                metrics_data.append({
                    "MetricName": "P95Latency",
                    "Dimensions": dimensions,
                    "Timestamp": timestamp,
                    "Value": p95_latency,
                    "Unit": "Milliseconds",
                })

            self._counters = defaultdict(int)
            self._latencies = []

        if metrics_data:
            try:
                for i in range(0, len(metrics_data), 20):
                    batch = metrics_data[i : i + 20]
                    self._client.put_metric_data(
                        Namespace=self.namespace,
                        MetricData=batch,
                    )
                logger.debug(f"Published {len(metrics_data)} metrics to CloudWatch")
            except ClientError as e:
                logger.error(f"Failed to publish metrics: {e}")

    def log_prediction(
        self,
        latency_ms: float,
        is_toxic: bool,
        is_error: bool = False,
    ) -> None:
        with self._counter_lock:
            self._counters["predictions"] += 1
            if is_toxic:
                self._counters["toxic_predictions"] += 1
            if is_error:
                self._counters["errors"] += 1
            self._latencies.append(latency_ms)

        if len(self._latencies) >= self.max_buffer_size:
            self._flush_aggregated_metrics()

    def log_batch_prediction(
        self,
        latency_ms: float,
        batch_size: int,
        toxic_count: int,
        is_error: bool = False,
    ) -> None:
        with self._counter_lock:
            self._counters["predictions"] += batch_size
            self._counters["toxic_predictions"] += toxic_count
            self._counters["batch_requests"] += 1
            if is_error:
                self._counters["errors"] += 1
            self._latencies.append(latency_ms)

    def log_custom_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "None",
        dimensions: Optional[Dict[str, str]] = None,
    ) -> None:
        all_dimensions = [{"Name": "Environment", "Value": self.environment}]
        if dimensions:
            all_dimensions.extend(
                [{"Name": k, "Value": v} for k, v in dimensions.items()]
            )

        try:
            self._client.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {
                        "MetricName": metric_name,
                        "Dimensions": all_dimensions,
                        "Timestamp": datetime.now(datetime.timezone.utc),
                        "Value": value,
                        "Unit": unit,
                    }
                ],
            )
        except ClientError as e:
            logger.error(f"Failed to log custom metric {metric_name}: {e}")

    def log_drift_score(self, drift_score: float) -> None:
        self.log_custom_metric(
            metric_name="DriftScore",
            value=drift_score,
            unit="None",
        )

    def log_endpoint_health(self, is_healthy: bool) -> None:
        self.log_custom_metric(
            metric_name="EndpointHealth",
            value=1.0 if is_healthy else 0.0,
            unit="None",
        )

    def flush(self) -> None:
        self._flush_aggregated_metrics()

    def stop(self) -> None:
        self._stop_flush = True
        self._flush_thread.join(timeout=5)
        self.flush()


class CloudWatchMetricsLoggerFactory:
    _instance: Optional[CloudWatchMetricsLogger] = None

    @classmethod
    def get_logger(
        cls,
        namespace: str = "ToxicCommentAPI",
        region: Optional[str] = None,
    ) -> CloudWatchMetricsLogger:
        if cls._instance is None:
            cls._instance = CloudWatchMetricsLogger(namespace, region)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        if cls._instance:
            cls._instance.stop()
        cls._instance = None
