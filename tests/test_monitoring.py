"""
Tests for monitoring components.
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime


class TestCloudWatchLogger:
    """Tests for CloudWatch metrics logger."""

    @pytest.fixture
    def mock_cloudwatch(self):
        """Mock CloudWatch client."""
        with patch("boto3.client") as mock_client:
            mock_cw = MagicMock()
            mock_client.return_value = mock_cw
            yield mock_cw

    def test_log_prediction(self, mock_cloudwatch):
        """Test logging a single prediction."""
        from src.monitoring.cloudwatch_logger import CloudWatchMetricsLogger

        logger = CloudWatchMetricsLogger(
            namespace="TestNamespace",
            flush_interval=0,  # Disable background flush
        )

        logger.log_prediction(
            latency_ms=50.0,
            is_toxic=True,
            is_error=False,
        )

        # Check internal counters
        assert logger._counters["predictions"] == 1
        assert logger._counters["toxic_predictions"] == 1
        assert logger._counters["errors"] == 0
        assert len(logger._latencies) == 1

        logger.stop()

    def test_log_batch_prediction(self, mock_cloudwatch):
        """Test logging a batch prediction."""
        from src.monitoring.cloudwatch_logger import CloudWatchMetricsLogger

        logger = CloudWatchMetricsLogger(
            namespace="TestNamespace",
            flush_interval=0,
        )

        logger.log_batch_prediction(
            latency_ms=200.0,
            batch_size=10,
            toxic_count=3,
            is_error=False,
        )

        assert logger._counters["predictions"] == 10
        assert logger._counters["toxic_predictions"] == 3
        assert logger._counters["batch_requests"] == 1

        logger.stop()

    def test_flush_publishes_metrics(self, mock_cloudwatch):
        """Test that flush publishes metrics to CloudWatch."""
        from src.monitoring.cloudwatch_logger import CloudWatchMetricsLogger

        logger = CloudWatchMetricsLogger(
            namespace="TestNamespace",
            flush_interval=0,
        )

        # Log some predictions
        logger.log_prediction(latency_ms=50.0, is_toxic=True, is_error=False)
        logger.log_prediction(latency_ms=100.0, is_toxic=False, is_error=False)

        # Flush metrics
        logger.flush()

        # Verify CloudWatch was called
        mock_cloudwatch.put_metric_data.assert_called()

        logger.stop()

    def test_custom_metric(self, mock_cloudwatch):
        """Test logging a custom metric."""
        from src.monitoring.cloudwatch_logger import CloudWatchMetricsLogger

        logger = CloudWatchMetricsLogger(
            namespace="TestNamespace",
            flush_interval=0,
        )

        logger.log_custom_metric(
            metric_name="CustomMetric",
            value=42.0,
            unit="Count",
            dimensions={"Custom": "Dimension"},
        )

        mock_cloudwatch.put_metric_data.assert_called_once()

        logger.stop()


class TestEvidentlyMonitor:
    """Tests for Evidently drift detection."""

    @pytest.fixture
    def mock_s3(self):
        """Mock S3 client."""
        with patch("boto3.client") as mock_client:
            mock_s3 = MagicMock()
            mock_client.return_value = mock_s3
            yield mock_s3

    def test_log_prediction(self, mock_s3):
        """Test logging a prediction for drift analysis."""
        from src.monitoring.evidently_monitor import EvidentlyMonitor

        monitor = EvidentlyMonitor(
            models_bucket="test-models",
            evidently_bucket="test-evidently",
        )

        monitor.log_prediction(
            text="Test comment",
            predictions={"toxic": 0.8, "severe_toxic": 0.1},
            is_toxic=True,
        )

        assert len(monitor._prediction_buffer) == 1
        assert monitor._prediction_buffer[0]["text"] == "Test comment"
        assert monitor._prediction_buffer[0]["is_toxic"] is True

    def test_buffer_flush(self, mock_s3):
        """Test that buffer flushes to S3."""
        from src.monitoring.evidently_monitor import EvidentlyMonitor

        monitor = EvidentlyMonitor(
            models_bucket="test-models",
            evidently_bucket="test-evidently",
        )
        monitor._buffer_size = 2  # Small buffer for testing

        # Log two predictions to trigger flush
        monitor.log_prediction("Comment 1", {"toxic": 0.5}, False)
        monitor.log_prediction("Comment 2", {"toxic": 0.8}, True)

        # Check S3 was called
        mock_s3.put_object.assert_called_once()

        # Buffer should be empty after flush
        assert len(monitor._prediction_buffer) == 0

    def test_extract_drift_score(self, mock_s3):
        """Test extracting drift score from Evidently report."""
        from src.monitoring.evidently_monitor import EvidentlyMonitor

        monitor = EvidentlyMonitor()

        result = {
            "metrics": [
                {
                    "result": {
                        "share_of_drifted_columns": 0.25,
                    }
                }
            ]
        }

        score = monitor._extract_drift_score(result)
        assert score == 0.25

    def test_extract_drift_score_missing(self, mock_s3):
        """Test extracting drift score when not present."""
        from src.monitoring.evidently_monitor import EvidentlyMonitor

        monitor = EvidentlyMonitor()

        result = {"metrics": []}
        score = monitor._extract_drift_score(result)
        assert score == 0.0


class TestAPIWithMonitoring:
    """Tests for API integration with monitoring."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        with patch.dict(os.environ, {
            "USE_SAGEMAKER": "false",
            "ENABLE_CLOUDWATCH": "false",
            "ENABLE_REVIEW_DB": "false",
        }):
            from src.api.app import create_app
            
            app = create_app()
            app.config["TESTING"] = True
            
            with app.test_client() as client:
                yield client

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.get_json()
        assert "status" in data
        assert "model_loaded" in data

    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.get_json()
        assert "version" in data
        assert "model_type" in data
        assert "target_labels" in data
