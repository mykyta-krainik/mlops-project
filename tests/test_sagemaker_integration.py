"""
Integration tests for SageMaker components.

These tests verify the SageMaker training and inference functionality.
Some tests require AWS credentials and a running SageMaker endpoint.
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch


class TestSageMakerClient:
    """Tests for the SageMaker client."""

    @pytest.fixture
    def mock_sagemaker_runtime(self):
        """Create a mock SageMaker runtime client."""
        with patch("boto3.client") as mock_client:
            mock_runtime = MagicMock()
            mock_client.return_value = mock_runtime
            yield mock_runtime

    def test_predict_single_comment(self, mock_sagemaker_runtime):
        """Test single comment prediction via SageMaker endpoint."""
        from src.api.sagemaker_client import SageMakerClient

        # Setup mock response
        mock_response = {
            "Body": MagicMock(
                read=lambda: json.dumps({
                    "predictions": {
                        "toxic": 0.8,
                        "severe_toxic": 0.1,
                        "obscene": 0.3,
                        "threat": 0.05,
                        "insult": 0.4,
                        "identity_hate": 0.02,
                    }
                }).encode()
            )
        }
        mock_sagemaker_runtime.invoke_endpoint.return_value = mock_response

        client = SageMakerClient(endpoint_name="test-endpoint")
        result = client.predict("This is a test comment")

        assert "toxic" in result
        assert result["toxic"] == 0.8
        mock_sagemaker_runtime.invoke_endpoint.assert_called_once()

    def test_predict_batch_comments(self, mock_sagemaker_runtime):
        """Test batch comment prediction via SageMaker endpoint."""
        from src.api.sagemaker_client import SageMakerClient

        # Setup mock response
        mock_response = {
            "Body": MagicMock(
                read=lambda: json.dumps({
                    "predictions": [
                        {"toxic": 0.8, "severe_toxic": 0.1, "obscene": 0.3, "threat": 0.05, "insult": 0.4, "identity_hate": 0.02},
                        {"toxic": 0.2, "severe_toxic": 0.0, "obscene": 0.1, "threat": 0.01, "insult": 0.1, "identity_hate": 0.01},
                    ]
                }).encode()
            )
        }
        mock_sagemaker_runtime.invoke_endpoint.return_value = mock_response

        client = SageMakerClient(endpoint_name="test-endpoint")
        result = client.predict_batch(["Comment 1", "Comment 2"])

        assert len(result) == 2
        assert result[0]["toxic"] == 0.8
        assert result[1]["toxic"] == 0.2


class TestSageMakerTraining:
    """Tests for SageMaker training script."""

    def test_text_preprocessor(self):
        """Test the text preprocessor."""
        import sys
        sys.path.insert(0, "sagemaker/training")
        
        # Import directly from the training script module
        from sagemaker.training.train import TextPreprocessor

        preprocessor = TextPreprocessor()

        # Test basic preprocessing
        result = preprocessor.preprocess_text("Hello, World!")
        assert result == "hello world"

        # Test URL removal
        result = preprocessor.preprocess_text("Check https://example.com now")
        assert "https" not in result

        # Test HTML removal
        result = preprocessor.preprocess_text("<p>Hello</p>")
        assert "<p>" not in result

    def test_toxic_classifier_params(self):
        """Test ToxicCommentClassifier parameter handling."""
        import sys
        sys.path.insert(0, "sagemaker/training")
        
        from sagemaker.training.train import ToxicCommentClassifier

        # Test baseline model params
        classifier = ToxicCommentClassifier(model_type="baseline")
        params = classifier.get_params()

        assert params["model_type"] == "baseline"
        assert "tfidf_max_features" in params

        # Test improved model params
        classifier = ToxicCommentClassifier(model_type="improved")
        params = classifier.get_params()

        assert params["model_type"] == "improved"

    def test_compute_metrics(self):
        """Test metrics computation."""
        import numpy as np
        import sys
        sys.path.insert(0, "sagemaker/training")
        
        from sagemaker.training.train import compute_metrics

        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[1, 0, 1], [0, 0, 0]])
        y_proba = np.array([[0.9, 0.1, 0.8], [0.2, 0.4, 0.3]])

        metrics = compute_metrics(y_true, y_pred, y_proba)

        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "roc_auc_macro" in metrics


class TestInferenceHandler:
    """Tests for SageMaker inference handler."""

    def test_input_parsing_single(self):
        """Test parsing single comment input."""
        import sys
        sys.path.insert(0, "sagemaker/inference")
        
        from sagemaker.inference.inference import input_fn

        body = json.dumps({"comment": "Test comment"})
        result = input_fn(body, "application/json")

        assert result["texts"] == ["Test comment"]
        assert result["batch"] is False

    def test_input_parsing_batch(self):
        """Test parsing batch comments input."""
        import sys
        sys.path.insert(0, "sagemaker/inference")
        
        from sagemaker.inference.inference import input_fn

        body = json.dumps({"comments": ["Comment 1", "Comment 2"]})
        result = input_fn(body, "application/json")

        assert result["texts"] == ["Comment 1", "Comment 2"]
        assert result["batch"] is True

    def test_input_parsing_invalid(self):
        """Test parsing invalid input."""
        import sys
        sys.path.insert(0, "sagemaker/inference")
        
        from sagemaker.inference.inference import input_fn

        body = json.dumps({})
        with pytest.raises(ValueError):
            input_fn(body, "application/json")


@pytest.mark.integration
class TestSageMakerEndpoint:
    """
    Integration tests that require a running SageMaker endpoint.
    
    These tests are skipped unless SAGEMAKER_ENDPOINT_NAME is set.
    """

    @pytest.fixture
    def endpoint_name(self):
        name = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
        if not name:
            pytest.skip("SAGEMAKER_ENDPOINT_NAME not set")
        return name

    def test_endpoint_health(self, endpoint_name):
        """Test that the endpoint is healthy."""
        import boto3

        sagemaker = boto3.client("sagemaker")
        response = sagemaker.describe_endpoint(EndpointName=endpoint_name)

        assert response["EndpointStatus"] == "InService"

    def test_endpoint_invocation(self, endpoint_name):
        """Test invoking the endpoint with a sample comment."""
        import boto3

        runtime = boto3.client("sagemaker-runtime")
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps({"comment": "This is a test comment"}),
        )

        result = json.loads(response["Body"].read().decode())
        assert "predictions" in result
