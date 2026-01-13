"""
SageMaker Client for invoking the toxic comment classification endpoint.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Union

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class SageMakerClient:
    """Client for invoking SageMaker endpoint for predictions."""

    def __init__(
        self,
        endpoint_name: str,
        region: Optional[str] = None,
    ):
        self.endpoint_name = endpoint_name
        self.region = region
        self._client = boto3.client(
            "sagemaker-runtime",
            region_name=region,
        )
        self._is_available = None

    @property
    def is_available(self) -> bool:
        """Check if the endpoint is available."""
        if self._is_available is None:
            self._is_available = self._check_endpoint()
        return self._is_available

    def _check_endpoint(self) -> bool:
        """Check if the endpoint exists and is in service."""
        try:
            sagemaker = boto3.client("sagemaker", region_name=self.region)
            response = sagemaker.describe_endpoint(EndpointName=self.endpoint_name)
            status = response.get("EndpointStatus", "")
            return status == "InService"
        except ClientError as e:
            logger.warning(f"Endpoint check failed: {e}")
            return False

    def predict(self, text: str) -> Dict[str, float]:
        """
        Make a prediction for a single comment.
        
        Args:
            text: The comment text to classify
            
        Returns:
            Dictionary with prediction scores for each toxicity category
        """
        start_time = time.time()

        try:
            response = self._client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps({"comment": text}),
            )

            result = json.loads(response["Body"].read().decode())
            predictions = result.get("predictions", {})

            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"Prediction latency: {latency_ms:.2f}ms")

            return predictions

        except ClientError as e:
            logger.error(f"SageMaker invocation failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Make predictions for a batch of comments.
        
        Args:
            texts: List of comment texts to classify
            
        Returns:
            List of dictionaries with prediction scores
        """
        start_time = time.time()

        try:
            response = self._client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps({"comments": texts}),
            )

            result = json.loads(response["Body"].read().decode())
            predictions = result.get("predictions", [])

            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"Batch prediction latency: {latency_ms:.2f}ms for {len(texts)} texts")

            return predictions

        except ClientError as e:
            logger.error(f"SageMaker batch invocation failed: {e}")
            raise RuntimeError(f"Batch prediction failed: {e}")


class SageMakerClientFactory:
    """Factory for creating SageMaker clients with fallback support."""

    _instance: Optional[SageMakerClient] = None

    @classmethod
    def get_client(
        cls,
        endpoint_name: str,
        region: Optional[str] = None,
    ) -> SageMakerClient:
        """Get or create a SageMaker client instance."""
        if cls._instance is None:
            cls._instance = SageMakerClient(endpoint_name, region)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the cached client instance."""
        cls._instance = None
