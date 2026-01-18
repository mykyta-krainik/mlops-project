"""
Databricks Model Serving Client.

This module provides a client for making predictions using Databricks Model Serving endpoints.
"""
import logging
from typing import Dict, List, Optional

import requests

from src.config import config

logger = logging.getLogger(__name__)


class DatabricksServingClient:
    """Client for Databricks Model Serving."""
    
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """
        Initialize Databricks Serving Client.
        
        Args:
            endpoint_url: Serving endpoint URL (defaults to config)
            token: Databricks token (defaults to config)
        """
        self.endpoint_url = endpoint_url or config.databricks.serving_endpoint
        self.token = token or config.databricks.token
        
        if not self.endpoint_url:
            raise ValueError("Databricks serving endpoint URL not configured")
        if not self.token:
            raise ValueError("Databricks token not configured")
    
    def predict(self, text: str) -> Dict[str, float]:
        """
        Make prediction for a single text.
        
        Args:
            text: Input text to classify
        
        Returns:
            Dictionary mapping label names to probabilities
        """
        return self.predict_batch([text])[0]
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Make predictions for multiple texts.
        
        Args:
            texts: List of input texts
        
        Returns:
            List of dictionaries mapping label names to probabilities
        """
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "inputs": texts,
        }
        
        try:
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Parse response
            # Databricks serving returns predictions in format:
            # {"predictions": [[prob1, prob2, ...], ...]}
            predictions = result.get("predictions", [])
            
            # Convert to label dictionaries
            target_labels = config.model.target_columns
            results = []
            
            for pred in predictions:
                label_dict = {
                    label: float(prob)
                    for label, prob in zip(target_labels, pred)
                }
                results.append(label_dict)
            
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Databricks serving request failed: {e}")
            raise RuntimeError(f"Databricks serving error: {e}")
    
    def health_check(self) -> bool:
        """
        Check if serving endpoint is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try a simple prediction
            self.predict("test")
            return True
        except Exception as e:
            logger.warning(f"Databricks serving health check failed: {e}")
            return False
