"""
SageMaker Inference Handler for Toxic Comment Classification.

This module provides the inference logic for the SageMaker endpoint.
Supports both sklearn pickle and ONNX model formats.
"""

import json
import logging
import os
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Target labels
TARGET_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Global model reference
model = None
model_type = None  # "sklearn" or "onnx"
onnx_session = None


class TextPreprocessor:
    """Text preprocessing for toxic comment classification."""

    def __init__(self):
        self._url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self._html_pattern = re.compile(r"<[^>]+>")
        self._newline_pattern = re.compile(r"\n+")
        self._whitespace_pattern = re.compile(r"\s+")
        self._special_char_pattern = re.compile(r"[^\w\s]")

    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        result = text
        result = self._html_pattern.sub(" ", result)
        result = self._url_pattern.sub(" ", result)
        result = result.lower()
        result = "".join(
            c
            for c in unicodedata.normalize("NFD", result)
            if unicodedata.category(c) != "Mn"
        )
        result = self._newline_pattern.sub(" ", result)
        result = self._special_char_pattern.sub(" ", result)
        result = self._whitespace_pattern.sub(" ", result).strip()

        return result if len(result) >= 1 else ""

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        return [self.preprocess_text(text) for text in texts]


preprocessor = TextPreprocessor()


def model_fn(model_dir: str) -> Any:
    """
    Load the model from the model directory.
    
    SageMaker calls this function to load the model when the endpoint starts.
    The model can be either a sklearn pickle or ONNX format.
    """
    global model, model_type, onnx_session

    model_dir = Path(model_dir)
    logger.info(f"Loading model from {model_dir}")
    logger.info(f"Contents: {list(model_dir.iterdir())}")

    # Try ONNX first
    onnx_path = model_dir / "model.onnx"
    if onnx_path.exists():
        try:
            import onnxruntime as ort

            onnx_session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            model_type = "onnx"
            logger.info("Loaded ONNX model successfully")
            return onnx_session
        except Exception as e:
            logger.warning(f"Failed to load ONNX model: {e}")

    # Fall back to sklearn pickle
    pkl_path = model_dir / "model.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        model = data["pipeline"]
        model_type = "sklearn"
        logger.info("Loaded sklearn model successfully")
        return model

    raise ValueError(f"No valid model found in {model_dir}")


def input_fn(request_body: str, content_type: str) -> Dict[str, Any]:
    """
    Parse the input request body.
    
    Supports JSON input with either:
    - {"comment": "text"} for single prediction
    - {"comments": ["text1", "text2"]} for batch prediction
    """
    logger.info(f"Received request with content_type: {content_type}")

    if content_type == "application/json":
        try:
            data = json.loads(request_body)
            
            # Handle single comment
            if "comment" in data:
                return {"texts": [data["comment"]], "batch": False}
            
            # Handle batch of comments
            if "comments" in data:
                return {"texts": data["comments"], "batch": True}
            
            # Handle raw text list
            if isinstance(data, list):
                return {"texts": data, "batch": True}
            
            raise ValueError("Request must contain 'comment' or 'comments' field")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    elif content_type == "text/plain":
        return {"texts": [request_body], "batch": False}

    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: Dict[str, Any], loaded_model: Any) -> Dict[str, Any]:
    """
    Make predictions using the loaded model.
    """
    global model_type, onnx_session

    texts = input_data["texts"]
    is_batch = input_data["batch"]

    # Preprocess texts
    processed_texts = preprocessor.preprocess_batch(texts)
    logger.info(f"Processing {len(processed_texts)} texts")

    # Make predictions based on model type
    if model_type == "onnx":
        input_name = onnx_session.get_inputs()[0].name
        inputs = {input_name: np.array(processed_texts)}
        outputs = onnx_session.run(None, inputs)
        
        # ONNX outputs: [predictions, probabilities]
        probabilities = outputs[1] if len(outputs) > 1 else outputs[0]
        
    elif model_type == "sklearn":
        probabilities = loaded_model.predict_proba(processed_texts)
        
    else:
        raise RuntimeError(f"Unknown model type: {model_type}")

    # Format results
    results = []
    for i, probs in enumerate(probabilities):
        prediction = {
            "predictions": {
                label: float(prob) for label, prob in zip(TARGET_LABELS, probs)
            },
            "is_toxic": bool(any(p > 0.5 for p in probs)),
            "original_text": texts[i],
        }
        results.append(prediction)

    return {"results": results, "batch": is_batch}


def output_fn(prediction: Dict[str, Any], accept: str) -> str:
    """
    Format the prediction output.
    """
    logger.info(f"Formatting output with accept: {accept}")

    if accept == "application/json" or accept == "*/*":
        results = prediction["results"]
        is_batch = prediction["batch"]

        if is_batch:
            output = {
                "predictions": [r["predictions"] for r in results],
                "is_toxic": [r["is_toxic"] for r in results],
            }
        else:
            result = results[0]
            output = {
                "predictions": result["predictions"],
                "is_toxic": result["is_toxic"],
            }

        return json.dumps(output)

    raise ValueError(f"Unsupported accept type: {accept}")


# For local testing
if __name__ == "__main__":
    import sys

    # Test with sample data
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/opt/ml/model"
    
    # Load model
    loaded_model = model_fn(model_dir)
    
    # Test prediction
    test_input = {"comment": "This is a test comment"}
    parsed_input = input_fn(json.dumps(test_input), "application/json")
    prediction = predict_fn(parsed_input, loaded_model)
    output = output_fn(prediction, "application/json")
    
    print(f"Output: {output}")
