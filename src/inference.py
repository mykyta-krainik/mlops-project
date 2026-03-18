import json
import os
import sys
from pathlib import Path

import boto3
import numpy as np

sys.path.insert(0, "/app")

from src.api.moderation import ModerationDecider as ModerationEngine
from src.config import config
from src.data.preprocessing import TextPreprocessor
from src.models.baseline import ToxicCommentClassifier

_cloudwatch = boto3.client("cloudwatch", region_name=config.aws.region)
_NAMESPACE = "mlops-toxic/serving"
_ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "unknown")


def model_fn(model_dir: str) -> dict:
    model_path = Path(model_dir) / "model.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"model.onnx not found in {model_dir}")

    classifier = ToxicCommentClassifier()
    classifier.load_onnx(model_path)

    preprocessor = TextPreprocessor()
    moderation = ModerationEngine()

    print(f"Loaded ONNX model from {model_path}")
    return {
        "classifier": classifier,
        "preprocessor": preprocessor,
        "moderation": moderation,
    }


def input_fn(request_body: str, content_type: str = "application/json") -> list[str]:
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    data = json.loads(request_body)

    if "comment" in data:
        return [data["comment"]]
    elif "comments" in data:
        comments = data["comments"]
        if not isinstance(comments, list):
            raise ValueError("'comments' must be a list")
        if len(comments) > 100:
            raise ValueError("Batch size exceeds maximum of 100")
        return comments
    else:
        raise ValueError("Request must contain 'comment' (str) or 'comments' (list)")


def predict_fn(comments: list[str], model_dict: dict) -> list[dict]:
    classifier: ToxicCommentClassifier = model_dict["classifier"]
    preprocessor: TextPreprocessor = model_dict["preprocessor"]
    moderation: ModerationEngine = model_dict["moderation"]

    cleaned = [preprocessor.preprocess_text(c) for c in comments]

    probas = classifier.predict_proba(cleaned)  # (N, 6)

    results = []
    for i, comment in enumerate(comments):
        proba_row = probas[i]
        predictions = {
            label: float(proba_row[j]) for j, label in enumerate(config.model.target_columns)
        }
        action = moderation.decide(predictions)
        is_toxic = action.value != "ALLOW"

        results.append(
            {
                "comment": comment,
                "predictions": predictions,
                "is_toxic": is_toxic,
                "moderation_action": action.value,
                "model_version": os.environ.get("MODEL_VERSION", "sagemaker"),
            }
        )

    try:
        toxic_rate = sum(1 for r in results if r["is_toxic"]) / len(results)
        mean_max_prob = float(np.mean([max(r["predictions"].values()) for r in results]))
        _cloudwatch.put_metric_data(
            Namespace=_NAMESPACE,
            MetricData=[
                {
                    "MetricName": "ToxicPredictionRate",
                    "Value": toxic_rate,
                    "Unit": "None",
                    "Dimensions": [{"Name": "EndpointName", "Value": _ENDPOINT_NAME}],
                },
                {
                    "MetricName": "MeanMaxProbability",
                    "Value": mean_max_prob,
                    "Unit": "None",
                    "Dimensions": [{"Name": "EndpointName", "Value": _ENDPOINT_NAME}],
                },
                {
                    "MetricName": "BatchSize",
                    "Value": float(len(results)),
                    "Unit": "Count",
                    "Dimensions": [{"Name": "EndpointName", "Value": _ENDPOINT_NAME}],
                },
            ],
        )
    except Exception:
        pass

    return results


def output_fn(predictions: list[dict], accept: str = "application/json") -> tuple[str, str]:
    if accept not in ("application/json", "*/*"):
        raise ValueError(f"Unsupported accept type: {accept}")

    if len(predictions) == 1:
        body = json.dumps(predictions[0])
    else:
        body = json.dumps(predictions)

    return body, "application/json"
