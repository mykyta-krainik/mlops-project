"""
Flask API for Toxic Comment Classification.

Supports both local model loading (MinIO/file) and SageMaker endpoint invocation.
The mode is controlled by the USE_SAGEMAKER environment variable.
"""

import logging
import os
import tempfile
import time
from typing import Optional

from flask import Flask, jsonify, request

from src.api.moderation import moderation_decider
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModerationAction,
    ModelInfoResponse,
    PredictRequest,
    PredictionResult,
)
from src.config import config
from src.data.preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature flag for SageMaker vs local model
USE_SAGEMAKER = os.environ.get("USE_SAGEMAKER", "false").lower() == "true"


class ModelManager:
    """Manages local model loading from MinIO or file."""

    def __init__(self):
        self._model = None
        self._model_source: Optional[str] = None
        self._preprocessor = TextPreprocessor()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._model.is_trained

    def load_from_minio(self, bucket: str, object_name: str) -> bool:
        try:
            from src.data.storage import MinioStorage
            from src.models.baseline import ToxicCommentClassifier

            storage = MinioStorage()

            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
                storage.download_file(bucket, object_name, tmp.name)
                self._model = ToxicCommentClassifier()
                self._model.load_onnx(tmp.name)
                self._model_source = f"minio://{bucket}/{object_name}"

            os.unlink(tmp.name)

            logger.info(f"Model loaded from {self._model_source}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model from Minio: {e}")
            return False

    def load_from_file(self, path: str, format: str = "onnx") -> bool:
        try:
            from src.models.baseline import ToxicCommentClassifier

            self._model = ToxicCommentClassifier()

            if format == "onnx":
                self._model.load_onnx(path)
            else:
                self._model.load_sklearn(path)

            self._model_source = f"file://{path}"
            logger.info(f"Model loaded from {self._model_source}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model from file: {e}")
            return False

    def predict(self, text: str) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        processed_text = self._preprocessor.preprocess_text(text)
        predictions = self._model.predict_single(processed_text)

        return predictions


class SageMakerModelManager:
    """Manages predictions via SageMaker endpoint."""

    def __init__(self, endpoint_name: str, region: Optional[str] = None):
        self.endpoint_name = endpoint_name
        self.region = region
        self._client = None
        self._is_available = None
        self._preprocessor = TextPreprocessor()

    @property
    def is_loaded(self) -> bool:
        """Check if SageMaker endpoint is available."""
        if self._is_available is None:
            self._is_available = self._check_endpoint()
        return self._is_available

    def _check_endpoint(self) -> bool:
        try:
            import boto3
            from botocore.exceptions import ClientError

            sagemaker = boto3.client("sagemaker", region_name=self.region)
            response = sagemaker.describe_endpoint(EndpointName=self.endpoint_name)
            return response.get("EndpointStatus") == "InService"
        except Exception as e:
            logger.warning(f"Endpoint check failed: {e}")
            return False

    def _get_client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("sagemaker-runtime", region_name=self.region)
        return self._client

    def predict(self, text: str) -> dict:
        import json

        processed_text = self._preprocessor.preprocess_text(text)

        response = self._get_client().invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps({"comment": processed_text}),
        )

        result = json.loads(response["Body"].read().decode())
        return result.get("predictions", {})


# Initialize the appropriate model manager
if USE_SAGEMAKER:
    model_manager = SageMakerModelManager(
        endpoint_name=config.aws.sagemaker_endpoint_name,
        region=config.aws.region,
    )
    logger.info(f"Using SageMaker endpoint: {config.aws.sagemaker_endpoint_name}")
else:
    model_manager = ModelManager()
    logger.info("Using local model manager")


# Optional: CloudWatch metrics logger
cloudwatch_logger = None
try:
    if USE_SAGEMAKER or os.environ.get("ENABLE_CLOUDWATCH", "false").lower() == "true":
        from src.monitoring.cloudwatch_logger import CloudWatchMetricsLogger

        cloudwatch_logger = CloudWatchMetricsLogger(
            namespace=config.aws.cloudwatch_namespace,
            region=config.aws.region,
        )
        logger.info("CloudWatch metrics logging enabled")
except ImportError:
    logger.info("CloudWatch metrics logging not available")

# Optional: Review database for storing REVIEW actions
review_db = None
try:
    if os.environ.get("ENABLE_REVIEW_DB", "false").lower() == "true":
        from src.review.database import ReviewDatabase

        review_db = ReviewDatabase()
        logger.info("Review database enabled")
except ImportError:
    logger.info("Review database not available")


def create_app() -> Flask:
    app = Flask(__name__)

    @app.before_request
    def ensure_model_loaded():
        if not USE_SAGEMAKER and not model_manager.is_loaded:
            try:
                model_manager.load_from_minio(
                    config.minio.models_bucket,
                    "latest/model.onnx",
                )
            except Exception as e:
                logger.warning(f"Could not load model from Minio: {e}")

    @app.route("/health", methods=["GET"])
    def health_check():
        is_healthy = model_manager.is_loaded
        response = HealthResponse(
            status="healthy" if is_healthy else "degraded",
            model_loaded=is_healthy,
            version=config.api.model_version,
        )
        return jsonify(response.model_dump())

    @app.route("/model/info", methods=["GET"])
    def model_info():
        if USE_SAGEMAKER:
            source = f"sagemaker://{config.aws.sagemaker_endpoint_name}"
            model_type = "SageMaker Endpoint"
        else:
            source = getattr(model_manager, "_model_source", None)
            model_type = "TF-IDF + Logistic Regression (ONNX)"

        response = ModelInfoResponse(
            version=config.api.model_version,
            model_type=model_type,
            target_labels=list(config.model.target_columns),
            loaded=model_manager.is_loaded,
            source=source,
        )
        return jsonify(response.model_dump())

    @app.route("/predict", methods=["POST"])
    def predict():
        if not model_manager.is_loaded:
            return jsonify(
                ErrorResponse(
                    error="Model not loaded",
                    detail="The model is not available. Please try again later.",
                ).model_dump()
            ), 503

        start_time = time.time()

        try:
            data = request.get_json()
            req = PredictRequest(**data)

            predictions = model_manager.predict(req.comment)

            action = moderation_decider.decide(predictions)
            is_toxic = moderation_decider.is_toxic(predictions)

            result = PredictionResult(
                comment=req.comment,
                predictions=predictions,
                is_toxic=is_toxic,
                moderation_action=action,
                model_version=config.api.model_version,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Log metrics to CloudWatch
            if cloudwatch_logger:
                cloudwatch_logger.log_prediction(
                    latency_ms=latency_ms,
                    is_toxic=is_toxic,
                    is_error=False,
                )

            # Store REVIEW actions for moderator feedback
            if review_db and action == ModerationAction.REVIEW:
                try:
                    review_db.add_pending_review(
                        comment_text=req.comment,
                        predictions=predictions,
                        model_version=config.api.model_version,
                        source="api",
                    )
                except Exception as e:
                    logger.warning(f"Failed to store review: {e}")

            return jsonify(result.model_dump())

        except Exception as e:
            logger.error(f"Prediction error: {e}")

            if cloudwatch_logger:
                cloudwatch_logger.log_prediction(
                    latency_ms=(time.time() - start_time) * 1000,
                    is_toxic=False,
                    is_error=True,
                )

            return jsonify(
                ErrorResponse(error="Prediction failed", detail=str(e)).model_dump()
            ), 400

    @app.route("/predict/batch", methods=["POST"])
    def predict_batch():
        if not model_manager.is_loaded:
            return jsonify(
                ErrorResponse(
                    error="Model not loaded",
                    detail="The model is not available. Please try again later.",
                ).model_dump()
            ), 503

        start_time = time.time()

        try:
            data = request.get_json()
            req = BatchPredictRequest(**data)

            results = []
            toxic_count = 0

            for comment in req.comments:
                predictions = model_manager.predict(comment)
                action = moderation_decider.decide(predictions)
                is_toxic = moderation_decider.is_toxic(predictions)

                if is_toxic:
                    toxic_count += 1

                result = PredictionResult(
                    comment=comment,
                    predictions=predictions,
                    is_toxic=is_toxic,
                    moderation_action=action,
                    model_version=config.api.model_version,
                )
                results.append(result)

                # Store REVIEW actions
                if review_db and action == ModerationAction.REVIEW:
                    try:
                        review_db.add_pending_review(
                            comment_text=comment,
                            predictions=predictions,
                            model_version=config.api.model_version,
                            source="api_batch",
                        )
                    except Exception as e:
                        logger.warning(f"Failed to store review: {e}")

            response = BatchPredictionResponse(
                results=[r.model_dump() for r in results],
                total=len(results),
            )

            latency_ms = (time.time() - start_time) * 1000

            if cloudwatch_logger:
                cloudwatch_logger.log_batch_prediction(
                    latency_ms=latency_ms,
                    batch_size=len(req.comments),
                    toxic_count=toxic_count,
                    is_error=False,
                )

            return jsonify(response.model_dump())

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")

            if cloudwatch_logger:
                cloudwatch_logger.log_batch_prediction(
                    latency_ms=(time.time() - start_time) * 1000,
                    batch_size=0,
                    toxic_count=0,
                    is_error=True,
                )

            return jsonify(
                ErrorResponse(error="Batch prediction failed", detail=str(e)).model_dump()
            ), 400

    @app.route("/model/reload", methods=["POST"])
    def reload_model():
        if USE_SAGEMAKER:
            # For SageMaker, just refresh the endpoint status check
            model_manager._is_available = None
            is_available = model_manager.is_loaded
            return jsonify({
                "status": "success" if is_available else "error",
                "message": "Endpoint status refreshed",
                "available": is_available,
            })

        try:
            success = model_manager.load_from_minio(
                config.minio.models_bucket,
                "latest/model.onnx",
            )

            if success:
                return jsonify({"status": "success", "message": "Model reloaded"})
            else:
                return jsonify(
                    ErrorResponse(
                        error="Reload failed",
                        detail="Could not load model from storage",
                    ).model_dump()
                ), 500

        except Exception as e:
            logger.error(f"Model reload error: {e}")
            return jsonify(
                ErrorResponse(error="Reload failed", detail=str(e)).model_dump()
            ), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=config.api.host,
        port=config.api.port,
        debug=True,
    )
