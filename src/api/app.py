import logging
import os
import tempfile
from typing import Optional

from flask import Flask, jsonify, request

from src.api.moderation import moderation_decider
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictionResult,
)
from src.config import config
from src.data.preprocessing import TextPreprocessor
from src.data.storage import MinioStorage
from src.models.baseline import ToxicCommentClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self._model: Optional[ToxicCommentClassifier] = None
        self._model_source: Optional[str] = None
        self._preprocessor = TextPreprocessor()
        self._databricks_client: Optional[object] = None
        
        # Initialize Databricks client if enabled
        if config.serving.use_databricks:
            try:
                from src.api.databricks_serving import DatabricksServingClient
                self._databricks_client = DatabricksServingClient()
                logger.info("Databricks serving client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Databricks client: {e}")
                self._databricks_client = None

    @property
    def model(self) -> Optional[ToxicCommentClassifier]:
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._model.is_trained

    @property
    def is_databricks_available(self) -> bool:
        """Check if Databricks serving is available."""
        if not config.serving.use_databricks:
            return False
        if self._databricks_client is None:
            return False
        try:
            return self._databricks_client.health_check()
        except:
            return False

    def load_from_minio(self, bucket: str, object_name: str) -> bool:
        try:
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
        """
        Make prediction using Databricks serving or local model.
        
        Tries Databricks first if enabled, falls back to local model.
        """
        # Try Databricks serving first
        if config.serving.use_databricks and self._databricks_client:
            try:
                processed_text = self._preprocessor.preprocess_text(text)
                predictions = self._databricks_client.predict(processed_text)
                logger.info("Prediction made via Databricks serving")
                return predictions
            except Exception as e:
                logger.warning(f"Databricks serving failed, falling back to local: {e}")
        
        # Fallback to local model
        if not self.is_loaded:
            raise RuntimeError("Model not loaded and Databricks serving unavailable")

        processed_text = self._preprocessor.preprocess_text(text)
        predictions = self._model.predict_single(processed_text)
        logger.info("Prediction made via local model")
        
        return predictions


model_manager = ModelManager()


def create_app() -> Flask:
    app = Flask(__name__)

    @app.before_request
    def ensure_model_loaded():
        if not model_manager.is_loaded:
            try:
                model_manager.load_from_minio(
                    config.minio.models_bucket,
                    "latest/model.onnx",
                )
            except Exception as e:
                logger.warning(f"Could not load model from Minio: {e}")

    @app.route("/health", methods=["GET"])
    def health_check():
        response = HealthResponse(
            status="healthy" if model_manager.is_loaded else "degraded",
            model_loaded=model_manager.is_loaded,
            version=config.api.model_version,
        )
        return jsonify(response.model_dump())

    @app.route("/model/info", methods=["GET"])
    def model_info():
        response = ModelInfoResponse(
            version=config.api.model_version,
            model_type="TF-IDF + Logistic Regression (ONNX)",
            target_labels=list(config.model.target_columns),
            loaded=model_manager.is_loaded,
            source=model_manager._model_source,
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

            return jsonify(result.model_dump())

        except Exception as e:
            logger.error(f"Prediction error: {e}")
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

        try:
            data = request.get_json()
            req = BatchPredictRequest(**data)

            results = []
            for comment in req.comments:
                predictions = model_manager.predict(comment)
                action = moderation_decider.decide(predictions)
                is_toxic = moderation_decider.is_toxic(predictions)

                result = PredictionResult(
                    comment=comment,
                    predictions=predictions,
                    is_toxic=is_toxic,
                    moderation_action=action,
                    model_version=config.api.model_version,
                )
                results.append(result)

            response = BatchPredictionResponse(
                results=[r.model_dump() for r in results],
                total=len(results),
            )

            return jsonify(response.model_dump())

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            return jsonify(
                ErrorResponse(error="Batch prediction failed", detail=str(e)).model_dump()
            ), 400

    @app.route("/model/reload", methods=["POST"])
    def reload_model():
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

    @app.route("/serving/status", methods=["GET"])
    def serving_status():
        """Check serving configuration and availability."""
        databricks_available = model_manager.is_databricks_available
        local_available = model_manager.is_loaded
        
        status = {
            "databricks_enabled": config.serving.use_databricks,
            "databricks_available": databricks_available,
            "local_model_loaded": local_available,
            "active_backend": "databricks" if databricks_available else "local" if local_available else "none",
        }
        
        return jsonify(status)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=config.api.host,
        port=config.api.port,
        debug=True,
    )
