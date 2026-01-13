import argparse
import json
import os
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import boto3
import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
SM_CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
SM_OUTPUT_DATA_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")


TARGET_COLUMNS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)


class TextPreprocessor:
    def __init__(self):
        import re
        import unicodedata

        self.re = re
        self.unicodedata = unicodedata
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
            for c in self.unicodedata.normalize("NFD", result)
            if self.unicodedata.category(c) != "Mn"
        )
        result = self._newline_pattern.sub(" ", result)
        result = self._special_char_pattern.sub(" ", result)
        result = self._whitespace_pattern.sub(" ", result).strip()

        return result if len(result) >= 1 else ""


class ToxicCommentClassifier:
    def __init__(
        self,
        model_type: str = "baseline",
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        C: float = 1.0,
        max_iter: int = 1000,
    ):
        self.model_type = model_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.C = C
        self.max_iter = max_iter
        self._pipeline = None

    def _build_baseline_pipeline(self) -> Pipeline:
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            analyzer="word",
            token_pattern=r"\w{1,}",
            sublinear_tf=True,
        )

        classifier = OneVsRestClassifier(
            LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                solver="lbfgs",
                class_weight="balanced",
                n_jobs=-1,
            )
        )

        return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])

    def _build_improved_pipeline(self) -> Pipeline:
        vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.9,
            analyzer="word",
            token_pattern=r"\w{1,}",
            sublinear_tf=True,
        )

        # Use stronger regularization and different solver
        classifier = OneVsRestClassifier(
            LogisticRegression(
                C=0.5,  # Stronger regularization
                max_iter=2000,
                solver="saga",  # Better for larger datasets
                class_weight="balanced",
                n_jobs=-1,
            )
        )

        return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])

    def fit(self, X: List[str], y: np.ndarray) -> "ToxicCommentClassifier":
        if self.model_type == "improved":
            self._pipeline = self._build_improved_pipeline()
        else:
            self._pipeline = self._build_baseline_pipeline()

        self._pipeline.fit(X, y)
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        return self._pipeline.predict(X)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        return self._pipeline.predict_proba(X)

    def get_params(self) -> Dict:
        return {
            "model_type": self.model_type,
            "tfidf_max_features": self.max_features,
            "tfidf_ngram_range": str(self.ngram_range),
            "tfidf_min_df": self.min_df,
            "tfidf_max_df": self.max_df,
            "lr_C": self.C,
            "lr_max_iter": self.max_iter,
        }


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["precision_macro"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["recall_macro"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    try:
        metrics["roc_auc_macro"] = roc_auc_score(y_true, y_proba, average="macro")
        metrics["roc_auc_micro"] = roc_auc_score(y_true, y_proba, average="micro")
    except ValueError:
        metrics["roc_auc_macro"] = 0.0
        metrics["roc_auc_micro"] = 0.0

    for i, label in enumerate(TARGET_COLUMNS):
        metrics[f"f1_{label}"] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        try:
            metrics[f"roc_auc_{label}"] = roc_auc_score(y_true[:, i], y_proba[:, i])
        except ValueError:
            metrics[f"roc_auc_{label}"] = 0.0

    return metrics


def load_training_data(input_path: str) -> pd.DataFrame:
    """Load training data from input channel."""
    input_path = Path(input_path)

    all_dfs = []

    # Load all CSV files in the input directory
    for csv_file in input_path.glob("*.csv"):
        print(f"Loading {csv_file}")
        df = pd.read_csv(csv_file)
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"No CSV files found in {input_path}")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} total samples")

    return combined_df


def save_model_onnx(model: ToxicCommentClassifier, output_path: Path) -> str:
    """Export model to ONNX format."""
    import onnx
    import pickle
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import StringTensorType

    onnx_path = output_path / "model.onnx"
    initial_type = [("input", StringTensorType([None]))]

    onnx_model = to_onnx(
        model._pipeline,
        initial_types=initial_type,
        target_opset={"": 15, "ai.onnx.ml": 2},
        options={id(model._pipeline.named_steps["classifier"]): {"zipmap": False}},
    )

    onnx.save_model(onnx_model, str(onnx_path))
    print(f"Saved ONNX model to {onnx_path}")

    return str(onnx_path)


def save_model_sklearn(model: ToxicCommentClassifier, output_path: Path) -> str:
    """Save sklearn model as pickle."""
    import pickle

    pkl_path = output_path / "model.pkl"

    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "pipeline": model._pipeline,
                "params": model.get_params(),
            },
            f,
        )

    print(f"Saved sklearn model to {pkl_path}")
    return str(pkl_path)


def create_model_tarball(model_dir: Path, output_path: Path) -> str:
    """Create model.tar.gz for SageMaker."""
    tarball_path = output_path / "model.tar.gz"

    with tarfile.open(tarball_path, "w:gz") as tar:
        for file_path in model_dir.iterdir():
            tar.add(file_path, arcname=file_path.name)

    print(f"Created model tarball at {tarball_path}")
    return str(tarball_path)


def register_model_to_registry(
    model_data_url: str,
    model_package_group_name: str,
    metrics: Dict[str, float],
    inference_image_uri: str,
    sagemaker_client=None,
) -> str:
    """Register model to SageMaker Model Registry."""
    if sagemaker_client is None:
        sagemaker_client = boto3.client("sagemaker")

    model_metrics = {
        "ModelQuality": {
            "Statistics": {
                "ContentType": "application/json",
                "S3Uri": "s3://placeholder",  # Would be set by actual metrics
            }
        }
    }

    response = sagemaker_client.create_model_package(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageDescription=f"Model trained at {datetime.now().isoformat()}",
        InferenceSpecification={
            "Containers": [
                {
                    "Image": inference_image_uri,
                    "ModelDataUrl": model_data_url,
                }
            ],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        ModelApprovalStatus="PendingManualApproval",
        CustomerMetadataProperties={
            "roc_auc_macro": str(metrics.get("roc_auc_macro", 0)),
            "f1_macro": str(metrics.get("f1_macro", 0)),
            "accuracy": str(metrics.get("accuracy", 0)),
        },
    )

    model_package_arn = response["ModelPackageArn"]
    print(f"Registered model package: {model_package_arn}")

    return model_package_arn


def train(args):
    """Main training function."""
    print("=" * 50)
    print(f"Starting training with model_type: {args.model_type}")
    print("=" * 50)

    # Setup MLflow
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        print(f"MLflow tracking URI: {args.mlflow_tracking_uri}")

    # Load data
    print("\nLoading training data...")
    df = load_training_data(SM_CHANNEL_TRAINING)

    # Preprocess
    print("\nPreprocessing text...")
    preprocessor = TextPreprocessor()
    df["processed_text"] = df["comment_text"].apply(preprocessor.preprocess_text)

    # Filter empty texts
    df = df[df["processed_text"].str.len() > 0]
    print(f"Samples after filtering: {len(df)}")

    # Prepare features and labels
    X = df["processed_text"].tolist()
    y = df[list(TARGET_COLUMNS)].values

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Create and train model
    model = ToxicCommentClassifier(
        model_type=args.model_type,
        max_features=args.max_features,
        C=args.lr_c,
        max_iter=args.max_iter,
    )

    run_name = f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        params = model.get_params()
        params["train_samples"] = len(X_train)
        params["val_samples"] = len(X_val)
        params["test_size"] = args.test_size
        mlflow.log_params(params)

        # Train model
        print("\nTraining model...")
        model.fit(X_train, y_train)

        # Evaluate
        print("\nEvaluating model...")
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        metrics = compute_metrics(y_val, y_pred, y_proba)

        # Log metrics
        mlflow.log_metrics(metrics)

        print("\nMetrics:")
        for name, value in sorted(metrics.items()):
            print(f"  {name}: {value:.4f}")

        # Save models
        model_dir = Path(SM_MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)

        save_model_sklearn(model, model_dir)
        save_model_onnx(model, model_dir)

        # Copy inference code to model directory
        code_dir = model_dir / "code"
        code_dir.mkdir(exist_ok=True)

        # Create inference script
        inference_script = code_dir / "inference.py"
        inference_script.write_text(INFERENCE_SCRIPT_CONTENT)

        # Log artifacts to MLflow
        mlflow.log_artifacts(str(model_dir))

        # Save metrics for SageMaker
        metrics_path = Path(SM_OUTPUT_DATA_DIR) / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\nTraining complete!")
        print(f"Model saved to: {SM_MODEL_DIR}")
        print(f"Metrics saved to: {metrics_path}")
        print(f"ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")


# Inference script content to be bundled with the model
INFERENCE_SCRIPT_CONTENT = '''
"""SageMaker inference handler for toxic comment classification."""

import json
import os
import pickle
from pathlib import Path

import numpy as np


def model_fn(model_dir):
    """Load the model from the model directory."""
    model_path = Path(model_dir) / "model.pkl"
    
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    
    return data["pipeline"]


def input_fn(request_body, content_type):
    """Parse input data."""
    if content_type == "application/json":
        data = json.loads(request_body)
        return data.get("comment", data.get("comments", []))
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """Make predictions."""
    if isinstance(input_data, str):
        input_data = [input_data]
    
    proba = model.predict_proba(input_data)
    
    return proba


def output_fn(prediction, accept):
    """Format prediction output."""
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    
    if len(prediction.shape) == 1:
        prediction = prediction.reshape(1, -1)
    
    results = []
    for probs in prediction:
        result = {label: float(prob) for label, prob in zip(labels, probs)}
        results.append(result)
    
    if len(results) == 1:
        output = {"predictions": results[0]}
    else:
        output = {"predictions": results}
    
    return json.dumps(output)
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default="baseline",
        choices=["baseline", "improved"],
        help="Model type to train",
    )
    parser.add_argument(
        "--max-features", type=int, default=10000, help="Max TF-IDF features"
    )
    parser.add_argument(
        "--lr-c", type=float, default=1.0, help="Logistic regression C parameter"
    )
    parser.add_argument(
        "--max-iter", type=int, default=1000, help="Max iterations for LogReg"
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    # MLflow arguments
    parser.add_argument(
        "--mlflow-tracking-uri", type=str, default="", help="MLflow tracking URI"
    )

    # SageMaker environment
    parser.add_argument(
        "--model-package-group",
        type=str,
        default="",
        help="SageMaker Model Package Group name",
    )
    parser.add_argument(
        "--inference-image-uri", type=str, default="", help="Inference container URI"
    )

    args = parser.parse_args()

    train(args)
