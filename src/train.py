"""
SageMaker TrainingStep entry point.

Used for both baseline and improved model variants — hyperparameters are passed
as CLI arguments so the same script can be invoked with different settings.

SageMaker mounts data at:
  /opt/ml/input/data/train/train.csv
  /opt/ml/input/data/validation/validation.csv

SageMaker expects model artifacts at:
  /opt/ml/model/            ← auto-uploaded to S3 after job completes

SageMaker expects output (non-model) data at:
  /opt/ml/output/data/      ← written here so evaluate.py can read metrics.json
"""

import argparse
import contextlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.preprocessing import TextPreprocessor
from src.models.baseline import ToxicCommentClassifier


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    metrics: dict = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        metrics["roc_auc_macro"] = roc_auc_score(y_true, y_proba, average="macro")
        metrics["roc_auc_micro"] = roc_auc_score(y_true, y_proba, average="micro")
    except ValueError:
        metrics["roc_auc_macro"] = 0.0
        metrics["roc_auc_micro"] = 0.0

    for i, label in enumerate(config.model.target_columns):
        metrics[f"f1_{label}"] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        metrics[f"precision_{label}"] = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        metrics[f"recall_{label}"] = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        try:
            metrics[f"roc_auc_{label}"] = roc_auc_score(y_true[:, i], y_proba[:, i])
        except ValueError:
            metrics[f"roc_auc_{label}"] = 0.0

    return metrics


def run_train(
    train_uri: str,
    val_uri: str,
    model_name: str,
    models_bucket: str,
    run_prefix: str,
    max_features: int,
    ngram_min: int,
    ngram_max: int,
    C: float,
    max_iter: int,
) -> dict:
    """Download train/val data, train model, save ONNX to S3. Returns metrics + model URI."""
    import tempfile
    import boto3

    s3 = boto3.client("s3")

    def _download(uri: str, local: Path) -> None:
        parts = uri.replace("s3://", "").split("/", 1)
        s3.download_file(parts[0], parts[1], str(local))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        train_csv = tmp / "train.csv"
        val_csv = tmp / "validation.csv"
        _download(train_uri, train_csv)
        _download(val_uri, val_csv)

        train_df = pd.read_csv(train_csv).dropna(subset=["comment_text"])
        val_df = pd.read_csv(val_csv).dropna(subset=["comment_text"])

        X_train = train_df["comment_text"].astype(str).tolist()
        y_train = train_df[list(config.model.target_columns)].values
        X_val = val_df["comment_text"].astype(str).tolist()
        y_val = val_df[list(config.model.target_columns)].values

        model = ToxicCommentClassifier(
            max_features=max_features,
            ngram_range=(ngram_min, ngram_max),
            C=C,
            max_iter=max_iter,
        )

        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow_ok = setup_mlflow(run_name)

        ctx = mlflow.start_run(run_name=run_name) if mlflow_ok else contextlib.nullcontext()
        with ctx:
            if mlflow_ok:
                mlflow.log_params({
                    **model.get_params(),
                    "model_name": model_name,
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                })
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)
            metrics = compute_metrics(y_val, y_pred, y_proba)
            if mlflow_ok:
                mlflow.log_metrics(metrics)

            onnx_path = tmp / "model.onnx"
            print("Exporting to ONNX…")
            model.save_onnx(onnx_path)
            if mlflow_ok:
                mlflow.log_artifact(str(onnx_path))

            # Package as model.tar.gz — SageMaker endpoint ModelDataUrl must be a tarball
            import tarfile
            tarball_path = tmp / "model.tar.gz"
            with tarfile.open(tarball_path, "w:gz") as tar:
                tar.add(onnx_path, arcname="model.onnx")

            model_key = f"{run_prefix}/{model_name}/model.tar.gz"
            s3.upload_file(str(tarball_path), models_bucket, model_key)
            model_s3_uri = f"s3://{models_bucket}/{model_key}"
            print(f"Uploaded model → {model_s3_uri}")

            print("\nMetrics:")
            for name, value in sorted(metrics.items()):
                print(f"  {name}: {value:.4f}")

    return {"model_name": model_name, "model_s3_uri": model_s3_uri, **metrics}


def setup_mlflow(run_name: str) -> bool:
    """Configure MLflow. Returns True if setup succeeded, False if unavailable."""
    try:
        tracking_uri = config.mlflow.tracking_uri
        if tracking_uri == "databricks":
            os.environ["DATABRICKS_HOST"] = config.mlflow.databricks_host
            os.environ["DATABRICKS_TOKEN"] = config.mlflow.databricks_token
            mlflow.set_tracking_uri("databricks")
        else:
            mlflow.set_tracking_uri(tracking_uri)

        experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
        if experiment is None:
            mlflow.create_experiment(config.mlflow.experiment_name)
        mlflow.set_experiment(config.mlflow.experiment_name)
        return True
    except Exception as e:
        print(f"MLflow setup failed (tracking disabled): {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train toxic comment classifier")
    parser.add_argument("--model-name", type=str, required=True, help="baseline or improved")
    parser.add_argument("--max-features", type=int, default=config.model.tfidf_max_features)
    parser.add_argument("--ngram-min", type=int, default=config.model.tfidf_ngram_range[0])
    parser.add_argument("--ngram-max", type=int, default=config.model.tfidf_ngram_range[1])
    parser.add_argument("--C", type=float, default=config.model.lr_C)
    parser.add_argument("--max-iter", type=int, default=config.model.lr_max_iter)

    # SageMaker injects these as environment variables too, but we also accept CLI
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--validation", type=str, default="/opt/ml/input/data/validation")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--output-data-dir", type=str, default="/opt/ml/output/data")

    args = parser.parse_args()

    run_name = f"{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    setup_mlflow(run_name)

    # ── Load data ────────────────────────────────────────────────────────────────
    train_path = Path(args.train)
    val_path = Path(args.validation)

    train_files = list(train_path.glob("*.csv")) if train_path.is_dir() else [train_path]
    val_files = list(val_path.glob("*.csv")) if val_path.is_dir() else [val_path]

    train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
    val_df = pd.concat([pd.read_csv(f) for f in val_files], ignore_index=True)

    print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")

    X_train = train_df["comment_text"].tolist()
    y_train = train_df[list(config.model.target_columns)].values
    X_val = val_df["comment_text"].tolist()
    y_val = val_df[list(config.model.target_columns)].values

    # ── Train ────────────────────────────────────────────────────────────────────
    model = ToxicCommentClassifier(
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        C=args.C,
        max_iter=args.max_iter,
    )

    params = {
        **model.get_params(),
        "model_name": args.model_name,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
    }

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        metrics = compute_metrics(y_val, y_pred, y_proba)

        mlflow.log_metrics(metrics)

        print("\nMetrics:")
        for name, value in sorted(metrics.items()):
            print(f"  {name}: {value:.4f}")

        # ── Save ONNX artifact (only format used in serving) ──────────────────────
        model_dir = Path(args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        onnx_path = model_dir / "model.onnx"
        print("Exporting to ONNX…")
        model.save_onnx(onnx_path)
        mlflow.log_artifact(str(onnx_path))
        print(f"Saved ONNX model to {onnx_path}")

        # ── Write metrics.json for evaluate.py ───────────────────────────────────
        output_dir = Path(args.output_data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({"model_name": args.model_name, **metrics}, f, indent=2)
        print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
