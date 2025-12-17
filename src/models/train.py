import argparse
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import config
from src.data.preprocessing import TextPreprocessor
from src.data.storage import MinioStorage
from src.models.baseline import ToxicCommentClassifier


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        metrics["roc_auc_macro"] = roc_auc_score(y_true, y_proba, average="macro")
        metrics["roc_auc_micro"] = roc_auc_score(y_true, y_proba, average="micro")
    except ValueError:
        # Handle case where some classes have no positive samples
        metrics["roc_auc_macro"] = 0.0
        metrics["roc_auc_micro"] = 0.0

    target_columns = config.model.target_columns

    for i, label in enumerate(target_columns):
        metrics[f"f1_{label}"] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        metrics[f"precision_{label}"] = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        metrics[f"recall_{label}"] = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)

        try:
            metrics[f"roc_auc_{label}"] = roc_auc_score(y_true[:, i], y_proba[:, i])
        except ValueError:
            metrics[f"roc_auc_{label}"] = 0.0

    return metrics


def load_data_from_minio(
    storage: MinioStorage,
    bucket: str,
    object_name: str,
) -> pd.DataFrame:
    if storage.object_exists(bucket, object_name):
        return storage.download_dataframe(bucket, object_name)

    objects = storage.list_objects(bucket, prefix=object_name, recursive=True)
    csv_objects = [obj for obj in objects if obj.endswith(".csv")]

    if not csv_objects:
        raise ValueError(f"No CSV files found in bucket '{bucket}' with prefix '{object_name}'")

    dfs = []
    for obj in csv_objects:
        dfs.append(storage.download_dataframe(bucket, obj))

    return pd.concat(dfs, ignore_index=True)


def load_data_from_file(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    
    if path.is_dir():
        files = sorted(path.glob("**/*.csv"))
        if not files:
            raise ValueError(f"No CSV files found in directory '{file_path}'")
        
        dfs = [pd.read_csv(f) for f in files]
        return pd.concat(dfs, ignore_index=True)
        
    return pd.read_csv(file_path)


def setup_mlflow() -> None:
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


def setup_wandb(run_name: str) -> wandb.sdk.wandb_run.Run:
    return wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity if config.wandb.entity else None,
        name=run_name,
        config={
            "model_type": "TF-IDF + LogReg",
            "framework": "sklearn",
        },
    )


def train(
    data_source: str,
    use_minio: bool = False,
    run_name: Optional[str] = None,
    save_to_minio: bool = True,
) -> Tuple[ToxicCommentClassifier, Dict[str, float]]:
    run_name = run_name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    setup_mlflow()
    wandb_run = setup_wandb(run_name)

    storage = MinioStorage() if (use_minio or save_to_minio) else None

    if use_minio:
        df = load_data_from_minio(storage, config.minio.raw_data_bucket, data_source)
    else:
        df = load_data_from_file(data_source)

    print(f"Loaded {len(df)} samples")

    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df)

    X = df["comment_text"].tolist()
    y = df[list(config.model.target_columns)].values

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config.model.test_size,
        random_state=config.model.random_state,
    )

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    model = ToxicCommentClassifier()

    params = model.get_params()
    params["train_samples"] = len(X_train)
    params["val_samples"] = len(X_val)
    params["test_size"] = config.model.test_size
    params["random_state"] = config.model.random_state

    wandb.config.update(params)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        metrics = compute_metrics(y_val, y_pred, y_proba)

        mlflow.log_metrics(metrics)
        wandb.log(metrics)

        print("\nMetrics:")
        for name, value in sorted(metrics.items()):
            print(f"  {name}: {value:.4f}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            sklearn_path = tmpdir / "model.pkl"
            model.save_sklearn(sklearn_path)
            mlflow.log_artifact(str(sklearn_path))

            onnx_path = tmpdir / "model.onnx"
            print("Exporting to ONNX format...")
            model.save_onnx(onnx_path)
            mlflow.log_artifact(str(onnx_path))

            artifact = wandb.Artifact(
                name=f"model-{run_name}",
                type="model",
                description="TF-IDF + LogReg toxic comment classifier",
            )
            artifact.add_file(str(sklearn_path))
            artifact.add_file(str(onnx_path))
            wandb_run.log_artifact(artifact)

            if save_to_minio and storage:
                storage.upload_file(
                    config.minio.models_bucket,
                    f"{run_name}/model.pkl",
                    sklearn_path,
                )
                storage.upload_file(
                    config.minio.models_bucket,
                    f"{run_name}/model.onnx",
                    onnx_path,
                )

    wandb_run.finish()

    return model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="jigsaw-toxic-comment-classification-challenge/train.csv",
        help="Path to training data CSV or Minio object name",
    )
    parser.add_argument(
        "--minio",
        action="store_true",
        help="Load data from Minio instead of local file",
    )
    parser.add_argument(
        "--run-name",
        "-n",
        type=str,
        default=None,
        help="Name for the training run",
    )
    parser.add_argument(
        "--no-minio-save",
        action="store_true",
        help="Don't save artifacts to Minio",
    )
    args = parser.parse_args()

    model, metrics = train(
        data_source=args.data,
        use_minio=args.minio,
        run_name=args.run_name,
        save_to_minio=not args.no_minio_save,
    )

    print(f"\nTraining complete!")
    print(f"Final ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")


if __name__ == "__main__":
    main()
