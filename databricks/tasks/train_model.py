"""
Model Training Task for Databricks Lakeflow Pipeline.

This task trains either a baseline or improved model based on MODEL_TYPE flag.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
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
from src.models.baseline import ToxicCommentClassifier


def get_spark() -> SparkSession:
    """Get or create Spark session."""
    return SparkSession.builder.appName("ModelTraining").getOrCreate()


def get_model_config(model_type: str) -> Dict:
    """
    Get model configuration based on model type.
    
    Args:
        model_type: "baseline" or "improved"
    
    Returns:
        Dictionary of model hyperparameters
    """
    if model_type == "baseline":
        return {
            "max_features": 10000,
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95,
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
        }
    elif model_type == "improved":
        # Improved model with optimized hyperparameters
        return {
            "max_features": 15000,  # More features
            "ngram_range": (1, 3),  # Include trigrams
            "min_df": 3,  # Slightly higher min_df
            "max_df": 0.9,  # Lower max_df to filter common words
            "C": 0.5,  # Regularization tuning
            "max_iter": 1500,  # More iterations
            "solver": "saga",  # Better for large datasets
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'baseline' or 'improved'")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
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


def train_model(catalog: str, schema: str, model_type: str, model_name: str):
    """
    Train model based on model_type flag.
    
    Args:
        catalog: Unity Catalog name
        schema: Schema name
        model_type: "baseline" or "improved"
        model_name: Model name for MLflow registry
    """
    spark = get_spark()
    
    # Read processed data
    processed_table = f"{catalog}.{schema}.processed_data"
    print(f"Reading data from {processed_table}")
    df_spark = spark.table(processed_table)
    
    # Convert to Pandas for sklearn
    df = df_spark.toPandas()
    print(f"Loaded {len(df)} samples")
    
    # Prepare features and labels
    X = df["processed_text"].tolist()
    y = df[list(config.model.target_columns)].values
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config.model.test_size,
        random_state=config.model.random_state,
    )
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Get model configuration
    model_config = get_model_config(model_type)
    print(f"\nModel Type: {model_type}")
    print(f"Configuration: {model_config}")
    
    # Initialize model with configuration
    model = ToxicCommentClassifier(**model_config)
    
    # Setup MLflow
    mlflow.set_experiment(config.mlflow.experiment_name)
    
    run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        params = {
            **model_config,
            "model_type": model_type,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_size": config.model.test_size,
            "random_state": config.model.random_state,
        }
        mlflow.log_params(params)
        
        # Train model
        print("\nTraining model...")
        model.fit(X_train, y_train)
        
        # Evaluate
        print("Evaluating model...")
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        
        metrics = compute_metrics(y_val, y_pred, y_proba)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        print("\n=== Metrics ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")
        
        # Log model to MLflow
        print("\nLogging model to MLflow...")
        mlflow.sklearn.log_model(
            model._pipeline,
            "model",
            registered_model_name=model_name,
        )
        
        # Add tags
        mlflow.set_tags({
            "model_type": model_type,
            "framework": "sklearn",
            "project": "mlops-toxic-comments",
        })
        
        run_id = run.info.run_id
        print(f"\n✓ Model training complete!")
        print(f"Run ID: {run_id}")
        print(f"Model registered as: {model_name}")
        
        # Store run ID and metrics for challenger task
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {catalog}.{schema}.training_runs (
                run_id STRING,
                model_type STRING,
                f1_macro DOUBLE,
                roc_auc_macro DOUBLE,
                timestamp TIMESTAMP
            ) USING DELTA
        """)
        
        run_data = spark.createDataFrame([{
            "run_id": run_id,
            "model_type": model_type,
            "f1_macro": metrics["f1_macro"],
            "roc_auc_macro": metrics["roc_auc_macro"],
            "timestamp": datetime.now(),
        }])
        
        run_data.write.format("delta").mode("append").saveAsTable(
            f"{catalog}.{schema}.training_runs"
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Model Training Task")
    parser.add_argument("--catalog", required=True, help="Unity Catalog name")
    parser.add_argument("--schema", required=True, help="Schema name")
    parser.add_argument("--model-type", required=True, choices=["baseline", "improved"],
                       help="Model type to train")
    parser.add_argument("--model-name", required=True, help="Model name for registry")
    
    args = parser.parse_args()
    
    train_model(
        catalog=args.catalog,
        schema=args.schema,
        model_type=args.model_type,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
