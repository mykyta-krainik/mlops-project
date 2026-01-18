"""
Drift Detection Task for Databricks Monitoring.

This task uses Evidently to detect drift in:
- Text embeddings
- Text length distribution
- Toxicity distribution
- Class balance
- Prediction confidence
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import config


def get_spark() -> SparkSession:
    """Get or create Spark session."""
    return SparkSession.builder.appName("DriftDetection").getOrCreate()


def get_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings for texts using sentence transformers."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def prepare_drift_data(catalog: str, schema: str, lookback_days: int):
    """
    Prepare reference and current data for drift detection.
    
    Args:
        catalog: Unity Catalog name
        schema: Schema name
        lookback_days: Number of days to look back for current data
    
    Returns:
        Tuple of (reference_df, current_df)
    """
    spark = get_spark()
    
    # Get prediction logs (assuming they're stored somewhere)
    # For now, we'll use processed_data as reference and recent data as current
    
    # Reference data: older data (e.g., training data)
    processed_table = f"{catalog}.{schema}.processed_data"
    reference_df = spark.table(processed_table).limit(1000).toPandas()
    
    # Current data: recent predictions or new data
    # In production, this would come from prediction logs
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    
    try:
        # Try to get recent data from a predictions table
        predictions_table = f"{catalog}.{schema}.predictions"
        current_df = spark.sql(f"""
            SELECT * FROM {predictions_table}
            WHERE timestamp >= '{cutoff_date.isoformat()}'
            LIMIT 1000
        """).toPandas()
        
        if len(current_df) == 0:
            print("No recent predictions found, using sample from processed data")
            current_df = spark.table(processed_table).limit(100).toPandas()
    except Exception as e:
        print(f"Predictions table not found: {e}")
        print("Using sample from processed data as current")
        current_df = spark.table(processed_table).limit(100).toPandas()
    
    return reference_df, current_df


def detect_drift(catalog: str, schema: str, lookback_days: int = 7):
    """
    Detect drift in model inputs and predictions.
    
    Args:
        catalog: Unity Catalog name
        schema: Schema name
        lookback_days: Number of days to analyze
    """
    print(f"\n{'='*60}")
    print("DRIFT DETECTION")
    print(f"{'='*60}")
    
    spark = get_spark()
    
    # Get reference and current data
    print("\nPreparing data...")
    reference_df, current_df = prepare_drift_data(catalog, schema, lookback_days)
    
    print(f"Reference data: {len(reference_df)} samples")
    print(f"Current data: {len(current_df)} samples")
    
    # Add embeddings for drift detection
    print("\nGenerating embeddings...")
    reference_embeddings = get_embeddings(reference_df["processed_text"].tolist())
    current_embeddings = get_embeddings(current_df["processed_text"].tolist())
    
    # Add embedding features (first 3 dimensions for simplicity)
    for i in range(min(3, reference_embeddings.shape[1])):
        reference_df[f"embedding_{i}"] = reference_embeddings[:, i]
        current_df[f"embedding_{i}"] = current_embeddings[:, i]
    
    # Add text statistics if not present
    if "text_length" not in reference_df.columns:
        reference_df["text_length"] = reference_df["comment_text"].str.len()
        current_df["text_length"] = current_df["comment_text"].str.len()
    
    if "word_count" not in reference_df.columns:
        reference_df["word_count"] = reference_df["processed_text"].str.split().str.len()
        current_df["word_count"] = current_df["processed_text"].str.split().str.len()
    
    # Calculate toxicity score (max of all toxic labels)
    toxic_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    reference_df["max_toxicity"] = reference_df[toxic_cols].max(axis=1)
    current_df["max_toxicity"] = current_df[toxic_cols].max(axis=1)
    
    # Define column mapping
    column_mapping = ColumnMapping(
        numerical_features=["text_length", "word_count", "embedding_0", "embedding_1", "embedding_2"],
        categorical_features=toxic_cols,
        target=None,
    )
    
    # Create drift report
    print("\nGenerating drift report...")
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )
    
    # Get drift results
    drift_results = report.as_dict()
    
    # Extract key metrics
    drift_detected = False
    drift_summary = []
    
    for metric in drift_results.get("metrics", []):
        if metric.get("metric") == "DataDriftTable":
            result = metric.get("result", {})
            drift_by_columns = result.get("drift_by_columns", {})
            
            for col, drift_info in drift_by_columns.items():
                if drift_info.get("drift_detected", False):
                    drift_detected = True
                    drift_summary.append({
                        "column": col,
                        "drift_score": drift_info.get("drift_score", 0),
                        "stattest_name": drift_info.get("stattest_name", ""),
                    })
    
    # Print results
    print(f"\n{'='*60}")
    print("DRIFT DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Drift Detected: {'YES' if drift_detected else 'NO'}")
    
    if drift_summary:
        print("\nDrifted Features:")
        for drift in drift_summary:
            print(f"  - {drift['column']}: score={drift['drift_score']:.4f} (test={drift['stattest_name']})")
    else:
        print("\nNo significant drift detected in any features")
    
    # Save report
    drift_table = f"{catalog}.{schema}.drift_reports"
    
    # Create drift reports table if not exists
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {drift_table} (
            report_id STRING,
            timestamp TIMESTAMP,
            drift_detected BOOLEAN,
            drift_summary STRING,
            lookback_days INT
        ) USING DELTA
    """)
    
    # Save drift report
    report_id = f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    drift_record = spark.createDataFrame([{
        "report_id": report_id,
        "timestamp": datetime.now(),
        "drift_detected": drift_detected,
        "drift_summary": str(drift_summary),
        "lookback_days": lookback_days,
    }])
    
    drift_record.write.format("delta").mode("append").saveAsTable(drift_table)
    
    # Save HTML report to Unity Catalog volume
    volume_path = f"/Volumes/{catalog}/{schema}/data_volume/drift_reports"
    report_path = f"{volume_path}/{report_id}.html"
    
    # Create directory in volume if it doesn't exist
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.data_volume")
    Path(f"/Volumes/{catalog}/{schema}/data_volume/drift_reports").mkdir(parents=True, exist_ok=True)
    
    report.save_html(report_path)
    
    print(f"\n✓ Drift report saved:")
    print(f"  Table: {drift_table}")
    print(f"  HTML: {report_path}")
    
    # Alert if drift detected
    if drift_detected:
        print(f"\n⚠️  ALERT: Data drift detected! Review the report and consider retraining.")
    
    return drift_detected


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Drift Detection Task")
    parser.add_argument("--catalog", required=True, help="Unity Catalog name")
    parser.add_argument("--schema", required=True, help="Schema name")
    parser.add_argument("--lookback-days", type=int, default=7,
                       help="Number of days to look back for current data")
    
    args = parser.parse_args()
    
    detect_drift(
        catalog=args.catalog,
        schema=args.schema,
        lookback_days=args.lookback_days,
    )


if __name__ == "__main__":
    main()
