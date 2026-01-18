"""
Feature Engineering Task for Databricks Lakeflow Pipeline.

This task applies text preprocessing and feature engineering to raw data.
"""
import argparse
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocessing import TextPreprocessor


def get_spark() -> SparkSession:
    """Get or create Spark session."""
    return SparkSession.builder.appName("FeatureEngineering").getOrCreate()


def preprocess_text_udf(preprocessor: TextPreprocessor):
    """Create UDF for text preprocessing."""
    def _preprocess(text: str) -> str:
        if text is None:
            return ""
        return preprocessor.preprocess_text(text)
    
    return F.udf(_preprocess, StringType())


def engineer_features(catalog: str, schema: str):
    """
    Apply feature engineering to raw data.
    
    Args:
        catalog: Unity Catalog name
        schema: Schema name
    """
    spark = get_spark()
    
    # Read raw data
    raw_table = f"{catalog}.{schema}.raw_data"
    print(f"Reading data from {raw_table}")
    df = spark.table(raw_table)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_urls=True,
        remove_html=True,
        remove_special_chars=True,
        remove_extra_whitespace=True,
    )
    
    # Apply text preprocessing
    preprocess_udf = preprocess_text_udf(preprocessor)
    df = df.withColumn("processed_text", preprocess_udf(F.col("comment_text")))
    
    # Add additional features
    df = df.withColumn("text_length", F.length(F.col("comment_text")))
    df = df.withColumn("processed_length", F.length(F.col("processed_text")))
    df = df.withColumn("word_count", F.size(F.split(F.col("processed_text"), " ")))
    df = df.withColumn("uppercase_ratio", 
                       F.length(F.regexp_replace(F.col("comment_text"), "[^A-Z]", "")) / 
                       F.greatest(F.length(F.col("comment_text")), F.lit(1)))
    df = df.withColumn("exclamation_count", 
                       F.length(F.col("comment_text")) - 
                       F.length(F.regexp_replace(F.col("comment_text"), "!", "")))
    df = df.withColumn("question_count",
                       F.length(F.col("comment_text")) - 
                       F.length(F.regexp_replace(F.col("comment_text"), "\\?", "")))
    
    # Add processing timestamp
    df = df.withColumn("processing_timestamp", F.current_timestamp())
    
    # Write to Delta table
    processed_table = f"{catalog}.{schema}.processed_data"
    print(f"Writing {df.count()} rows to {processed_table}")
    
    df.write.format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(processed_table)
    
    print(f"✓ Feature engineering complete: {processed_table}")
    
    # Show sample
    print("\nSample processed data:")
    spark.table(processed_table).select(
        "id", "comment_text", "processed_text", "text_length", "word_count"
    ).show(5, truncate=False)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Feature Engineering Task")
    parser.add_argument("--catalog", required=True, help="Unity Catalog name")
    parser.add_argument("--schema", required=True, help="Schema name")
    
    args = parser.parse_args()
    
    engineer_features(
        catalog=args.catalog,
        schema=args.schema,
    )


if __name__ == "__main__":
    main()
