"""
Data Ingestion Task for Databricks Lakeflow Pipeline.

This task pulls data from Unity Catalog or S3, validates it, and saves to Delta table.
"""
import argparse
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import config


def get_spark() -> SparkSession:
    """Get or create Spark session."""
    return SparkSession.builder.appName("DataIngestion").getOrCreate()


def validate_schema(df):
    """Validate that dataframe has required columns."""
    required_columns = {
        "id",
        "comment_text",
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    }
    
    actual_columns = set(df.columns)
    missing = required_columns - actual_columns
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def ingest_data(catalog: str, schema: str, source_path: str = None):
    """
    Ingest data from source and save to Delta table.
    
    Args:
        catalog: Unity Catalog name
        schema: Schema name
        source_path: Optional S3 path or local path to CSV files
    """
    spark = get_spark()
    
    # Create schema if not exists
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    
    # Read data from source
    if source_path:
        print(f"Reading data from {source_path}")
        df = spark.read.csv(source_path, header=True, inferSchema=True)
    else:
        # Check Unity Catalog Volume for CSV files
        volume_path = f"/Volumes/{catalog}/{schema}/data_volume/*.csv"
        try:
            # Check if any files exist in volume
            # We use try/except since verifying file existence via python os module 
            # might tricky with volume paths depending on cluster access mode, 
            # but spark.read will fail if no files match
            print(f"Checking for data in volume: {volume_path}")
            df = spark.read.csv(volume_path, header=True, inferSchema=True)
            print(f"Found data in volume!")
        except Exception:
            print("No new data found in volume.")
            
            # Try to read from existing raw_data table
            try:
                df = spark.table(f"{catalog}.{schema}.raw_data")
                print(f"Using existing data from {catalog}.{schema}.raw_data")
            except Exception as e:
                print(f"No existing data found, using sample data: {e}")
                # For initial setup, create sample data
                df = create_sample_data(spark)
    
    # Validate schema
    df = validate_schema(df)
    
    # Add metadata columns
    df = df.withColumn("ingestion_timestamp", F.current_timestamp())
    df = df.withColumn("source", F.lit(source_path or "existing_table"))
    
    # Write to Delta table
    table_name = f"{catalog}.{schema}.raw_data"
    print(f"Writing {df.count()} rows to {table_name}")
    
    df.write.format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(table_name)
    
    print(f"✓ Data ingestion complete: {table_name}")
    
    # Show sample
    print("\nSample data:")
    spark.table(table_name).show(5, truncate=False)


def create_sample_data(spark):
    """Create sample data for testing."""
    data = [
        ("1", "This is a normal comment", 0, 0, 0, 0, 0, 0),
        ("2", "You are stupid and I hate you", 1, 0, 1, 0, 1, 0),
        ("3", "I will kill you", 1, 1, 0, 1, 0, 0),
    ]
    
    schema = StructType([
        StructField("id", StringType(), False),
        StructField("comment_text", StringType(), False),
        StructField("toxic", IntegerType(), False),
        StructField("severe_toxic", IntegerType(), False),
        StructField("obscene", IntegerType(), False),
        StructField("threat", IntegerType(), False),
        StructField("insult", IntegerType(), False),
        StructField("identity_hate", IntegerType(), False),
    ])
    
    return spark.createDataFrame(data, schema)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Data Ingestion Task")
    parser.add_argument("--catalog", required=True, help="Unity Catalog name")
    parser.add_argument("--schema", required=True, help="Schema name")
    parser.add_argument("--source-path", help="Optional source path (S3 or local)")
    
    args = parser.parse_args()
    
    ingest_data(
        catalog=args.catalog,
        schema=args.schema,
        source_path=args.source_path,
    )


if __name__ == "__main__":
    main()
