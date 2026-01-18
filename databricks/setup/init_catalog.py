from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession


def init_catalog(catalog_name: str, schema_name: str):
    """
    Initialize Unity Catalog and schema.
    
    Args:
        catalog_name: Name of the catalog to create
        schema_name: Name of the schema to create
    """
    print(f"\n{'='*60}")
    print("INITIALIZING UNITY CATALOG")
    print(f"{'='*60}")
    
    spark = SparkSession.builder.appName("InitCatalog").getOrCreate()
    
    # Create catalog
    print(f"\nCreating catalog: {catalog_name}")
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
    
    # Create schema
    print(f"Creating schema: {catalog_name}.{schema_name}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
    
    # Create volume for data files
    print(f"\nCreating volume for data storage...")
    spark.sql(f"""
        CREATE VOLUME IF NOT EXISTS {catalog_name}.{schema_name}.data_volume
        COMMENT 'Volume for storing raw data files and reports'
    """)
    print(f"  ✓ {catalog_name}.{schema_name}.data_volume")
    
    # Create tables
    print("\nCreating Delta tables...")
    
    # Raw data table
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {catalog_name}.{schema_name}.raw_data (
            id STRING,
            comment_text STRING,
            toxic INT,
            severe_toxic INT,
            obscene INT,
            threat INT,
            insult INT,
            identity_hate INT,
            ingestion_timestamp TIMESTAMP,
            source STRING
        ) USING DELTA
    """)
    print(f"  ✓ {catalog_name}.{schema_name}.raw_data")
    
    # Processed data table
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {catalog_name}.{schema_name}.processed_data (
            id STRING,
            comment_text STRING,
            processed_text STRING,
            toxic INT,
            severe_toxic INT,
            obscene INT,
            threat INT,
            insult INT,
            identity_hate INT,
            text_length INT,
            processed_length INT,
            word_count INT,
            uppercase_ratio DOUBLE,
            exclamation_count INT,
            question_count INT,
            processing_timestamp TIMESTAMP,
            ingestion_timestamp TIMESTAMP,
            source STRING
        ) USING DELTA
    """)
    print(f"  ✓ {catalog_name}.{schema_name}.processed_data")
    
    # Training runs table
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {catalog_name}.{schema_name}.training_runs (
            run_id STRING,
            model_type STRING,
            f1_macro DOUBLE,
            roc_auc_macro DOUBLE,
            timestamp TIMESTAMP
        ) USING DELTA
    """)
    print(f"  ✓ {catalog_name}.{schema_name}.training_runs")
    
    # Predictions table (for monitoring)
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {catalog_name}.{schema_name}.predictions (
            id STRING,
            comment_text STRING,
            processed_text STRING,
            toxic_pred DOUBLE,
            severe_toxic_pred DOUBLE,
            obscene_pred DOUBLE,
            threat_pred DOUBLE,
            insult_pred DOUBLE,
            identity_hate_pred DOUBLE,
            timestamp TIMESTAMP,
            model_version STRING
        ) USING DELTA
    """)
    print(f"  ✓ {catalog_name}.{schema_name}.predictions")
    
    # Drift reports table
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {catalog_name}.{schema_name}.drift_reports (
            report_id STRING,
            timestamp TIMESTAMP,
            drift_detected BOOLEAN,
            drift_summary STRING,
            lookback_days INT
        ) USING DELTA
    """)
    print(f"  ✓ {catalog_name}.{schema_name}.drift_reports")
    
    print(f"\n{'='*60}")
    print("✓ CATALOG INITIALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nCatalog: {catalog_name}")
    print(f"Schema: {catalog_name}.{schema_name}")
    print("\nTables created:")
    print(f"  - raw_data")
    print(f"  - processed_data")
    print(f"  - training_runs")
    print(f"  - predictions")
    print(f"  - drift_reports")

init_catalog(
    catalog_name="mlops_catalog",
    schema_name="toxic_comments",
)

