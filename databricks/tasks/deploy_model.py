# Databricks notebook source
"""
Deploy Model Task for Databricks Lakeflow Pipeline.

This task deploys the production model to Databricks Model Serving endpoint.
"""
import argparse
import sys
import time
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    TrafficConfig,
    Route,
)
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_production_model_version(model_name: str) -> str:
    """Get the current production model version."""
    client = MlflowClient()
    
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    
    if not prod_versions:
        raise ValueError(f"No production model found for {model_name}")
    
    return prod_versions[0].version


def deploy_model(model_name: str, endpoint_name: str):
    """
    Deploy production model to serving endpoint.
    
    Args:
        model_name: Model name in registry
        endpoint_name: Serving endpoint name
    """
    print(f"\n{'='*60}")
    print("DEPLOY MODEL TO SERVING")
    print(f"{'='*60}")
    
    # Get production model version
    model_version = get_production_model_version(model_name)
    print(f"\nProduction Model Version: {model_version}")
    
    # Initialize Databricks client
    w = WorkspaceClient()
    
    # Check if endpoint exists
    try:
        endpoint = w.serving_endpoints.get(endpoint_name)
        print(f"\nEndpoint '{endpoint_name}' exists. Updating...")
        update_mode = True
    except Exception:
        print(f"\nEndpoint '{endpoint_name}' does not exist. Creating...")
        update_mode = False
    
    # Configure served entity
    served_entity = ServedEntityInput(
        entity_name=model_name,
        entity_version=model_version,
        scale_to_zero_enabled=True,
        workload_size="Small",
    )
    
    if update_mode:
        # Update existing endpoint
        w.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=[served_entity],
            traffic_config=TrafficConfig(
                routes=[Route(served_model_name=model_name, traffic_percentage=100)]
            ),
        )
        print(f"\n✓ Endpoint '{endpoint_name}' update initiated")
    else:
        # Create new endpoint
        w.serving_endpoints.create(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                served_entities=[served_entity],
            ),
        )
        print(f"\n✓ Endpoint '{endpoint_name}' creation initiated")
    
    # Wait for endpoint to be ready
    print("\nWaiting for endpoint to be ready...")
    max_wait = 600  # 10 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        endpoint = w.serving_endpoints.get(endpoint_name)
        state = endpoint.state.config_update if endpoint.state else None
        
        if state and state.value == "NOT_UPDATING":
            print(f"\n✓ Endpoint is ready!")
            break
        
        print(f"  Status: {state.value if state else 'UNKNOWN'}")
        time.sleep(30)
    else:
        print(f"\n⚠ Endpoint deployment timed out after {max_wait}s")
        print("  Check Databricks console for status")
    
    # Get endpoint URL
    endpoint = w.serving_endpoints.get(endpoint_name)
    endpoint_url = f"{w.config.host}/serving-endpoints/{endpoint_name}/invocations"
    
    print(f"\n{'='*60}")
    print("DEPLOYMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Endpoint Name: {endpoint_name}")
    print(f"Model: {model_name} v{model_version}")
    print(f"Endpoint URL: {endpoint_url}")
    print(f"\nTest with:")
    print(f"  curl -X POST {endpoint_url} \\")
    print(f"    -H 'Authorization: Bearer $DATABRICKS_TOKEN' \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"inputs\": [\"test comment\"]}}'")


def get_arg(key, default=None, args=None):
    """Get argument from argparse args or dbutils widgets."""
    # Try getting from args first (local execution)
    if args and hasattr(args, key.replace('-', '_')) and getattr(args, key.replace('-', '_')) is not None:
        return getattr(args, key.replace('-', '_'))
    
    # Try getting from dbutils widgets (Databricks execution)
    try:
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)
        val = dbutils.widgets.get(key)
        if val:
            return val
    except Exception:
        pass
        
    return default


def main():
    """Main entry point."""
    # Define display for local execution if not exists
    if 'display' not in globals():
        def display(df):
            # If it's a spark dataframe, show it
            if hasattr(df, 'show'):
                df.show(5, truncate=False)
            # If it's a pandas dataframe, print it
            elif hasattr(df, 'head'):
                print(df.head())
            else:
                print(df)

    parser = argparse.ArgumentParser(description="Deploy Model Task")
    parser.add_argument("--model-name", help="Model name in registry")
    parser.add_argument("--endpoint-name", help="Serving endpoint name")
    
    try:
        args, _ = parser.parse_known_args()
    except:
        args = None
    
    # Get parameters (priority: args > widgets)
    model_name = get_arg("model-name", args=args)
    endpoint_name = get_arg("endpoint-name", args=args)
    
    if not model_name or not endpoint_name:
        raise ValueError("Model name and endpoint name are required")
    
    deploy_model(
        model_name=model_name,
        endpoint_name=endpoint_name,
    )


if __name__ == "__main__":
    main()
