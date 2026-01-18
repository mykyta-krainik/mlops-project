"""
Challenger Task for Databricks Lakeflow Pipeline.

This task compares the newly trained model with the current production model
and promotes the new model if it beats the threshold.
"""
import argparse
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_production_model_metrics(model_name: str) -> dict:
    """
    Get metrics for current production model.
    
    Args:
        model_name: Model name in registry
    
    Returns:
        Dictionary with metrics, or None if no production model
    """
    client = MlflowClient()
    
    try:
        # Get production model version
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if not prod_versions:
            print("No production model found")
            return None
        
        prod_version = prod_versions[0]
        run_id = prod_version.run_id
        
        # Get run metrics
        run = client.get_run(run_id)
        metrics = run.data.metrics
        
        print(f"\nCurrent Production Model:")
        print(f"  Version: {prod_version.version}")
        print(f"  Run ID: {run_id}")
        print(f"  F1 (macro): {metrics.get('f1_macro', 0):.4f}")
        print(f"  ROC-AUC (macro): {metrics.get('roc_auc_macro', 0):.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"Error getting production model: {e}")
        return None


def get_latest_model_metrics(model_name: str) -> tuple:
    """
    Get metrics for the latest trained model.
    
    Args:
        model_name: Model name in registry
    
    Returns:
        Tuple of (version, metrics)
    """
    client = MlflowClient()
    
    # Get latest version (not in production yet)
    versions = client.search_model_versions(f"name='{model_name}'")
    
    if not versions:
        raise ValueError(f"No versions found for model {model_name}")
    
    # Sort by version number and get latest
    latest_version = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    run_id = latest_version.run_id
    
    # Get run metrics
    run = client.get_run(run_id)
    metrics = run.data.metrics
    
    print(f"\nLatest Trained Model:")
    print(f"  Version: {latest_version.version}")
    print(f"  Run ID: {run_id}")
    print(f"  F1 (macro): {metrics.get('f1_macro', 0):.4f}")
    print(f"  ROC-AUC (macro): {metrics.get('roc_auc_macro', 0):.4f}")
    
    return latest_version, metrics


def challenger(model_name: str, metric_threshold: float):
    """
    Compare new model with production and promote if better.
    
    Args:
        model_name: Model name in registry
        metric_threshold: Minimum improvement percentage to promote (e.g., 2.0 for 2%)
    """
    client = MlflowClient()
    
    print(f"\n{'='*60}")
    print("CHALLENGER TASK: Model Comparison & Promotion")
    print(f"{'='*60}")
    
    # Get production model metrics
    prod_metrics = get_production_model_metrics(model_name)
    
    # Get latest model metrics
    latest_version, latest_metrics = get_latest_model_metrics(model_name)
    
    # Primary metric for comparison
    metric_name = "f1_macro"
    
    latest_score = latest_metrics.get(metric_name, 0)
    
    if prod_metrics is None:
        # No production model, promote latest
        print(f"\n✓ No production model exists. Promoting version {latest_version.version} to Production")
        promote = True
        improvement = None
    else:
        prod_score = prod_metrics.get(metric_name, 0)
        
        # Calculate improvement percentage
        if prod_score > 0:
            improvement = ((latest_score - prod_score) / prod_score) * 100
        else:
            improvement = 100.0 if latest_score > 0 else 0.0
        
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"Metric: {metric_name}")
        print(f"Production Score: {prod_score:.4f}")
        print(f"Latest Score: {latest_score:.4f}")
        print(f"Improvement: {improvement:+.2f}%")
        print(f"Threshold: {metric_threshold:+.2f}%")
        
        promote = improvement >= metric_threshold
    
    if promote:
        print(f"\n✓ PROMOTING model version {latest_version.version} to Production")
        
        # Transition current production to archived
        if prod_metrics is not None:
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])
            for pv in prod_versions:
                client.transition_model_version_stage(
                    name=model_name,
                    version=pv.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )
                print(f"  Archived previous production version {pv.version}")
        
        # Promote latest to production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage="Production",
            archive_existing_versions=False,
        )
        
        # Add description
        if improvement is not None:
            description = f"Promoted to production with {improvement:+.2f}% improvement in {metric_name}"
        else:
            description = "Promoted to production (first production model)"
        
        client.update_model_version(
            name=model_name,
            version=latest_version.version,
            description=description,
        )
        
        print(f"\n{'='*60}")
        print("✓ PROMOTION SUCCESSFUL")
        print(f"{'='*60}")
        
    else:
        print(f"\n✗ Model does NOT meet threshold. Keeping current production model.")
        print(f"  Required improvement: {metric_threshold:+.2f}%")
        print(f"  Actual improvement: {improvement:+.2f}%")
        
        # Transition to archived
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage="Archived",
            archive_existing_versions=False,
        )
        
        print(f"\n{'='*60}")
        print("✗ NO PROMOTION")
        print(f"{'='*60}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Challenger Task")
    parser.add_argument("--model-name", required=True, help="Model name in registry")
    parser.add_argument("--metric-threshold", type=float, default=2.0,
                       help="Minimum improvement percentage to promote (default: 2.0)")
    
    args = parser.parse_args()
    
    challenger(
        model_name=args.model_name,
        metric_threshold=args.metric_threshold,
    )


if __name__ == "__main__":
    main()
