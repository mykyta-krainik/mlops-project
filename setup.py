"""Setup configuration for MLOps Toxic Comments package."""
from setuptools import setup, find_packages

setup(
    name="mlops_toxic_comments",
    version="1.0.0",
    author="MLOps Team",
    description="Toxic comment classification MLOps pipeline for Databricks",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn>=1.3.0",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "mlflow>=2.9.0",
        "databricks-sdk>=0.18.0",
        "pyspark>=3.4.0",
        "delta-spark>=2.4.0",
        "evidently>=0.4.0",
        "sentence-transformers>=2.2.0",
    ],
    entry_points={
        "console_scripts": [
            "data_ingestion=databricks.tasks.data_ingestion:main",
            "feature_engineering=databricks.tasks.feature_engineering:main",
            "train_model=databricks.tasks.train_model:main",
            "challenger=databricks.tasks.challenger:main",
            "deploy_model=databricks.tasks.deploy_model:main",
            "drift_detection=databricks.monitoring.drift_detection:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
    ],
)
