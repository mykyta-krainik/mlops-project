import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _get_minio_endpoint() -> str:
    use_local = os.getenv("USE_LOCAL_MINIO", "true").lower() == "true"
    
    if use_local:
        return os.getenv("MINIO_ENDPOINT_LOCAL", "localhost:9000")
    
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    if endpoint.startswith("https://"):
        endpoint = endpoint[8:]
    elif endpoint.startswith("http://"):
        endpoint = endpoint[7:]
    return endpoint


def _get_minio_secure() -> bool:
    use_local = os.getenv("USE_LOCAL_MINIO", "true").lower() == "true"
    
    if use_local:
        return False
    
    return os.getenv("MINIO_SECURE", "true").lower() == "true"


@dataclass
class MinioConfig:
    endpoint: str = field(default_factory=_get_minio_endpoint)
    access_key: str = field(default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "minioadmin"))
    secret_key: str = field(default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "minioadmin"))
    secure: bool = field(default_factory=_get_minio_secure)

    raw_data_bucket: str = "raw-data"
    processed_data_bucket: str = "processed-data"
    models_bucket: str = "models"
    mlflow_artifacts_bucket: str = "mlflow-artifacts"


@dataclass
class MLflowConfig:
    tracking_uri: str = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "databricks"))
    databricks_host: str = field(
        default_factory=lambda: os.getenv("DATABRICKS_HOST", "")
    )
    databricks_token: str = field(default_factory=lambda: os.getenv("DATABRICKS_TOKEN", ""))
    experiment_name: str = "/Users/krainik.mykyta@lll.kpi.ua/mlops-project"


@dataclass
class WandbConfig:
    api_key: str = field(default_factory=lambda: os.getenv("WANDB_API_KEY", ""))
    project: str = field(default_factory=lambda: os.getenv("WANDB_PROJECT", "mlops-project"))
    entity: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_ENTITY"))


@dataclass
class ModerationThresholds:
    ban_severe_toxic: float = field(
        default_factory=lambda: float(os.getenv("THRESHOLD_BAN_SEVERE_TOXIC", "0.7"))
    )
    ban_threat: float = field(default_factory=lambda: float(os.getenv("THRESHOLD_BAN_THREAT", "0.6")))
    ban_toxic: float = field(default_factory=lambda: float(os.getenv("THRESHOLD_BAN_TOXIC", "0.85")))
    ban_obscene: float = field(default_factory=lambda: float(os.getenv("THRESHOLD_BAN_OBSCENE", "0.8")))
    ban_insult: float = field(default_factory=lambda: float(os.getenv("THRESHOLD_BAN_INSULT", "0.8")))
    ban_identity_hate: float = field(default_factory=lambda: float(os.getenv("THRESHOLD_BAN_IDENTITY_HATE", "0.7")))
    review_min: float = field(default_factory=lambda: float(os.getenv("THRESHOLD_REVIEW_MIN", "0.5")))


@dataclass
class APIConfig:
    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "5000")))
    model_version: str = field(default_factory=lambda: os.getenv("MODEL_VERSION", "v1.0.0"))


@dataclass
class ModelConfig:
    model_type: str = "tfidf_lr"

    tfidf_max_features: int = 10000
    tfidf_ngram_range: tuple = (1, 2)
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95

    lr_C: float = 1.0
    lr_max_iter: int = 1000
    lr_solver: str = "lbfgs"

    test_size: float = 0.2
    random_state: int = 42

    target_columns: tuple = (
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    )


@dataclass
class DatabricksConfig:
    host: str = field(default_factory=lambda: os.getenv("DATABRICKS_HOST", ""))
    token: str = field(default_factory=lambda: os.getenv("DATABRICKS_TOKEN", ""))
    catalog: str = field(default_factory=lambda: os.getenv("DATABRICKS_CATALOG", "mlops_catalog"))
    schema: str = field(default_factory=lambda: os.getenv("DATABRICKS_SCHEMA", "toxic_comments"))
    serving_endpoint: str = field(default_factory=lambda: os.getenv("DATABRICKS_SERVING_ENDPOINT", ""))
    cluster_id: str = field(default_factory=lambda: os.getenv("DATABRICKS_CLUSTER_ID", ""))
    bundle_env: str = field(default_factory=lambda: os.getenv("DATABRICKS_BUNDLE_ENV", "dev"))


@dataclass
class ServingConfig:
    use_databricks: bool = field(
        default_factory=lambda: os.getenv("USE_DATABRICKS_SERVING", "true").lower() == "true"
    )
    model_type: str = field(default_factory=lambda: os.getenv("MODEL_TYPE", "baseline"))
    metric_threshold: float = field(default_factory=lambda: float(os.getenv("METRIC_THRESHOLD", "2.0")))


@dataclass
class MonitoringConfig:
    drift_threshold: float = field(
        default_factory=lambda: float(os.getenv("EVIDENTLY_DRIFT_THRESHOLD", "0.1"))
    )
    lookback_days: int = field(default_factory=lambda: int(os.getenv("MONITORING_LOOKBACK_DAYS", "7")))


@dataclass
class Config:
    minio: MinioConfig = field(default_factory=MinioConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    moderation: ModerationThresholds = field(default_factory=ModerationThresholds)
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    databricks: DatabricksConfig = field(default_factory=DatabricksConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


config = Config()

