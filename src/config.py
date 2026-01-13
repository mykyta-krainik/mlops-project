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
    """Configuration for MinIO/S3 compatible storage (local development)."""
    endpoint: str = field(default_factory=_get_minio_endpoint)
    access_key: str = field(default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "minioadmin"))
    secret_key: str = field(default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "minioadmin"))
    secure: bool = field(default_factory=_get_minio_secure)

    raw_data_bucket: str = "raw-data"
    processed_data_bucket: str = "processed-data"
    models_bucket: str = "models"
    mlflow_artifacts_bucket: str = "mlflow-artifacts"


@dataclass
class AWSConfig:
    """Configuration for AWS services."""
    region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    
    # S3 buckets (production)
    s3_raw_bucket: str = field(
        default_factory=lambda: os.getenv("S3_RAW_BUCKET", "mlops-toxic-raw-data")
    )
    s3_processed_bucket: str = field(
        default_factory=lambda: os.getenv("S3_PROCESSED_BUCKET", "mlops-toxic-processed")
    )
    s3_models_bucket: str = field(
        default_factory=lambda: os.getenv("S3_MODELS_BUCKET", "mlops-toxic-models")
    )
    s3_evidently_bucket: str = field(
        default_factory=lambda: os.getenv("S3_EVIDENTLY_BUCKET", "mlops-toxic-evidently")
    )
    
    # SageMaker configuration
    sagemaker_endpoint_name: str = field(
        default_factory=lambda: os.getenv("SAGEMAKER_ENDPOINT_NAME", "mlops-toxic-dev-endpoint")
    )
    sagemaker_role_arn: str = field(
        default_factory=lambda: os.getenv("SAGEMAKER_ROLE_ARN", "")
    )
    
    # RDS configuration
    rds_host: str = field(default_factory=lambda: os.getenv("RDS_HOST", ""))
    rds_port: int = field(default_factory=lambda: int(os.getenv("RDS_PORT", "5432")))
    rds_database: str = field(default_factory=lambda: os.getenv("RDS_DATABASE", "reviews"))
    rds_user: str = field(default_factory=lambda: os.getenv("RDS_USER", "admin"))
    rds_password: str = field(default_factory=lambda: os.getenv("RDS_PASSWORD", ""))
    db_secret_arn: str = field(
        default_factory=lambda: os.getenv("DB_SECRET_ARN", "")
    )
    
    # MLflow configuration
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "")
    )
    
    # CloudWatch configuration
    cloudwatch_namespace: str = field(
        default_factory=lambda: os.getenv("CLOUDWATCH_NAMESPACE", "ToxicCommentAPI")
    )
    
    # SNS configuration
    sns_topic_arn: str = field(
        default_factory=lambda: os.getenv("SNS_TOPIC_ARN", "")
    )
    
    # Model promotion threshold (ROC-AUC improvement required to promote)
    model_promotion_threshold: float = field(
        default_factory=lambda: float(os.getenv("MODEL_PROMOTION_THRESHOLD", "0.02"))
    )


@dataclass
class MLflowConfig:
    """Configuration for MLflow experiment tracking."""
    tracking_uri: str = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "databricks"))
    databricks_host: str = field(
        default_factory=lambda: os.getenv("DATABRICKS_HOST", "")
    )
    databricks_token: str = field(default_factory=lambda: os.getenv("DATABRICKS_TOKEN", ""))
    experiment_name: str = "/Users/krainik.mykyta@lll.kpi.ua/mlops-project"


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases experiment tracking."""
    api_key: str = field(default_factory=lambda: os.getenv("WANDB_API_KEY", ""))
    project: str = field(default_factory=lambda: os.getenv("WANDB_PROJECT", "mlops-project"))
    entity: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_ENTITY"))


@dataclass
class ModerationThresholds:
    """Thresholds for moderation decisions."""
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
    """Configuration for the Flask API."""
    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "5000")))
    model_version: str = field(default_factory=lambda: os.getenv("MODEL_VERSION", "v1.0.0"))


@dataclass
class ReviewConfig:
    """Configuration for the review system."""
    host: str = field(default_factory=lambda: os.getenv("REVIEW_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("REVIEW_PORT", "5001")))
    auth_username: str = field(default_factory=lambda: os.getenv("REVIEW_AUTH_USERNAME", "admin"))
    auth_password: str = field(default_factory=lambda: os.getenv("REVIEW_AUTH_PASSWORD", "changeme"))


@dataclass
class ModelConfig:
    """Configuration for the ML model."""
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
class MonitoringConfig:
    """Configuration for monitoring and alerting."""
    enable_cloudwatch: bool = field(
        default_factory=lambda: os.getenv("ENABLE_CLOUDWATCH", "false").lower() == "true"
    )
    enable_evidently: bool = field(
        default_factory=lambda: os.getenv("ENABLE_EVIDENTLY", "false").lower() == "true"
    )
    drift_threshold: float = field(
        default_factory=lambda: float(os.getenv("DRIFT_THRESHOLD", "0.15"))
    )
    metrics_flush_interval: int = field(
        default_factory=lambda: int(os.getenv("METRICS_FLUSH_INTERVAL", "60"))
    )


@dataclass
class Config:
    """Main configuration container."""
    minio: MinioConfig = field(default_factory=MinioConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    moderation: ModerationThresholds = field(default_factory=ModerationThresholds)
    api: APIConfig = field(default_factory=APIConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


config = Config()
