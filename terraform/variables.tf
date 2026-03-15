variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Project name prefix for all resources"
  type        = string
  default     = "mlops-toxic"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "prod"
}

variable "github_org" {
  description = "GitHub organisation or user name (for OIDC trust)"
  type        = string
}

variable "github_repo" {
  description = "GitHub repository name (for OIDC trust)"
  type        = string
}

variable "alert_email" {
  description = "Email address to receive CloudWatch alarm notifications"
  type        = string
}

variable "sagemaker_instance_type" {
  description = "Instance type for SageMaker endpoints"
  type        = string
  default     = "ml.t3.medium"
}

variable "drift_threshold" {
  description = "Share of drifted columns that triggers a retrain alarm (0–1)"
  type        = number
  default     = 0.3
}
