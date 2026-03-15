output "ecr_repository_url" {
  description = "ECR repository URL for the training/inference image"
  value       = aws_ecr_repository.mlops.repository_url
}

output "sagemaker_role_arn" {
  description = "IAM role ARN to pass as SAGEMAKER_ROLE_ARN"
  value       = aws_iam_role.sagemaker_exec.arn
}

output "github_actions_role_arn" {
  description = "IAM role ARN to use in GitHub Actions OIDC (aws-actions/configure-aws-credentials)"
  value       = aws_iam_role.github_actions.arn
}

output "s3_raw_bucket" {
  value = aws_s3_bucket.buckets["raw"].bucket
}

output "s3_processed_bucket" {
  value = aws_s3_bucket.buckets["processed"].bucket
}

output "s3_models_bucket" {
  value = aws_s3_bucket.buckets["models"].bucket
}

output "s3_pipeline_bucket" {
  value = aws_s3_bucket.buckets["pipeline"].bucket
}

output "staging_endpoint_name" {
  description = "SageMaker staging endpoint name"
  value       = aws_sagemaker_endpoint.staging.name
}

output "prod_endpoint_name" {
  description = "SageMaker production endpoint name"
  value       = aws_sagemaker_endpoint.prod.name
}

output "sns_alert_topic_arn" {
  value = aws_sns_topic.alerts.arn
}
