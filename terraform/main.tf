terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    # Bootstrap: create this bucket manually once before running terraform init.
    # e.g. aws s3 mb s3://mlops-toxic-tf-state --region us-east-1
    bucket         = "mlops-toxic-tf-state"
    key            = "mlops-project/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "mlops-toxic-tf-lock"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# ── Remote state lock table ────────────────────────────────────────────────────
resource "aws_dynamodb_table" "tf_lock" {
  name         = "mlops-toxic-tf-lock"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}

# ── S3 buckets ─────────────────────────────────────────────────────────────────
locals {
  buckets = {
    raw       = "${var.project}-raw"
    processed = "${var.project}-processed"
    models    = "${var.project}-models"
    pipeline  = "${var.project}-pipeline"
  }
}

resource "aws_s3_bucket" "buckets" {
  for_each = local.buckets
  bucket   = each.value
}

resource "aws_s3_bucket_versioning" "buckets" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.buckets[each.key].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "buckets" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.buckets[each.key].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "buckets" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.buckets[each.key].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Keep only last 5 non-current model versions to save cost
resource "aws_s3_bucket_lifecycle_configuration" "models" {
  bucket = aws_s3_bucket.buckets["models"].id

  rule {
    id     = "expire-old-models"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

# ── ECR repository ─────────────────────────────────────────────────────────────
resource "aws_ecr_repository" "mlops" {
  name                 = var.project
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "mlops" {
  repository = aws_ecr_repository.mlops.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# ── IAM: SageMaker execution role ─────────────────────────────────────────────
data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_exec" {
  name               = "${var.project}-sagemaker-exec"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  role       = aws_iam_role.sagemaker_exec.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

data "aws_iam_policy_document" "sagemaker_s3" {
  statement {
    actions = [
      "s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket",
    ]
    resources = [
      for b in aws_s3_bucket.buckets : "${b.arn}/*"
    ]
  }
  statement {
    actions   = ["s3:ListBucket"]
    resources = [for b in aws_s3_bucket.buckets : b.arn]
  }
  statement {
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]
  }
  statement {
    actions   = ["secretsmanager:GetSecretValue"]
    resources = ["arn:aws:secretsmanager:${var.aws_region}:*:secret:${var.project}/*"]
  }
  statement {
    actions   = ["ecr:GetAuthorizationToken", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "sagemaker_s3" {
  name   = "${var.project}-sagemaker-s3"
  policy = data.aws_iam_policy_document.sagemaker_s3.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_s3" {
  role       = aws_iam_role.sagemaker_exec.name
  policy_arn = aws_iam_policy.sagemaker_s3.arn
}

# ── IAM: GitHub Actions OIDC ──────────────────────────────────────────────────
data "aws_caller_identity" "current" {}

resource "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"

  client_id_list = ["sts.amazonaws.com"]

  # GitHub's OIDC thumbprint (stable)
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

data "aws_iam_policy_document" "github_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_org}/${var.github_repo}:*"]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "github_actions" {
  name               = "${var.project}-github-actions"
  assume_role_policy = data.aws_iam_policy_document.github_assume.json
}

data "aws_iam_policy_document" "github_actions" {
  statement {
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload",
      "ecr:PutImage",
    ]
    resources = ["*"]
  }
  statement {
    actions   = ["sagemaker:StartPipelineExecution", "sagemaker:DescribePipelineExecution"]
    resources = ["arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:pipeline/${var.project}-*"]
  }
  statement {
    actions   = ["sagemaker:UpdateEndpoint", "sagemaker:DescribeEndpoint"]
    resources = ["arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:endpoint/${var.project}-*"]
  }
  statement {
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [for b in aws_s3_bucket.buckets : "${b.arn}/*"]
  }
  statement {
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.sagemaker_exec.arn]
  }
  statement {
    actions   = ["sagemaker:CreatePipeline", "sagemaker:UpdatePipeline", "sagemaker:DescribePipeline"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "github_actions" {
  name   = "${var.project}-github-actions"
  policy = data.aws_iam_policy_document.github_actions.json
}

resource "aws_iam_role_policy_attachment" "github_actions" {
  role       = aws_iam_role.github_actions.name
  policy_arn = aws_iam_policy.github_actions.arn
}
