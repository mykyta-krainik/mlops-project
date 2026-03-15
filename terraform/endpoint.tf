# SageMaker Model Package Group (model registry)
resource "aws_sagemaker_model_package_group" "toxic" {
  model_package_group_name        = "${var.project}-models"
  model_package_group_description = "Toxic comment classifier model versions"
}

# ── Placeholder model (bootstrap) ─────────────────────────────────────────────
# The promote.py step registers real model versions at pipeline runtime.
# Terraform manages the endpoint lifecycle; model versions are managed by the pipeline.
#
# Before running terraform apply for the first time you need at least one model
# package in Approved status. The pipeline's promote step handles this on first run.
# We create the endpoints pointing at a dummy config that the first pipeline run
# will immediately update.

data "aws_caller_identity" "endpoint" {}
data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.endpoint.account_id
  region     = data.aws_region.current.name
}

# Data capture config for production endpoint (feeds Evidently drift detection)
resource "aws_sagemaker_endpoint_configuration" "prod_initial" {
  name = "${var.project}-prod-initial"

  production_variants {
    variant_name           = "blue"
    model_name             = aws_sagemaker_model.placeholder.name
    instance_type          = var.sagemaker_instance_type
    initial_instance_count = 1
    initial_variant_weight = 1
  }

  data_capture_config {
    enable_capture              = true
    initial_sampling_percentage = 100
    destination_s3_uri          = "s3://${aws_s3_bucket.buckets["pipeline"].bucket}/data-capture"

    capture_options {
      capture_mode = "Input"
    }
    capture_options {
      capture_mode = "Output"
    }

    capture_content_type_header {
      json_content_types = ["application/json"]
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "staging_initial" {
  name = "${var.project}-staging-initial"

  production_variants {
    variant_name           = "blue"
    model_name             = aws_sagemaker_model.placeholder.name
    instance_type          = var.sagemaker_instance_type
    initial_instance_count = 1
    initial_variant_weight = 1
  }
}

# Placeholder SageMaker model — uses a public sklearn container so Terraform
# can create the endpoint without a real model artifact on first apply.
# The first pipeline run replaces this via update_endpoint().
resource "aws_sagemaker_model" "placeholder" {
  name               = "${var.project}-placeholder"
  execution_role_arn = aws_iam_role.sagemaker_exec.arn

  primary_container {
    # AWS SKLearn DLC — no custom image needed for bootstrap
    image          = "683313688378.dkr.ecr.${local.region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    model_data_url = "s3://${aws_s3_bucket.buckets["models"].bucket}/placeholder/model.tar.gz"

    environment = {
      SAGEMAKER_PROGRAM = "inference.py"
    }
  }
}

# Endpoints
resource "aws_sagemaker_endpoint" "staging" {
  name                 = "${var.project}-staging"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.staging_initial.name

  lifecycle {
    # Prevent Terraform from reverting endpoint config changes made by promote.py
    ignore_changes = [endpoint_config_name]
  }
}

resource "aws_sagemaker_endpoint" "prod" {
  name                 = "${var.project}-prod"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.prod_initial.name

  lifecycle {
    ignore_changes = [endpoint_config_name]
  }
}
