# ── SNS: alert topic ───────────────────────────────────────────────────────────
resource "aws_sns_topic" "alerts" {
  name = "${var.project}-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ── CloudWatch: log groups ─────────────────────────────────────────────────────
resource "aws_cloudwatch_log_group" "staging_endpoint" {
  name              = "/aws/sagemaker/Endpoints/${var.project}-staging"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "prod_endpoint" {
  name              = "/aws/sagemaker/Endpoints/${var.project}-prod"
  retention_in_days = 90
}

# ── CloudWatch: alarms ─────────────────────────────────────────────────────────
resource "aws_cloudwatch_metric_alarm" "prod_high_latency" {
  alarm_name          = "${var.project}-prod-high-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ModelLatency"
  namespace           = "AWS/SageMaker"
  period              = 60
  statistic           = "p99"
  threshold           = 500000 # microseconds → 500ms
  alarm_description   = "Prod endpoint p99 latency exceeded 500ms"
  treat_missing_data  = "notBreaching"

  dimensions = {
    EndpointName = "${var.project}-prod"
    VariantName  = "blue"
  }

  alarm_actions = [aws_sns_topic.alerts.arn, aws_lambda_function.rollback.arn]
  ok_actions    = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "prod_high_error_rate" {
  alarm_name          = "${var.project}-prod-high-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  threshold           = 1
  alarm_description   = "Prod endpoint error rate exceeded 1%"
  treat_missing_data  = "notBreaching"

  metric_query {
    id          = "error_rate"
    expression  = "100 * (e4xx + e5xx) / MAX([e4xx + e5xx, invocations])"
    label       = "ErrorRate"
    return_data = true
  }

  metric_query {
    id = "e4xx"
    metric {
      metric_name = "Invocation4XXErrors"
      namespace   = "AWS/SageMaker"
      period      = 60
      stat        = "Sum"
      dimensions = {
        EndpointName = "${var.project}-prod"
        VariantName  = "blue"
      }
    }
  }

  metric_query {
    id = "e5xx"
    metric {
      metric_name = "Invocation5XXErrors"
      namespace   = "AWS/SageMaker"
      period      = 60
      stat        = "Sum"
      dimensions = {
        EndpointName = "${var.project}-prod"
        VariantName  = "blue"
      }
    }
  }

  metric_query {
    id = "invocations"
    metric {
      metric_name = "Invocations"
      namespace   = "AWS/SageMaker"
      period      = 60
      stat        = "Sum"
      dimensions = {
        EndpointName = "${var.project}-prod"
        VariantName  = "blue"
      }
    }
  }

  alarm_actions = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "drift_detected" {
  alarm_name          = "${var.project}-drift-detected"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "DriftScore"
  namespace           = "${var.project}/monitoring"
  period              = 86400 # daily
  statistic           = "Maximum"
  threshold           = var.drift_threshold
  alarm_description   = "Data drift score exceeded threshold — triggering retrain"
  treat_missing_data  = "notBreaching"

  alarm_actions = [aws_lambda_function.retrain_trigger.arn]
}

# ── CloudWatch: dashboard ─────────────────────────────────────────────────────
resource "aws_cloudwatch_dashboard" "mlops" {
  dashboard_name = "${var.project}-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric"
        properties = {
          title  = "Invocations (prod)"
          period = 60
          stat   = "Sum"
          metrics = [
            ["AWS/SageMaker", "Invocations", "EndpointName", "${var.project}-prod", "VariantName", "blue"]
          ]
        }
      },
      {
        type = "metric"
        properties = {
          title  = "Model Latency p50/p99 (prod, ms)"
          period = 60
          metrics = [
            ["AWS/SageMaker", "ModelLatency", "EndpointName", "${var.project}-prod", "VariantName", "blue", { stat = "p50", label = "p50" }],
            ["AWS/SageMaker", "ModelLatency", "EndpointName", "${var.project}-prod", "VariantName", "blue", { stat = "p99", label = "p99" }]
          ]
        }
      },
      {
        type = "metric"
        properties = {
          title  = "4XX / 5XX Errors (prod)"
          period = 60
          stat   = "Sum"
          metrics = [
            ["AWS/SageMaker", "Invocation4XXErrors", "EndpointName", "${var.project}-prod", "VariantName", "blue"],
            ["AWS/SageMaker", "Invocation5XXErrors", "EndpointName", "${var.project}-prod", "VariantName", "blue"]
          ]
        }
      },
      {
        type = "metric"
        properties = {
          title  = "Drift Score (daily)"
          period = 86400
          stat   = "Maximum"
          metrics = [
            ["${var.project}/monitoring", "DriftScore"]
          ]
        }
      }
    ]
  })
}

# ── Lambda: rollback on high latency ──────────────────────────────────────────
data "archive_file" "rollback" {
  type        = "zip"
  output_path = "${path.module}/.lambda/rollback.zip"

  source {
    content  = <<-PYTHON
import boto3
import os

sm = boto3.client("sagemaker")

def handler(event, context):
    endpoint_name = os.environ["ENDPOINT_NAME"]
    fallback_config = os.environ["FALLBACK_CONFIG"]
    resp = sm.describe_endpoint(EndpointName=endpoint_name)
    current = resp["EndpointConfigName"]
    if current == fallback_config:
        print(f"Already on fallback config {fallback_config}, nothing to do")
        return
    print(f"Rolling back {endpoint_name}: {current} → {fallback_config}")
    sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=fallback_config)
PYTHON
    filename = "lambda_function.py"
  }
}

resource "aws_iam_role" "lambda_exec" {
  name = "${var.project}-lambda-exec"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "lambda_sagemaker" {
  statement {
    actions   = ["sagemaker:UpdateEndpoint", "sagemaker:DescribeEndpoint", "sagemaker:StartPipelineExecution"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "lambda_sagemaker" {
  name   = "${var.project}-lambda-sagemaker"
  policy = data.aws_iam_policy_document.lambda_sagemaker.json
}

resource "aws_iam_role_policy_attachment" "lambda_sagemaker" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = aws_iam_policy.lambda_sagemaker.arn
}

resource "aws_lambda_function" "rollback" {
  function_name    = "${var.project}-rollback"
  role             = aws_iam_role.lambda_exec.arn
  handler          = "lambda_function.handler"
  runtime          = "python3.12"
  filename         = data.archive_file.rollback.output_path
  source_code_hash = data.archive_file.rollback.output_base64sha256
  timeout          = 60

  environment {
    variables = {
      ENDPOINT_NAME   = "${var.project}-prod"
      FALLBACK_CONFIG = "${var.project}-prod-initial"
    }
  }
}

resource "aws_lambda_permission" "rollback_cloudwatch" {
  statement_id  = "AllowCloudWatchAlarm"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.rollback.function_name
  principal     = "lambda.alarms.cloudwatch.amazonaws.com"
  source_arn    = aws_cloudwatch_metric_alarm.prod_high_latency.arn
}

# ── Lambda: retrain trigger ────────────────────────────────────────────────────
data "archive_file" "retrain_trigger" {
  type        = "zip"
  output_path = "${path.module}/.lambda/retrain_trigger.zip"

  source {
    content  = <<-PYTHON
import boto3
import os

sm = boto3.client("sagemaker")

def handler(event, context):
    pipeline_name = os.environ["PIPELINE_NAME"]
    print(f"Triggering retrain pipeline: {pipeline_name}")
    resp = sm.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineExecutionDisplayName="auto-retrain-drift",
    )
    print(f"Pipeline execution ARN: {resp['PipelineExecutionArn']}")
    return resp["PipelineExecutionArn"]
PYTHON
    filename = "lambda_function.py"
  }
}

resource "aws_lambda_function" "retrain_trigger" {
  function_name    = "${var.project}-retrain-trigger"
  role             = aws_iam_role.lambda_exec.arn
  handler          = "lambda_function.handler"
  runtime          = "python3.12"
  filename         = data.archive_file.retrain_trigger.output_path
  source_code_hash = data.archive_file.retrain_trigger.output_base64sha256
  timeout          = 60

  environment {
    variables = {
      PIPELINE_NAME = "mlops-toxic-pipeline"
    }
  }
}

# ── EventBridge: new data in S3 → retrain ─────────────────────────────────────
resource "aws_cloudwatch_event_rule" "new_data" {
  name        = "${var.project}-new-data"
  description = "Trigger retrain when new raw data lands in S3"

  event_pattern = jsonencode({
    source      = ["aws.s3"]
    detail-type = ["Object Created"]
    detail = {
      bucket = { name = [aws_s3_bucket.buckets["raw"].bucket] }
    }
  })
}

resource "aws_cloudwatch_event_target" "new_data_lambda" {
  rule      = aws_cloudwatch_event_rule.new_data.name
  target_id = "retrain-trigger"
  arn       = aws_lambda_function.retrain_trigger.arn
}

resource "aws_lambda_permission" "retrain_eventbridge" {
  statement_id  = "AllowEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.retrain_trigger.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.new_data.arn
}

# Enable S3 EventBridge notifications on the raw bucket
resource "aws_s3_bucket_notification" "raw_eventbridge" {
  bucket      = aws_s3_bucket.buckets["raw"].id
  eventbridge = true
}
