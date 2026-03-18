# Toxic Comment Classifier — MLOps Project

End-to-end MLOps system for detecting toxic comments. Trains a TF-IDF + Logistic Regression model, packages it as ONNX, and serves it through a SageMaker real-time endpoint with automated retraining, drift monitoring, and canary deployments.

## Architecture

```
GitHub Push → CI (lint/test) → Build Docker → Push ECR
                                                    ↓
                              SageMaker Pipeline (Ingest → Preprocess → Train × 2 → Evaluate → Promote)
                                                    ↓
                              Staging Endpoint (canary: 80% blue / 20% green)
                                                    ↓
                              Locust Load Test → SLO check → Production Endpoint
```

**AWS services:** S3, ECR, SageMaker Pipelines, SageMaker Endpoints, CloudWatch, Lambda, EventBridge, CloudFormation
**MLflow tracking:** Databricks (optional — non-fatal if unavailable)

---

## Prerequisites

- Python 3.12
- Docker + Docker Compose
- AWS CLI v2 configured with a profile that has access to `us-east-1`
- An AWS account with permissions for SageMaker, S3, ECR, CloudFormation, IAM, Lambda

---

## Quick Start — Local Development

```bash
# 1. Clone and configure environment
cp .env.example .env
# Edit .env — at minimum set MINIO_* and API_* values

# 2. Start local services (MinIO, Airflow, Flask API, PostgreSQL)
make up

# 3. Train a model locally and upload to MinIO
make train

# 4. Test the local API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"comment": "you are an idiot"}'

# 5. Stop services
make down
```

Local services:
| Service | URL | Credentials |
|---------|-----|-------------|
| Flask API | http://localhost:5000 | — |
| MinIO console | http://localhost:9001 | minioadmin / minioadmin |
| Airflow UI | http://localhost:8080 | admin / admin |

---

## AWS Deployment

### 1. Bootstrap infrastructure

```bash
export GITHUB_ORG=your-github-org
export GITHUB_REPO=your-repo-name
export ALERT_EMAIL=your@email.com

make cf-deploy
```

This creates: S3 buckets, ECR repository, IAM roles (including GitHub OIDC), SageMaker endpoint placeholders, CloudWatch alarms, Lambda + EventBridge triggers.

### 2. Upload training data

```bash
# Place your Kaggle toxic comments CSV at data/raw/train.csv, then:
make upload-data
```

Or upload directly:
```bash
python scripts/upload_data.py \
  --local-path data/raw/train.csv \
  --bucket mlops-toxic-raw \
  --key train/train.csv
```

### 3. Build and push the Docker image

```bash
make ecr-push
```

### 4. Run the SageMaker pipeline

```bash
make sm-pipeline-run
```

Pipeline steps: `IngestStep → PreprocessStep → TrainBaselineStep + TrainImprovedStep → EvaluateStep → ConditionStep → PromoteStep`

The `PromoteStep` deploys the winning model to the staging endpoint with an 80/20 canary split.

### 5. Test the staging endpoint

```bash
echo '{"comment": "you are an idiot"}' > /tmp/req.json

aws sagemaker-runtime invoke-endpoint \
  --endpoint-name toxic-comment-staging \
  --content-type application/json \
  --body fileb:///tmp/req.json \
  --region us-east-1 \
  /tmp/response.json && cat /tmp/response.json
```

Expected response:
```json
{
  "comment": "you are an idiot",
  "predictions": {"toxic": 0.91, "severe_toxic": 0.12, ...},
  "is_toxic": true,
  "moderation_action": "BAN",
  "model_version": "sagemaker"
}
```

### 6. Promote to production

After the load test passes:
```bash
python src/promote.py --to-prod
```

Or trigger via the `deploy.yml` workflow (runs automatically after `make sm-pipeline-run` in CI).

---

## CI/CD

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `ci.yml` | Push / PR | Lint (ruff), test (pytest), validate CloudFormation |
| `build.yml` | Push to `main` | Build + push Docker image to ECR, call deploy.yml |
| `deploy.yml` | Called by build or manual | Update stack → run pipeline → load test → promote to prod |
| `schedule.yml` | Weekly Mon 02:00 UTC | Retrain pipeline |
| `schedule.yml` | Daily 08:00 UTC | Drift check + SLO check |

GitHub Actions uses OIDC (no stored AWS keys). The `GitHubActionsRole` is created by CloudFormation.

---

## API Endpoints

The Flask API (local) and SageMaker endpoints share the same request/response format.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model status |
| `/model/info` | GET | Model version, labels, source |
| `/predict` | POST | Single comment classification |
| `/predict/batch` | POST | Batch classification (max 100) |
| `/model/reload` | POST | Reload model from MinIO |

**Single prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"comment": "Have a great day!"}'
```

**Batch prediction:**
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"comments": ["Hello!", "You are terrible", "Nice work"]}'
```

---

## Monitoring

```bash
# Drift detection (Evidently — compares current inference data to training reference)
make drift-check

# SLO check (CloudWatch — P95 latency < 500ms, error rate < 1%)
make slo-check

# Load test (Locust — 100 users, 2 minutes against staging)
make load-test
```

Scheduled monitoring runs daily at 08:00 UTC. Failures automatically open GitHub Issues.

---

## Configuration

All config is in `src/config.py` and read from environment variables. Key variables:

```bash
# AWS
AWS_DEFAULT_REGION=us-east-1
AWS_PIPELINE_BUCKET=mlops-toxic-pipeline
SAGEMAKER_ROLE_ARN=arn:aws:iam::ACCOUNT:role/SageMakerExecRole
SAGEMAKER_ECR_IMAGE_URI=ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mlops-toxic:latest
SAGEMAKER_INSTANCE_TYPE=ml.m5.large
SAGEMAKER_STAGING_ENDPOINT=toxic-comment-staging
SAGEMAKER_PROD_ENDPOINT=toxic-comment-prod

# MLflow / Databricks (optional)
MLFLOW_TRACKING_URI=databricks
DATABRICKS_HOST=https://your-workspace.azuredatabricks.net
DATABRICKS_TOKEN=dapi...
MLFLOW_EXPERIMENT_NAME=/Shared/mlops-project

# Moderation thresholds
THRESHOLD_BAN_SEVERE_TOXIC=0.7
THRESHOLD_BAN_THREAT=0.6
THRESHOLD_BAN_TOXIC=0.85
THRESHOLD_REVIEW_MIN=0.5
```

See `.env.example` for the full list.

---

## Development

```bash
# Run tests
make test

# Lint
make lint

# Run a single test file
pytest tests/test_preprocessing.py -v
```

Tests cover: preprocessing, model training, API endpoints, pipeline steps.

---

## Useful Commands

```bash
make help          # Full command reference
make cf-outputs    # Show CloudFormation stack outputs (bucket names, ARNs, etc.)
make cf-delete     # Tear down AWS infrastructure (endpoints are retained)
make clean         # Remove local Docker volumes
```
