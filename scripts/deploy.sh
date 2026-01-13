#!/bin/bash
# Deployment script for Toxic Comment Classification MLOps Infrastructure
#
# This script deploys the complete infrastructure to AWS using CloudFormation.
# It handles:
# - Creating S3 bucket for CloudFormation templates
# - Building and pushing Docker images to ECR
# - Deploying CloudFormation stacks in the correct order
# - Uploading initial training data
#
# Prerequisites:
# - AWS CLI configured with appropriate credentials
# - Docker installed and running
# - jq installed for JSON parsing

set -e

# Configuration
PROJECT_NAME="${PROJECT_NAME:-mlops-toxic}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
AWS_REGION="${AWS_REGION:-eu-central-1}"
ALERT_EMAIL="${ALERT_EMAIL:-krainik.mykyta@lll.kpi.ua}"

# Derived names
STACK_NAME="${PROJECT_NAME}-${ENVIRONMENT}"
TEMPLATES_BUCKET="${PROJECT_NAME}-${ENVIRONMENT}-cloudformation-templates"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "=============================================="
echo "Deploying Toxic Comment Classification MLOps"
echo "=============================================="
echo "Project: ${PROJECT_NAME}"
echo "Environment: ${ENVIRONMENT}"
echo "Region: ${AWS_REGION}"
echo "Account: ${ACCOUNT_ID}"
echo "=============================================="

# Function to wait for stack completion
wait_for_stack() {
    local stack_name=$1
    echo "Waiting for stack ${stack_name} to complete..."
    
    aws cloudformation wait stack-create-complete \
        --stack-name "${stack_name}" \
        --region "${AWS_REGION}" 2>/dev/null || \
    aws cloudformation wait stack-update-complete \
        --stack-name "${stack_name}" \
        --region "${AWS_REGION}" 2>/dev/null || true
    
    local status=$(aws cloudformation describe-stacks \
        --stack-name "${stack_name}" \
        --region "${AWS_REGION}" \
        --query 'Stacks[0].StackStatus' \
        --output text)
    
    if [[ "${status}" == *"FAILED"* ]] || [[ "${status}" == *"ROLLBACK"* ]]; then
        echo "Stack ${stack_name} failed with status: ${status}"
        exit 1
    fi
    
    echo "Stack ${stack_name} completed with status: ${status}"
}

# Step 1: Create S3 bucket for CloudFormation templates
echo ""
echo "Step 1: Creating S3 bucket for templates..."
if aws s3 ls "s3://${TEMPLATES_BUCKET}" 2>&1 | grep -q 'NoSuchBucket'; then
    aws s3 mb "s3://${TEMPLATES_BUCKET}" --region "${AWS_REGION}"
    aws s3api put-bucket-versioning \
        --bucket "${TEMPLATES_BUCKET}" \
        --versioning-configuration Status=Enabled
fi

# Step 2: Upload CloudFormation templates
echo ""
echo "Step 2: Uploading CloudFormation templates..."
aws s3 sync cloudformation/ "s3://${TEMPLATES_BUCKET}/cloudformation/" \
    --region "${AWS_REGION}"

# Step 2.5: Deploy registry stack (ECR repositories)
echo ""
echo "Step 2.5: Deploying container registry stack..."
aws cloudformation deploy \
    --template-file cloudformation/registry.yaml \
    --stack-name "${STACK_NAME}-registry" \
    --region "${AWS_REGION}" \
    --parameter-overrides \
        ProjectName="${PROJECT_NAME}" \
        Environment="${ENVIRONMENT}" \
    --no-fail-on-empty-changeset

wait_for_stack "${STACK_NAME}-registry"

# Step 3: Deploy storage stack
echo ""
echo "Step 3: Deploying storage stack..."
echo "Note: RDS will automatically generate and manage the master password in Secrets Manager"

aws cloudformation deploy \
    --template-file cloudformation/storage.yaml \
    --stack-name "${STACK_NAME}-storage" \
    --region "${AWS_REGION}" \
    --parameter-overrides \
        ProjectName="${PROJECT_NAME}" \
        Environment="${ENVIRONMENT}" \
    --capabilities CAPABILITY_NAMED_IAM \
    --no-fail-on-empty-changeset

wait_for_stack "${STACK_NAME}-storage"

# Step 4: Build and push training container
echo ""
echo "Step 4: Building and pushing training container..."

# Login to ECR (repos created by registry stack)
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin \
    "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Build and push
TRAINING_IMAGE="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}-${ENVIRONMENT}-training:latest"
docker build -t "${TRAINING_IMAGE}" sagemaker/training/
docker push "${TRAINING_IMAGE}"

# Step 5: Build and push inference container
echo ""
echo "Step 5: Building and pushing inference container..."

INFERENCE_IMAGE="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}-${ENVIRONMENT}-inference:latest"
docker build -t "${INFERENCE_IMAGE}" sagemaker/inference/
docker push "${INFERENCE_IMAGE}"

# Step 5.5: Build and push Lambda container images
echo ""
echo "Step 5.5: Building and pushing Lambda container images..."

# Data Preparation Lambda
DATA_PREP_IMAGE="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}-${ENVIRONMENT}-lambda-data-prep:latest"
echo "Building data preparation Lambda..."
docker build --no-cache -t "${DATA_PREP_IMAGE}" lambda/data_preparation/
docker push "${DATA_PREP_IMAGE}"

# Get the digest for the image we just pushed
echo "Retrieving data preparation image digest..."
DATA_PREP_DIGEST=$(aws ecr describe-images \
    --repository-name "${PROJECT_NAME}-${ENVIRONMENT}-lambda-data-prep" \
    --image-ids imageTag=latest \
    --query 'imageDetails[0].imageDigest' \
    --output text)
DATA_PREP_IMAGE_WITH_DIGEST="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}-${ENVIRONMENT}-lambda-data-prep@${DATA_PREP_DIGEST}"
echo "Data prep image digest: ${DATA_PREP_DIGEST}"

# Model Promotion Lambda
MODEL_PROMOTION_IMAGE="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}-${ENVIRONMENT}-lambda-model-promotion:latest"
echo "Building model promotion Lambda..."
docker build --no-cache -t "${MODEL_PROMOTION_IMAGE}" lambda/model_promotion/
docker push "${MODEL_PROMOTION_IMAGE}"

# Get the digest for the image we just pushed
echo "Retrieving model promotion image digest..."
MODEL_PROMOTION_DIGEST=$(aws ecr describe-images \
    --repository-name "${PROJECT_NAME}-${ENVIRONMENT}-lambda-model-promotion" \
    --image-ids imageTag=latest \
    --query 'imageDetails[0].imageDigest' \
    --output text)
MODEL_PROMOTION_IMAGE_WITH_DIGEST="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}-${ENVIRONMENT}-lambda-model-promotion@${MODEL_PROMOTION_DIGEST}"
echo "Model promotion image digest: ${MODEL_PROMOTION_DIGEST}"

# Step 6: Deploy training stack
echo ""
echo "Step 6: Deploying training stack..."
aws cloudformation deploy \
    --template-file cloudformation/training.yaml \
    --stack-name "${STACK_NAME}-training" \
    --region "${AWS_REGION}" \
    --parameter-overrides \
        ProjectName="${PROJECT_NAME}" \
        Environment="${ENVIRONMENT}" \
        TrainingImageUri="${TRAINING_IMAGE}" \
        DataPreparationImageUri="${DATA_PREP_IMAGE_WITH_DIGEST}" \
        ModelPromotionImageUri="${MODEL_PROMOTION_IMAGE_WITH_DIGEST}" \
    --capabilities CAPABILITY_NAMED_IAM \
    --no-fail-on-empty-changeset

wait_for_stack "${STACK_NAME}-training"

# Step 7: Upload initial training data (if available)
echo ""
echo "Step 7: Uploading initial training data..."
RAW_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}-storage" \
    --region "${AWS_REGION}" \
    --query 'Stacks[0].Outputs[?OutputKey==`RawDataBucketName`].OutputValue' \
    --output text)

if [ -d "data" ]; then
    aws s3 sync data/ "s3://${RAW_BUCKET}/data/" --region "${AWS_REGION}"
    echo "Training data uploaded to s3://${RAW_BUCKET}/data/"
fi

# Step 8: Run initial training (optional)
echo ""
echo "Step 8: Running initial training job..."
read -p "Run initial training job? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    aws lambda invoke \
        --function-name "${PROJECT_NAME}-${ENVIRONMENT}-data-preparation" \
        --region "${AWS_REGION}" \
        --payload '{}' \
        response.json
    
    echo "Training job triggered. Check SageMaker console for progress."
    cat response.json
    rm response.json
fi

# Step 9: Deploy serving stack (after model is available)
echo ""
echo "Step 9: Deploying serving stack..."
echo "Note: This requires a trained model in the registry."
read -p "Deploy serving stack now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    aws cloudformation deploy \
        --template-file cloudformation/serving.yaml \
        --stack-name "${STACK_NAME}-serving" \
        --region "${AWS_REGION}" \
        --parameter-overrides \
            ProjectName="${PROJECT_NAME}" \
            Environment="${ENVIRONMENT}" \
            InferenceImageUri="${INFERENCE_IMAGE}" \
        --capabilities CAPABILITY_NAMED_IAM \
        --no-fail-on-empty-changeset
    
    wait_for_stack "${STACK_NAME}-serving"
fi

# Step 10: Deploy monitoring stack
echo ""
echo "Step 10: Deploying monitoring stack..."
aws cloudformation deploy \
    --template-file cloudformation/monitoring.yaml \
    --stack-name "${STACK_NAME}-monitoring" \
    --region "${AWS_REGION}" \
    --parameter-overrides \
        ProjectName="${PROJECT_NAME}" \
        Environment="${ENVIRONMENT}" \
        AlertEmail="${ALERT_EMAIL}" \
    --capabilities CAPABILITY_NAMED_IAM \
    --no-fail-on-empty-changeset

wait_for_stack "${STACK_NAME}-monitoring"

# Step 11: Initialize database schema
echo ""
echo "Step 11: Initializing database schema..."
RDS_ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}-storage" \
    --region "${AWS_REGION}" \
    --query 'Stacks[0].Outputs[?OutputKey==`ReviewsDBEndpoint`].OutputValue' \
    --output text)

echo "RDS Endpoint: ${RDS_ENDPOINT}"
echo ""
echo "To initialize the database schema:"
echo "1. Retrieve the auto-generated password from Secrets Manager:"
echo "   aws secretsmanager get-secret-value --secret-id ${STACK_NAME}-storage-rds-credentials --query SecretString --output text | jq -r .password"
echo ""
echo "2. Run the initialization script:"
echo "   PGPASSWORD='<password-from-step-1>' psql -h ${RDS_ENDPOINT} -U dbadmin -d reviews -f scripts/init_db.sql"

# Print summary
echo ""
echo "=============================================="
echo "Deployment Complete!"
echo "=============================================="
echo ""
echo "Stacks deployed:"
echo "  - ${STACK_NAME}-registry"
echo "  - ${STACK_NAME}-storage"
echo "  - ${STACK_NAME}-training"
if [[ $REPLY =~ ^[Yy]$ ]]; then
echo "  - ${STACK_NAME}-serving"
fi
echo "  - ${STACK_NAME}-monitoring"
echo ""
echo "ECR Images:"
echo "  - ${TRAINING_IMAGE}"
echo "  - ${INFERENCE_IMAGE}"
echo "  - ${DATA_PREP_IMAGE}"
echo "  - ${MODEL_PROMOTION_IMAGE}"
echo ""
echo "Next steps:"
echo "  1. Confirm email subscription in SNS"
echo "  2. Upload training data to s3://${RAW_BUCKET}/"
echo "  3. Trigger training via Lambda or EventBridge"
echo "  4. Monitor training in SageMaker console"
echo "  5. Deploy serving stack after model is trained"
echo ""
