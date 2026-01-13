set -e

PROJECT_NAME="${PROJECT_NAME:-mlops-toxic}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
AWS_REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

TARGET="${1:-all}"

echo "Building containers for ${PROJECT_NAME}-${ENVIRONMENT}"
echo "Region: ${AWS_REGION}"
echo "Account: ${ACCOUNT_ID}"
echo ""

# Login to ECR
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin \
    "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

build_training() {
    echo "Building training container..."
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories \
        --repository-names "${PROJECT_NAME}-${ENVIRONMENT}-training" \
        --region "${AWS_REGION}" 2>/dev/null || \
    aws ecr create-repository \
        --repository-name "${PROJECT_NAME}-${ENVIRONMENT}-training" \
        --region "${AWS_REGION}"
    
    # Build and push
    TRAINING_IMAGE="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}-${ENVIRONMENT}-training:latest"
    docker build -t "${TRAINING_IMAGE}" sagemaker/training/
    docker push "${TRAINING_IMAGE}"
    
    echo "Training container pushed: ${TRAINING_IMAGE}"
}

build_inference() {
    echo "Building inference container..."
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories \
        --repository-names "${PROJECT_NAME}-${ENVIRONMENT}-inference" \
        --region "${AWS_REGION}" 2>/dev/null || \
    aws ecr create-repository \
        --repository-name "${PROJECT_NAME}-${ENVIRONMENT}-inference" \
        --region "${AWS_REGION}"
    
    # Build and push
    INFERENCE_IMAGE="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}-${ENVIRONMENT}-inference:latest"
    docker build -t "${INFERENCE_IMAGE}" sagemaker/inference/
    docker push "${INFERENCE_IMAGE}"
    
    echo "Inference container pushed: ${INFERENCE_IMAGE}"
}

case "${TARGET}" in
    training)
        build_training
        ;;
    inference)
        build_inference
        ;;
    all)
        build_training
        build_inference
        ;;
    *)
        echo "Usage: $0 [training|inference|all]"
        exit 1
        ;;
esac

echo ""
echo "Container build complete!"
