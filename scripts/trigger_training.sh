set -e

PROJECT_NAME="${PROJECT_NAME:-mlops-toxic}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
AWS_REGION="${AWS_REGION:-eu-central-1}"

FUNCTION_NAME="${PROJECT_NAME}-${ENVIRONMENT}-data-preparation"

echo "Triggering training job via Lambda: ${FUNCTION_NAME}"

aws lambda invoke \
    --function-name "${FUNCTION_NAME}" \
    --region "${AWS_REGION}" \
    --payload '{"source": "manual-trigger"}' \
    --cli-binary-format raw-in-base64-out \
    response.json

echo ""
echo "Response:"
cat response.json
echo ""
rm response.json

echo ""
echo "Training job triggered. Check SageMaker console for progress."
echo "Console: https://${AWS_REGION}.console.aws.amazon.com/sagemaker/home?region=${AWS_REGION}#/jobs"
