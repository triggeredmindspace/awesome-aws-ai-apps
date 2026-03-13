#!/usr/bin/env bash
# deploy.sh — Deploy the Product Review Intelligence Platform to AWS
# Usage:  ./aws/deploy.sh [--env dev|staging|prod] [--region us-east-1] [--email you@example.com]
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
ENVIRONMENT="dev"
AWS_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
DIGEST_EMAIL=""
STACK_NAME="product-review-intelligence"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --env)       ENVIRONMENT="$2"; shift 2 ;;
    --region)    AWS_REGION="$2";  shift 2 ;;
    --email)     DIGEST_EMAIL="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

FULL_STACK_NAME="${STACK_NAME}-${ENVIRONMENT}"

echo "========================================"
echo " Product Review Intelligence — Deploy"
echo "========================================"
echo "  Environment : ${ENVIRONMENT}"
echo "  Region      : ${AWS_REGION}"
echo "  Stack       : ${FULL_STACK_NAME}"
echo "  Digest email: ${DIGEST_EMAIL:-<disabled>}"
echo "========================================"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
command -v aws  >/dev/null 2>&1 || { echo "ERROR: aws CLI not found"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }
command -v zip  >/dev/null 2>&1 || { echo "ERROR: zip not found"; exit 1; }

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$AWS_REGION")
DEPLOYMENT_BUCKET="product-review-deploy-${AWS_ACCOUNT_ID}-${AWS_REGION}"

# ── Create deployment bucket if it doesn't exist ──────────────────────────────
echo ""
echo ">>> Ensuring deployment bucket: ${DEPLOYMENT_BUCKET}"
if ! aws s3api head-bucket --bucket "$DEPLOYMENT_BUCKET" --region "$AWS_REGION" 2>/dev/null; then
  if [[ "$AWS_REGION" == "us-east-1" ]]; then
    aws s3api create-bucket \
      --bucket "$DEPLOYMENT_BUCKET" \
      --region "$AWS_REGION"
  else
    aws s3api create-bucket \
      --bucket "$DEPLOYMENT_BUCKET" \
      --region "$AWS_REGION" \
      --create-bucket-configuration LocationConstraint="$AWS_REGION"
  fi
  echo "    Bucket created."
else
  echo "    Bucket already exists."
fi

# ── Package Lambda ────────────────────────────────────────────────────────────
echo ""
echo ">>> Packaging Lambda function"
BUILD_DIR=$(mktemp -d)
trap 'rm -rf "$BUILD_DIR"' EXIT

pip install \
  --quiet \
  --target "$BUILD_DIR" \
  --requirement "$APP_DIR/requirements.txt"

cp "$APP_DIR/app.py" "$BUILD_DIR/"
cp "$APP_DIR/config.yaml" "$BUILD_DIR/"

LAMBDA_ZIP="$BUILD_DIR/lambda.zip"
(cd "$BUILD_DIR" && zip -q -r "$LAMBDA_ZIP" .)
echo "    Lambda package: $(du -sh "$LAMBDA_ZIP" | cut -f1)"

# ── Upload Lambda package ─────────────────────────────────────────────────────
S3_KEY="product-review-intelligence/lambda.zip"
echo ""
echo ">>> Uploading Lambda package to s3://${DEPLOYMENT_BUCKET}/${S3_KEY}"
aws s3 cp "$LAMBDA_ZIP" "s3://${DEPLOYMENT_BUCKET}/${S3_KEY}" \
  --region "$AWS_REGION" \
  --quiet
echo "    Upload complete."

# ── Deploy CloudFormation stack ───────────────────────────────────────────────
echo ""
echo ">>> Deploying CloudFormation stack: ${FULL_STACK_NAME}"
PARAMS=(
  "ParameterKey=Environment,ParameterValue=${ENVIRONMENT}"
  "ParameterKey=DeploymentBucketParam,ParameterValue=${DEPLOYMENT_BUCKET}"
)
[[ -n "$DIGEST_EMAIL" ]] && PARAMS+=("ParameterKey=DigestEmail,ParameterValue=${DIGEST_EMAIL}")

aws cloudformation deploy \
  --stack-name "$FULL_STACK_NAME" \
  --template-file "$APP_DIR/aws/cloudformation/template.yaml" \
  --parameter-overrides "${PARAMS[@]}" \
  --capabilities CAPABILITY_NAMED_IAM \
  --region "$AWS_REGION" \
  --no-fail-on-empty-changeset

# ── Print outputs ─────────────────────────────────────────────────────────────
echo ""
echo ">>> Stack outputs"
aws cloudformation describe-stacks \
  --stack-name "$FULL_STACK_NAME" \
  --region "$AWS_REGION" \
  --query "Stacks[0].Outputs[*].[OutputKey,OutputValue]" \
  --output table

echo ""
echo "========================================"
echo " Deployment complete!"
echo "========================================"
