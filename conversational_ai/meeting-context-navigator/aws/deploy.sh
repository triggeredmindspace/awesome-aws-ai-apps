#!/bin/bash
# Deployment script for Meeting Context Navigator

set -e

echo "Deploying Meeting Context Navigator to AWS..."

# Set AWS region
export AWS_REGION=${AWS_REGION:-us-east-1}

# Deploy CloudFormation stack
aws cloudformation deploy \
    --template-file cloudformation/template.yaml \
    --stack-name meeting-context-navigator \
    --capabilities CAPABILITY_IAM \
    --region $AWS_REGION

echo "Deployment complete!"
