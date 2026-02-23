#!/bin/bash
# Deployment script for Smart Invoice Intelligence Platform

set -e

echo "Deploying Smart Invoice Intelligence Platform to AWS..."

# Set AWS region
export AWS_REGION=${AWS_REGION:-us-east-1}

# Deploy CloudFormation stack
aws cloudformation deploy \
    --template-file cloudformation/template.yaml \
    --stack-name smart-invoice-intelligence-platform \
    --capabilities CAPABILITY_IAM \
    --region $AWS_REGION

echo "Deployment complete!"
