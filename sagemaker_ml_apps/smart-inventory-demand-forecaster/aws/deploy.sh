#!/bin/bash
# Deployment script for Smart Inventory Demand Forecaster

set -e

echo "Deploying Smart Inventory Demand Forecaster to AWS..."

# Set AWS region
export AWS_REGION=${AWS_REGION:-us-east-1}

# Deploy CloudFormation stack
aws cloudformation deploy \
    --template-file cloudformation/template.yaml \
    --stack-name smart-inventory-demand-forecaster \
    --capabilities CAPABILITY_IAM \
    --region $AWS_REGION

echo "Deployment complete!"
