#!/bin/bash
# Deployment script for LiveStream Sentiment Pulse Analytics

set -e

echo "Deploying LiveStream Sentiment Pulse Analytics to AWS..."

# Set AWS region
export AWS_REGION=${AWS_REGION:-us-east-1}

# Deploy CloudFormation stack
aws cloudformation deploy \
    --template-file cloudformation/template.yaml \
    --stack-name livestream-sentiment-pulse-analytics \
    --capabilities CAPABILITY_IAM \
    --region $AWS_REGION

echo "Deployment complete!"
