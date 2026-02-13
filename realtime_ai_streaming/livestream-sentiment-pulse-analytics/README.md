# 🎥 LiveStream Sentiment Pulse Analytics

A sophisticated real-time AI system that analyzes live video streams and social media feeds simultaneously to detect audience sentiment, emotional reactions, and engagement patterns during live events, broadcasts, or product launches. Empower your content strategy with instant insights and actionable alerts.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![AWS](https://img.shields.io/badge/AWS-Advanced-orange.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [AWS Setup](#aws-setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Cost Considerations](#cost-considerations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### Real-Time Analysis
- 🎬 **Live Video Stream Processing**: Analyze video content frame-by-frame for visual sentiment
- 🗣️ **Speech-to-Text Transcription**: Convert live audio to text for sentiment analysis
- 😊 **Emotion Detection**: Identify facial expressions and emotional states using AWS Rekognition
- 💬 **Social Media Integration**: Monitor Twitter, Instagram, and other platforms for audience reactions
- 📊 **Multi-Modal Sentiment Analysis**: Combine video, audio, and text sentiment for comprehensive insights

### Intelligence & Insights
- 🤖 **AI-Powered Analysis**: Leverage AWS Bedrock for advanced natural language understanding
- 📈 **Engagement Metrics**: Track viewer retention, interaction rates, and emotional peaks
- ⚡ **Real-Time Alerts**: Instant notifications when sentiment shifts or engagement drops
- 📉 **Trend Detection**: Identify emerging topics and audience interests during broadcasts
- 🎯 **Audience Segmentation**: Understand different demographic responses

### Dashboard & Reporting
- 📱 **Real-Time Dashboard**: Live visualization of sentiment trends and engagement metrics
- 📊 **Historical Analytics**: Compare performance across multiple events
- 📧 **Automated Reports**: Post-event summaries with actionable insights
- 🔔 **Custom Alerts**: Configure thresholds for sentiment changes and engagement drops

## 🏗️ Architecture

```
┌─────────────────┐         ┌──────────────────┐
│  Live Stream    │────────▶│  Amazon Kinesis  │
│  Input Source   │         │  Video Streams   │
└─────────────────┘         └──────────────────┘
                                     │
┌─────────────────┐                 ▼
│  Social Media   │         ┌──────────────────┐
│  APIs           │────────▶│  API Gateway     │
└─────────────────┘         └──────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │ Rekognition  │ │ Transcribe   │ │   Lambda     │
            │  (Emotions)  │ │  (Speech)    │ │ (Processing) │
            └──────────────┘ └──────────────┘ └──────────────┘
                    │                │                │
                    └────────────────┼────────────────┘
                                     ▼
                            ┌──────────────────┐
                            │   AWS Bedrock    │
                            │ (AI Analysis)    │
                            └──────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │  DynamoDB    │ │     ECS      │ │ EventBridge  │
            │  (Storage)   │ │ (Analytics)  │ │   (Alerts)   │
            └──────────────┘ └──────────────┘ └──────────────┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │   CloudWatch     │
                            │  (Monitoring)    │
                            └──────────────────┘
```

## 📦 Prerequisites

### Required Accounts & Services
- ✅ AWS Account with appropriate permissions
- ✅ Python 3.9 or higher
- ✅ Node.js 16+ (for frontend dashboard)
- ✅ Docker (for local development and ECS deployment)
- ✅ AWS CLI configured with credentials
- ✅ Social media API keys (Twitter, Instagram, etc.)

### AWS Service Limits
Ensure your AWS account has sufficient service limits for:
- Kinesis Video Streams
- Lambda concurrent executions
- ECS tasks
- DynamoDB read/write capacity

### Required IAM Permissions
Your AWS user/role needs permissions for:
- Kinesis (CreateStream, PutRecord, GetRecords)
- Rekognition (DetectFaces, RecognizeCelebrities)
- Transcribe (StartStreamTranscription)
- Bedrock (InvokeModel)
- Lambda (CreateFunction, InvokeFunction)
- DynamoDB (CreateTable, PutItem, Query)
- ECS (CreateCluster, RunTask)
- CloudWatch (PutMetricData, CreateLogGroup)
- EventBridge (PutEvents, CreateRule)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/livestream-sentiment-pulse.git
cd livestream-sentiment-pulse
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 5. Install AWS SAM CLI (for deployment)

```bash
# macOS
brew install aws-sam-cli

# Windows
choco install aws-sam-cli

# Linux
pip install aws-sam-cli
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=your-account-id

# Kinesis Configuration
KINESIS_VIDEO_STREAM_NAME=livestream-video-input
KINESIS_DATA_STREAM_NAME=livestream-sentiment-data

# DynamoDB Configuration
DYNAMODB_TABLE_NAME=sentiment-analytics
DYNAMODB_EVENTS_TABLE=live-events

# API Gateway Configuration
API_GATEWAY_ENDPOINT=https://your-api-id.execute-api.us-east-1.amazonaws.com

# Social Media APIs
TWITTER_API_KEY=your-twitter-api-key
TWITTER_API_SECRET=your-twitter-api-secret
TWITTER_BEARER_TOKEN=your-twitter-bearer-token
INSTAGRAM_ACCESS_TOKEN=your-instagram-token

# AWS Bedrock Configuration
BEDROCK_MODEL_ID=anthropic.claude-v2
BEDROCK_REGION=us-east-1

# Alert Configuration
ALERT_EMAIL=your-email@example.com
SNS_TOPIC_ARN=arn:aws:sns:us-east-1:account-id:sentiment-alerts

# Application Configuration
LOG_LEVEL=INFO
SENTIMENT_THRESHOLD_NEGATIVE=-0.5
SENTIMENT_THRESHOLD_POSITIVE=0.5
ENGAGEMENT_DROP_THRESHOLD=30  # percentage

# ECS Configuration
ECS_CLUSTER_NAME=sentiment-analytics-cluster
ECS_TASK_DEFINITION=sentiment-processor
ECS_DESIRED_COUNT=2

# CloudWatch Configuration
CLOUDWATCH_NAMESPACE=