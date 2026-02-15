# 🎯 Smart Inventory Demand Forecaster

An intelligent, ML-powered inventory management system that leverages time-series forecasting to predict product demand across multiple warehouses. Built on AWS infrastructure, this application analyzes historical sales data, seasonal patterns, and external factors to optimize stock levels, reduce waste, and prevent costly stockouts.

![Architecture](https://img.shields.io/badge/AWS-Advanced-orange) ![Python](https://img.shields.io/badge/Python-3.9+-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [AWS Setup Guide](#aws-setup-guide)
- [Usage Instructions](#usage-instructions)
- [API Reference](#api-reference)
- [Cost Considerations](#cost-considerations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **🤖 ML-Powered Forecasting**: Uses Amazon SageMaker with DeepAR algorithm for accurate time-series predictions
- **📊 Real-Time Dashboards**: Interactive visualizations showing demand trends, stock levels, and forecast accuracy
- **🏢 Multi-Warehouse Support**: Manage inventory across multiple locations with location-specific forecasting
- **📈 Seasonal Pattern Detection**: Automatically identifies and accounts for seasonal variations in demand
- **🔔 Intelligent Alerts**: Automated notifications for low stock, overstock situations, and forecast anomalies
- **📉 Waste Reduction**: Optimize stock levels to minimize expired or obsolete inventory
- **🔍 Advanced Search**: OpenSearch-powered analytics for historical data exploration
- **🔄 Automated Retraining**: Scheduled model updates to maintain forecast accuracy
- **📱 RESTful API**: Easy integration with existing ERP and inventory management systems
- **🎨 Customizable Thresholds**: Configure reorder points and safety stock levels per product

## 🏗️ Architecture Overview

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Client    │─────▶│ API Gateway  │─────▶│   Lambda    │
│ Application │      │              │      │  Functions  │
└─────────────┘      └──────────────┘      └──────┬──────┘
                                                   │
                     ┌─────────────────────────────┼────────────────┐
                     │                             │                │
              ┌──────▼──────┐            ┌────────▼────────┐  ┌───▼────┐
              │  SageMaker  │            │   OpenSearch    │  │   S3   │
              │   Models    │            │    Cluster      │  │ Bucket │
              └─────────────┘            └─────────────────┘  └────────┘
                     │                             │
              ┌──────▼──────┐            ┌────────▼────────┐
              │ EventBridge │            │   CloudWatch    │
              │    Rules    │            │  Logs/Metrics   │
              └─────────────┘            └─────────────────┘
                     │
              ┌──────▼──────┐
              │     ECS     │
              │  Dashboard  │
              └─────────────┘
```

## 📦 Prerequisites

Before you begin, ensure you have the following:

- **AWS Account** with appropriate permissions for:
  - SageMaker (full access)
  - Lambda, API Gateway, ECS
  - S3, OpenSearch, EventBridge, CloudWatch
  - IAM role creation
- **Python 3.9+** installed locally
- **AWS CLI** configured with credentials (`aws configure`)
- **Docker** (for local testing and ECS deployment)
- **Node.js 16+** (for dashboard frontend)
- **Terraform** or **AWS CDK** (optional, for infrastructure as code)
- **Git** for version control

### Required Python Packages

```
boto3>=1.26.0
sagemaker>=2.150.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
fastapi>=0.100.0
uvicorn>=0.23.0
opensearch-py>=2.3.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/smart-inventory-forecaster.git
cd smart-inventory-forecaster
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies

```bash
cd dashboard
npm install
cd ..
```

### 5. Set Up Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=123456789012

# S3 Configuration
S3_BUCKET_NAME=inventory-forecaster-data
S3_MODEL_PREFIX=models/
S3_DATA_PREFIX=data/

# SageMaker Configuration
SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/SageMakerExecutionRole
SAGEMAKER_INSTANCE_TYPE=ml.m5.xlarge
SAGEMAKER_ENDPOINT_NAME=inventory-demand-forecaster

# OpenSearch Configuration
OPENSEARCH_ENDPOINT=https://search-inventory-xxxxx.us-east-1.es.amazonaws.com
OPENSEARCH_INDEX=inventory-data
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=YourSecurePassword123!

# API Gateway Configuration
API_GATEWAY_URL=https://xxxxx.execute-api.us-east-1.amazonaws.com/prod

# Lambda Configuration
LAMBDA_MEMORY_SIZE=512
LAMBDA_TIMEOUT=300

# ECS Configuration
ECS_CLUSTER_NAME=inventory-dashboard-cluster
ECS_SERVICE_NAME=dashboard-service
ECS_TASK_CPU=512
ECS_TASK_MEMORY=1024

# Application Configuration
FORECAST_HORIZON_DAYS=30
RETRAINING_FREQUENCY_DAYS=7
MIN_HISTORICAL_DAYS=90
CONFIDENCE_INTERVAL=0.95

# Alert Thresholds
LOW_STOCK_THRESHOLD=0.2
OVERSTOCK_THRESHOLD=2.0
FORECAST_ACCURACY_THRESHOLD=0.85

# EventBridge Configuration
RETRAINING_SCHEDULE=cron(0 2 ? * SUN *)
DATA_INGESTION_SCHEDULE=rate(1 hour)

# CloudWatch Configuration
LOG_LEVEL=INFO
METRICS_NAMESPACE=InventoryForecaster
```

### Configuration Files

Create `config/app_config.yaml`:

```yaml
forecasting:
  algorithm: deepar
  context_length: 30
  prediction_length: 30
  num_cells: 40
  num_layers: 2
  dropout_rate: 0.1
  
warehouses:
  - id: WH001
    name: "East Coast Distribution"
    location: "New York, NY"
  - id: WH002
    name: "West Coast Distribution"
    location: "Los Angeles, CA"
  - id: WH003
    name: "Central Distribution"
    location: "Chicago, IL"

products:
  categories:
    - electronics
    - clothing
    - food
    - household
    
alerts:
  email_recipients:
    - inventory@company.com
    - procurement@company.com
  sns_topic_arn: arn:aws:sns:us-east-1:123456789012:inventory-alerts
```

## 🛠️