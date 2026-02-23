# 🧾 Smart Invoice Intelligence Platform

An automated invoice processing system that leverages AI and AWS services to extract, validate, and categorize invoice data in real-time. This platform eliminates manual data entry, detects anomalies, matches purchase orders, and seamlessly integrates with accounting systems to accelerate accounts payable workflows.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![AWS](https://img.shields.io/badge/AWS-Cloud-orange.svg)

---

## 🌟 Features

- **📄 Intelligent Document Processing**: Automatically extract data from invoices in multiple formats (PDF, PNG, JPEG) using AWS Textract
- **🤖 AI-Powered Validation**: Leverage Amazon SageMaker for anomaly detection and data validation
- **🔄 Real-time Processing**: Event-driven architecture using Lambda and EventBridge for instant invoice processing
- **📊 Purchase Order Matching**: Automatically match invoices with existing purchase orders
- **🔍 Anomaly Detection**: Identify duplicate invoices, pricing discrepancies, and suspicious patterns
- **🔐 Secure Authentication**: User management and authentication via AWS Cognito
- **📈 Analytics Dashboard**: Track processing metrics and invoice status through CloudWatch
- **🔔 Smart Notifications**: Receive alerts for anomalies and processing status via SNS
- **🔌 API Integration**: RESTful API for seamless integration with existing accounting systems
- **💾 Scalable Storage**: Secure document storage in S3 with metadata in DynamoDB

---

## 📋 Prerequisites

Before you begin, ensure you have the following:

- **AWS Account** with appropriate permissions
- **Python 3.9 or higher** installed locally
- **AWS CLI** configured with credentials
- **Node.js 14+** (for AWS CDK deployment, optional)
- **Git** for version control
- **Basic knowledge** of AWS services and serverless architecture
- **Accounting system API credentials** (QuickBooks, Xero, etc.) for integration

---

## 🚀 Installation Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/smart-invoice-platform.git
cd smart-invoice-platform
```

### 2. Set Up Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install AWS SAM CLI (Optional, for local testing)

```bash
# macOS
brew install aws-sam-cli

# Windows
choco install aws-sam-cli

# Linux
pip install aws-sam-cli
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=your-account-id

# S3 Buckets
INVOICE_BUCKET_NAME=smart-invoice-uploads
PROCESSED_BUCKET_NAME=smart-invoice-processed

# DynamoDB Tables
INVOICE_TABLE_NAME=InvoiceMetadata
PO_TABLE_NAME=PurchaseOrders
VENDOR_TABLE_NAME=VendorMaster

# API Gateway
API_GATEWAY_ENDPOINT=https://your-api-id.execute-api.us-east-1.amazonaws.com/prod

# Cognito
COGNITO_USER_POOL_ID=us-east-1_xxxxxxxxx
COGNITO_CLIENT_ID=your-client-id

# SageMaker
SAGEMAKER_ENDPOINT_NAME=invoice-anomaly-detector

# SNS Topics
SNS_ALERT_TOPIC_ARN=arn:aws:sns:us-east-1:account-id:invoice-alerts

# Processing Configuration
MAX_INVOICE_SIZE_MB=10
CONFIDENCE_THRESHOLD=0.85
ANOMALY_THRESHOLD=0.75

# Accounting Integration (Optional)
QUICKBOOKS_CLIENT_ID=your-quickbooks-client-id
QUICKBOOKS_CLIENT_SECRET=your-quickbooks-secret
XERO_CLIENT_ID=your-xero-client-id
XERO_CLIENT_SECRET=your-xero-secret
```

### Configuration File

Edit `config/settings.json` for additional settings:

```json
{
  "invoice_processing": {
    "supported_formats": ["pdf", "png", "jpg", "jpeg"],
    "max_file_size_mb": 10,
    "ocr_confidence_threshold": 0.85
  },
  "validation_rules": {
    "duplicate_check_days": 90,
    "price_variance_threshold": 0.10,
    "required_fields": ["invoice_number", "vendor_name", "total_amount", "invoice_date"]
  },
  "notifications": {
    "send_on_anomaly": true,
    "send_on_completion": false,
    "send_on_error": true
  }
}
```

---

## 📖 Usage Instructions

### Uploading Invoices via API

```python
import requests
import json

# Authenticate and get token
auth_response = requests.post(
    f"{API_GATEWAY_ENDPOINT}/auth/login",
    json={
        "username": "your-username",
        "password": "your-password"
    }
)
token = auth_response.json()["token"]

# Upload invoice
headers = {"Authorization": f"Bearer {token}"}
files = {"file": open("invoice.pdf", "rb")}
metadata = {
    "vendor_name": "Acme Corp",
    "po_number": "PO-12345"
}

response = requests.post(
    f"{API_GATEWAY_ENDPOINT}/invoices/upload",
    headers=headers,
    files=files,
    data={"metadata": json.dumps(metadata)}
)

print(response.json())
# Output: {"invoice_id": "inv-abc123", "status": "processing"}
```

### Checking Invoice Status

```python
invoice_id = "inv-abc123"
response = requests.get(
    f"{API_GATEWAY_ENDPOINT}/invoices/{invoice_id}",
    headers=headers
)

print(response.json())
```

### Retrieving Processed Data

```python
response = requests.get(
    f"{API_GATEWAY_ENDPOINT}/invoices/{invoice_id}/data",
    headers=headers
)

invoice_data = response.json()
print(f"Vendor: {invoice_data['vendor_name']}")
print(f"Amount: ${invoice_data['total_amount']}")
print(f"Status: {invoice_data['validation_status']}")
```

### Using the Python SDK

```python
from smart_invoice import InvoiceClient

# Initialize client
client = InvoiceClient(
    api_endpoint=API_GATEWAY_ENDPOINT,
    username="your-username",
    password="your-password"
)

# Upload and process invoice
invoice = client.upload_invoice(
    file_path="invoice.pdf",
    vendor_name="Acme Corp",
    po_number="PO-12345"
)

# Wait for processing
invoice.wait_until_processed(timeout=60)

# Get results
if invoice.has_anomalies():
    print(f"Anomalies detected: {invoice.anomalies}")
else:
    print(f"Invoice validated successfully")
    print(f"Extracted data: {invoice.data}")
```

---

## 🏗️ AWS Setup Guide

### Step 1: Create S3 Buckets

```bash
# Create buckets for invoice storage
aws s3 mb s3://smart-invoice-uploads --region us-east-1
aws s3 mb s3://smart-invoice-processed --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket smart-invoice-uploads \
    --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption \
    --bucket smart-invoice-uploads \
    --server-side-encryption-configuration '{
        "Rules": [{
            "ApplyServerSideEncryptionByDefault": {