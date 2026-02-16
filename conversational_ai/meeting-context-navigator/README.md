# 🎯 Meeting Context Navigator

An intelligent conversational assistant that indexes and understands your meeting transcripts, allowing natural language queries across all past conversations. Quickly retrieve decisions, action items, and key discussions without manually reviewing hours of recordings or documents.

## 📋 Overview

Meeting Context Navigator leverages AWS serverless architecture and AI to transform your meeting transcripts into a searchable, queryable knowledge base. Ask questions in natural language like "What decisions did we make about the Q4 budget?" or "Find all action items assigned to Sarah" and get accurate, context-aware answers with source references.

Perfect for enterprise teams, project managers, and executives who need instant access to meeting insights across distributed teams.

## ✨ Features

- 🤖 **Natural Language Queries** - Ask questions conversationally and get intelligent responses
- 📚 **Semantic Search** - Find relevant information even when exact keywords don't match
- 🔍 **Source Attribution** - Every answer includes references to specific meetings and timestamps
- 📊 **Multi-Meeting Context** - Synthesize information across multiple meetings
- 🏷️ **Smart Tagging** - Automatically categorize meetings by topic, participants, and themes
- ⚡ **Real-time Indexing** - New transcripts are automatically processed and made searchable
- 🔐 **Secure & Private** - Enterprise-grade security with AWS infrastructure
- 📈 **Analytics Dashboard** - Track meeting trends, action items, and decision patterns
- 🔔 **Smart Notifications** - Get alerts for unresolved action items and follow-ups
- 🌐 **RESTful API** - Easy integration with existing tools and workflows

## 🔧 Prerequisites

Before you begin, ensure you have the following:

- **AWS Account** with appropriate permissions for:
  - Lambda, Step Functions, OpenSearch, Bedrock, S3, DynamoDB, EventBridge, API Gateway
- **AWS CLI** installed and configured ([Installation Guide](https://aws.amazon.com/cli/))
- **Python 3.11+** installed locally
- **Node.js 18+** (for AWS CDK deployment)
- **Docker** (for local testing with SAM)
- **Git** for version control
- **AWS Bedrock Model Access** - Request access to Claude 3 or similar models in your AWS region
- **Terraform** or **AWS CDK** knowledge (optional, for infrastructure as code)

### Required AWS Permissions

Your IAM user/role needs permissions for:
```
lambda:*, states:*, es:*, bedrock:*, s3:*, dynamodb:*, 
events:*, apigateway:*, logs:*, iam:CreateRole, iam:AttachRolePolicy
```

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/meeting-context-navigator.git
cd meeting-context-navigator
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 3. Install AWS CDK (if using CDK for deployment)

```bash
npm install -g aws-cdk
cd infrastructure
npm install
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=123456789012

# OpenSearch Configuration
OPENSEARCH_DOMAIN_ENDPOINT=https://your-domain.us-east-1.es.amazonaws.com
OPENSEARCH_INDEX_NAME=meeting-transcripts

# Bedrock Configuration
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_REGION=us-east-1

# S3 Configuration
TRANSCRIPT_BUCKET_NAME=meeting-transcripts-bucket
PROCESSED_BUCKET_NAME=processed-transcripts-bucket

# DynamoDB Configuration
MEETINGS_TABLE_NAME=MeetingsMetadata
QUERIES_TABLE_NAME=QueryHistory

# API Configuration
API_GATEWAY_STAGE=prod
API_KEY_REQUIRED=true

# Step Functions
PROCESSING_STATE_MACHINE_ARN=arn:aws:states:region:account:stateMachine:TranscriptProcessor

# EventBridge
EVENT_BUS_NAME=meeting-events

# Application Settings
MAX_QUERY_RESULTS=10
EMBEDDING_DIMENSION=1536
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Configuration File

Create `config/settings.yaml`:

```yaml
application:
  name: meeting-context-navigator
  version: 1.0.0
  log_level: INFO

indexing:
  chunk_size: 1000
  chunk_overlap: 200
  batch_size: 100
  embedding_model: amazon.titan-embed-text-v1

search:
  max_results: 10
  similarity_threshold: 0.7
  boost_recent_meetings: true
  recency_weight: 0.2

llm:
  model_id: anthropic.claude-3-sonnet-20240229-v1:0
  temperature: 0.3
  max_tokens: 2000
  system_prompt: |
    You are an intelligent meeting assistant. Analyze meeting transcripts 
    and provide accurate, concise answers with source references.

notifications:
  enabled: true
  action_item_reminder_days: 3
  unresolved_followup_days: 7
```

## 🚀 Usage

### Upload Meeting Transcripts

```bash
# Using AWS CLI
aws s3 cp meeting-transcript.json s3://meeting-transcripts-bucket/2024/01/meeting-001.json

# Using Python SDK
python scripts/upload_transcript.py --file meeting-transcript.json --meeting-id meeting-001
```

**Transcript Format (JSON):**
```json
{
  "meeting_id": "meeting-001",
  "title": "Q4 Budget Planning",
  "date": "2024-01-15T14:00:00Z",
  "participants": ["John Doe", "Sarah Smith", "Mike Johnson"],
  "duration_minutes": 60,
  "transcript": [
    {
      "timestamp": "00:00:15",
      "speaker": "John Doe",
      "text": "Let's discuss the Q4 budget allocation for marketing."
    },
    {
      "timestamp": "00:01:30",
      "speaker": "Sarah Smith",
      "text": "I propose we increase the digital marketing budget by 20%."
    }
  ],
  "action_items": [
    {
      "assignee": "Sarah Smith",
      "task": "Prepare detailed marketing budget breakdown",
      "due_date": "2024-01-22"
    }
  ]
}
```

### Query Meetings via API

```bash
# REST API Query
curl -X POST https://your-api-id.execute-api.us-east-1.amazonaws.com/prod/query \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "query": "What decisions did we make about the Q4 budget?",
    "max_results": 5,
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-01-31"
    }
  }'
```

**Response:**
```json
{
  "query": "What decisions did we make about the Q4 budget?",
  "answer": "The team decided to increase the digital marketing budget by 20% for Q4. This decision was made during the budget planning meeting on January 15th, with Sarah Smith proposing the increase to focus on social media campaigns.",
  "sources": [
    {
      "meeting_id": "meeting-001",
      "meeting_title": "Q4 Budget Planning",
      "date": "2024-01-15",
      "timestamp": "00:01:30",
      "speaker": "Sarah Smith",
      "excerpt": "I propose we increase the digital marketing budget by 20%...",
      "relevance_score": 0.94
    }
  ],
  "confidence": 0.92,
  "