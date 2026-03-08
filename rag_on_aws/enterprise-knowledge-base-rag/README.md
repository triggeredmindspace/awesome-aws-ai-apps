# 📚 Enterprise Knowledge Base RAG

An intelligent Retrieval-Augmented Generation (RAG) system that transforms your enterprise document repositories into a queryable knowledge base. Employees can ask plain-English questions and receive accurate, cited answers—powered by Amazon Bedrock (Claude), Amazon OpenSearch Serverless, and Amazon S3.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![AWS](https://img.shields.io/badge/AWS-Cloud-orange.svg)

---

## 🌟 Features

- **📄 Multi-format Ingestion**: Upload PDF, TXT, and Markdown documents; text is extracted from PDFs using Amazon Textract
- **🧩 Intelligent Chunking**: Overlapping sentence-boundary-aware chunking maximises retrieval recall
- **🔢 Vector Embeddings**: Amazon Titan Embeddings v2 converts every chunk into a high-dimensional vector
- **🔍 Semantic Search**: k-NN vector search on Amazon OpenSearch Serverless surfaces the most relevant context
- **🤖 Grounded Generation**: Amazon Bedrock (Claude) synthesises accurate, citation-backed answers
- **💬 Multi-turn Conversations**: Conversation history is preserved in DynamoDB for contextual follow-up questions
- **🏷️ Category Filtering**: Documents are tagged by category (policy, technical, legal, HR, finance) for targeted search
- **🔗 Named Entity Detection**: Amazon Comprehend extracts entities from queries for richer analytics
- **📊 Quality Feedback**: Users can rate responses (1–5); data feeds a continuous improvement loop
- **📈 Usage Statistics**: Built-in `/stats` endpoint tracks query volume and indexed document count
- **🔔 Ingestion Notifications**: SNS alerts fire whenever a document is successfully indexed
- **🏥 Health Checks**: `/health` endpoint verifies S3, Bedrock, and DynamoDB connectivity

---

## 🏗️ Architecture

```
┌─────────────┐     REST API      ┌──────────────────────────────────────┐
│   Client    │ ◄────────────────► │         FastAPI Application          │
└─────────────┘                   └──────┬───────────────────────────────┘
                                         │
           ┌─────────────────────────────┼──────────────────────────────┐
           │                             │                              │
    ┌──────▼──────┐            ┌─────────▼──────┐           ┌──────────▼───────┐
    │  Amazon S3  │            │ Amazon Bedrock  │           │   Amazon         │
    │  (Document  │            │ Claude (Answer) │           │   OpenSearch     │
    │   Storage)  │            │ Titan (Embed)   │           │   Serverless     │
    └──────┬──────┘            └────────────────┘           │   (Vector Store) │
           │                                                └──────────────────┘
    ┌──────▼──────┐   ┌───────────────┐   ┌─────────────┐
    │  Textract   │   │   DynamoDB    │   │  Comprehend │
    │  (PDF OCR)  │   │  (Query Log)  │   │  (Entities) │
    └─────────────┘   └───────────────┘   └─────────────┘
```

**Document ingestion flow:**
1. Document uploaded → stored in S3
2. Text extracted (Textract for PDFs, decode for text)
3. Text chunked into overlapping segments
4. Each chunk embedded via Titan Embeddings v2
5. Chunk + embedding indexed into OpenSearch Serverless

**Query flow:**
1. Query embedded via Titan Embeddings v2
2. k-NN search retrieves top-K relevant chunks from OpenSearch
3. Chunks + conversation history passed to Bedrock (Claude)
4. Grounded answer returned with source citations

---

## 📋 Prerequisites

- **AWS Account** with permissions for: Bedrock, OpenSearch Serverless, S3, DynamoDB, Textract, Comprehend, SNS, Lambda
- **Python 3.10+**
- **AWS CLI** configured (`aws configure`)
- **Amazon Bedrock model access** — enable Claude and Titan Embeddings v2 in the Bedrock console
- **OpenSearch Serverless collection** with a k-NN index (see setup below)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/awesome-aws-ai-apps.git
cd awesome-aws-ai-apps/rag_on_aws/enterprise-knowledge-base-rag
```

### 2. Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Deploy AWS Infrastructure

```bash
chmod +x aws/deploy.sh
./aws/deploy.sh
```

This script deploys the CloudFormation stack which provisions:
- S3 bucket for document storage
- DynamoDB table for query history
- OpenSearch Serverless collection and k-NN index
- IAM roles and policies
- SNS topic for ingestion notifications

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```bash
# AWS
AWS_REGION=us-east-1

# S3
S3_BUCKET=enterprise-kb-documents

# OpenSearch Serverless
OPENSEARCH_ENDPOINT=https://<collection-id>.us-east-1.aoss.amazonaws.com
OPENSEARCH_INDEX=enterprise-knowledge-base

# Amazon Bedrock
BEDROCK_MODEL_ID=anthropic.claude-sonnet-4-6
BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0

# DynamoDB
DYNAMODB_TABLE=kb-query-history

# SNS (optional)
SNS_TOPIC_ARN=arn:aws:sns:us-east-1:<account-id>:kb-ingestion-notifications

# Retrieval tuning
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
```

---

## 📖 Usage

### Start the API Server

```bash
python app.py
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Upload a Document

```python
import requests

files = {'file': open('company_policy.pdf', 'rb')}
params = {'category': 'policy', 'tags': 'hr,onboarding,2024'}

response = requests.post(
    'http://localhost:8000/documents/upload',
    files=files,
    params=params,
)
print(response.json())
# {
#   "document_id": "a1b2c3d4-...",
#   "filename": "company_policy.pdf",
#   "status": "processing",
#   "s3_key": "documents/policy/a1b2c3d4.../company_policy.pdf",
#   "chunks_created": 0,
#   "message": "Document uploaded. Indexing is in progress."
# }
```

### Query the Knowledge Base

```python
payload = {
    "query": "What is the remote work policy for international employees?",
    "category": "policy",
    "top_k": 5,
    "include_sources": True,
}

response = requests.post('http://localhost:8000/query', json=payload)
result = response.json()

print(result['answer'])
print("\nSources:")
for src in result['sources']:
    print(f"  - {src['filename']} (score: {src['score']})")
    print(f"    {src['excerpt']}")
```

### Multi-turn Conversation

```python
# First question
resp1 = requests.post('http://localhost:8000/query', json={
    "query": "What are the employee benefits?",
    "include_sources": True,
})
conversation_id = resp1.json()['conversation_id']

# Follow-up — pass the same conversation_id
resp2 = requests.post('http://localhost:8000/query', json={
    "query": "Which of those benefits apply to part-time employees?",
    "conversation_id": conversation_id,
    "include_sources": True,
})
print(resp2.json()['answer'])
```

### Submit Feedback

```python
requests.post('http://localhost:8000/feedback', json={
    "query_id": result['query_id'],
    "rating": 5,
    "comment": "Accurate and well-cited answer.",
})
```

### Check Health & Stats

```bash
curl http://localhost:8000/health
curl http://localhost:8000/stats
```

---

## 🏗️ AWS Setup Guide

### Step 1: Enable Bedrock Model Access

1. Open the [Amazon Bedrock console](https://console.aws.amazon.com/bedrock/)
2. Navigate to **Model access**
3. Enable **Claude (Anthropic)** and **Titan Embeddings v2 (Amazon)**

### Step 2: Create OpenSearch Serverless Collection

```bash
# Create collection
aws opensearchserverless create-collection \
    --name enterprise-knowledge-base \
    --type VECTORSEARCH \
    --region us-east-1

# Wait for ACTIVE status
aws opensearchserverless batch-get-collection \
    --names enterprise-knowledge-base

# Create k-NN index (replace <ENDPOINT> with your collection endpoint)
curl -XPUT "https://<ENDPOINT>/enterprise-knowledge-base" \
    -H 'Content-Type: application/json' \
    -d '{
      "settings": {"index": {"knn": true}},
      "mappings": {
        "properties": {
          "embedding": {"type": "knn_vector", "dimension": 1536},
          "content":   {"type": "text"},
          "category":  {"type": "keyword"},
          "filename":  {"type": "keyword"},
          "tags":      {"type": "keyword"}
        }
      }
    }'
```

### Step 3: Create S3 Bucket

```bash
aws s3 mb s3://enterprise-kb-documents --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket enterprise-kb-documents \
    --versioning-configuration Status=Enabled

# Block public access
aws s3api put-public-access-block \
    --bucket enterprise-kb-documents \
    --public-access-block-configuration \
        "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

### Step 4: Create DynamoDB Table

```bash
aws dynamodb create-table \
    --table-name kb-query-history \
    --attribute-definitions \
        AttributeName=query_id,AttributeType=S \
        AttributeName=conversation_id,AttributeType=S \
    --key-schema AttributeName=query_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --global-secondary-indexes '[{
        "IndexName": "conversation_id-index",
        "KeySchema": [{"AttributeName": "conversation_id", "KeyType": "HASH"}],
        "Projection": {"ProjectionType": "ALL"}
    }]'
```

---

## 📂 Project Structure

```
enterprise-knowledge-base-rag/
├── app.py                        # FastAPI application (ingestion + query + feedback)
├── requirements.txt              # Python dependencies
├── config.yaml                   # Chunking and retrieval configuration
├── README.md                     # This file
└── aws/
    ├── deploy.sh                 # One-command deployment script
    └── cloudformation/
        └── template.yaml         # Full infrastructure-as-code template
```

---

## 💡 Use Cases

| Industry | Use Case |
|----------|----------|
| **Legal** | Query contracts, case law, and compliance documents |
| **Healthcare** | Surface clinical guidelines and formulary information |
| **Finance** | Retrieve policy documents, audit reports, and regulations |
| **HR** | Answer employee questions about benefits and policies |
| **Engineering** | Search runbooks, architecture docs, and RFCs |
| **Customer Support** | Resolve tickets using product manuals and FAQs |

---

## 💰 Cost Estimate

| Service | Usage | Estimated Monthly Cost |
|---------|-------|----------------------|
| Amazon Bedrock (Claude Sonnet) | 10,000 queries × ~2K tokens | ~$30 |
| Amazon Bedrock (Titan Embeddings v2) | 500K embeddings | ~$5 |
| OpenSearch Serverless | 1 OCU (indexing) + 1 OCU (search) | ~$350 |
| Amazon S3 | 10 GB storage + requests | ~$2 |
| Amazon DynamoDB | On-demand, 10K WCU + 50K RCU | ~$5 |
| Amazon Textract | 1,000 PDF pages | ~$1.50 |
| **Total** | | **~$395/month** |

> Costs vary by region and actual usage. Use [AWS Pricing Calculator](https://calculator.aws) for precise estimates.

---

## 🔒 Security Best Practices

- All S3 buckets have public access blocked and SSE-S3 encryption enabled
- OpenSearch access policies restrict ingestion to the application IAM role only
- DynamoDB items include a 90-day TTL to limit PII retention
- API Gateway + Cognito can be added for user-level authentication (see `aws/cloudformation/template.yaml`)
- Bedrock model invocations are logged to CloudTrail

---

## 📝 License

MIT License — see the root [LICENSE](../../LICENSE) file for details.
