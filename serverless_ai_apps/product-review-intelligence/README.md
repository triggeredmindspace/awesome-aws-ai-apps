# AI-Powered Product Review Intelligence Platform

A fully serverless platform that ingests customer product reviews from multiple sources, analyses them with **Amazon Bedrock (Claude)**, and surfaces actionable insights through a REST API and automated weekly email digests — with zero servers to manage and pay-per-use billing.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12+-green.svg)
![AWS](https://img.shields.io/badge/AWS-Serverless-orange.svg)

---

## Features

- **Real-Time Review Ingestion** — POST individual reviews via REST API or drop JSON batch files into S3 for bulk processing
- **AI-Powered Analysis** — Claude on Amazon Bedrock extracts sentiment, star rating, themes, praise, and issues from free-text reviews
- **Executive Insight Summaries** — Aggregate dozens of reviews into a single VP-ready summary with recommended actions
- **Scheduled Weekly Digests** — EventBridge triggers automatic insight generation every Monday, with optional SES email delivery
- **Async, Fault-Tolerant Processing** — SQS FIFO queue with dead-letter queue ensures no review is lost under load spikes
- **Fully Observable** — AWS X-Ray tracing on every Lambda function; CloudWatch metrics and logs out-of-the-box

---

## Architecture

```
                        ┌──────────────────────────────────────────────────┐
                        │                  Ingestion Layer                  │
                        │                                                  │
  REST API Client ──▶  API Gateway  ──▶  IngestReview Lambda              │
  Batch JSON File ──▶  S3 Bucket   ──▶  S3BatchIngest Lambda  ──┐         │
                        └──────────────────────────────────────┼─┘         │
                                                               │
                                         SQS FIFO Queue ◀─────┘
                                               │
                                               ▼
                                   ProcessReview Lambda
                                         │
                              Amazon Bedrock (Claude)
                                         │
                              DynamoDB  (reviews table)
                                         │
                         ┌───────────────┴──────────────────┐
                         │          Insights Layer           │
                         │                                  │
          GET /insights  │  GetInsights Lambda              │
     POST /insights/gen  │  GenerateInsights Lambda ──▶  DynamoDB (insights)
    EventBridge (weekly) │  WeeklyDigest Lambda    ──▶  SES Email
                         └──────────────────────────────────┘
```

### AWS Services Used

| Service | Role |
|---|---|
| **API Gateway** | REST API entry point |
| **Lambda (×6)** | Ingest, process, query, generate, digest, batch ingest |
| **Amazon Bedrock** | Claude for review analysis and executive summaries |
| **SQS (FIFO + DLQ)** | Async review processing queue with retry/dead-letter |
| **DynamoDB** | Reviews and insights storage with GSIs for product queries |
| **S3** | Batch review file ingestion and Lambda deployment packages |
| **EventBridge** | Weekly digest scheduler |
| **SES** | Digest email delivery |
| **X-Ray** | Distributed tracing |
| **CloudFormation** | Infrastructure as Code |

---

## Prerequisites

- AWS account with permissions for Lambda, API Gateway, DynamoDB, S3, SQS, Bedrock, SES, and CloudFormation
- Python 3.12+
- AWS CLI configured (`aws configure`)
- Amazon Bedrock access enabled for `amazon.nova-pro-v1:0` (or your chosen model) in your region
- *(Optional)* A verified SES email identity for digest emails

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/your-org/awesome-aws-ai-apps.git
cd awesome-aws-ai-apps/serverless_ai_apps/product-review-intelligence
```

### 2. Deploy

```bash
chmod +x aws/deploy.sh

# Basic deploy (dev environment, no email)
./aws/deploy.sh

# Production deploy with weekly email digest
./aws/deploy.sh --env prod --region us-east-1 --email alerts@yourcompany.com
```

The script will:
1. Create an S3 deployment bucket
2. Install dependencies and package the Lambda zip
3. Upload the package to S3
4. Deploy (or update) the CloudFormation stack
5. Print the API endpoint and resource names

### 3. Submit a review

```bash
API_URL="<ApiEndpoint from deploy output>"

curl -X POST "${API_URL}/reviews" \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "wireless-headphones-v2",
    "product_name": "Wireless Headphones V2",
    "review_text": "Great sound quality and the noise cancellation is impressive. Battery lasts all day. However the ear cushions feel a bit cheap and the app crashes occasionally."
  }'
# → { "review_id": "...", "status": "queued" }
```

### 4. Generate insights

```bash
curl -X POST "${API_URL}/insights/wireless-headphones-v2/generate" \
  -H "Content-Type: application/json" \
  -d '{"product_name": "Wireless Headphones V2", "send_email": false}'
```

### 5. Retrieve insights

```bash
curl "${API_URL}/insights/wireless-headphones-v2"
```

---

## API Reference

### `POST /reviews`

Ingest a single customer review for async processing.

**Request body**

| Field | Type | Required | Description |
|---|---|---|---|
| `product_id` | string | Yes | Unique product identifier |
| `product_name` | string | Yes | Human-readable product name |
| `review_text` | string | Yes | Raw review text (max 5 000 chars) |
| `source` | string | No | Optional source label (e.g. `amazon`, `app_store`) |

**Response** `202 Accepted`
```json
{ "review_id": "uuid", "status": "queued" }
```

---

### `GET /insights/{product_id}`

Retrieve the most recent executive insight for a product.

**Response** `200 OK`
```json
{
  "insight_id": "uuid",
  "product_id": "wireless-headphones-v2",
  "product_name": "Wireless Headphones V2",
  "review_count": 42,
  "summary": {
    "overall_sentiment": "positive",
    "average_rating": 4.1,
    "top_themes": ["battery life", "noise cancellation", "app stability"],
    "top_issues": ["app crashes", "ear cushion quality"],
    "top_praise": ["sound quality", "all-day battery", "build quality"],
    "executive_summary": "...",
    "recommended_actions": ["Fix app crash on Bluetooth reconnect", "Upgrade ear cushion material"]
  },
  "created_at": "2025-01-01T08:00:00+00:00"
}
```

---

### `POST /insights/{product_id}/generate`

Trigger on-demand insight generation from the latest reviews.

**Request body**

| Field | Type | Required | Description |
|---|---|---|---|
| `product_name` | string | Yes | Human-readable product name |
| `send_email` | boolean | No | Send digest via SES (default: `false`) |

**Response** `200 OK` — returns the generated insight object.

---

## Batch Ingestion via S3

Drop a JSON file into the ingestion S3 bucket (prefix `batch/`) to process multiple reviews at once:

```json
[
  {
    "product_id": "widget-pro",
    "product_name": "Widget Pro",
    "review_text": "Excellent build quality, arrived on time."
  },
  {
    "product_id": "widget-pro",
    "product_name": "Widget Pro",
    "review_text": "Stopped working after two weeks. Very disappointed."
  }
]
```

```bash
aws s3 cp reviews_batch.json s3://<ReviewsBucket>/batch/reviews_batch.json
```

---

## Configuration

Edit `config.yaml` to customise the Bedrock model, DynamoDB table names, SQS settings, Lambda memory/timeout, and the EventBridge schedule before deploying.

Key settings:

| Key | Default | Description |
|---|---|---|
| `bedrock.model_id` | `amazon.nova-pro-v1:0` | Bedrock model for analysis |
| `eventbridge.digest_schedule` | `cron(0 8 ? * MON *)` | Weekly digest schedule |
| `ses.digest_email` | *(empty)* | Verified SES email; leave empty to disable |
| `lambda.memory_mb` | `512` | Lambda memory allocation |

---

## Clean Up

```bash
# Delete the CloudFormation stack (removes Lambda, API GW, DynamoDB, SQS, EventBridge)
aws cloudformation delete-stack \
  --stack-name product-review-intelligence-dev \
  --region us-east-1

# Optionally remove the S3 buckets (empty them first)
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws s3 rm s3://product-review-intelligence-dev-${ACCOUNT_ID} --recursive
aws s3 rm s3://product-review-deploy-${ACCOUNT_ID}-us-east-1 --recursive
aws s3api delete-bucket --bucket product-review-intelligence-dev-${ACCOUNT_ID}
aws s3api delete-bucket --bucket product-review-deploy-${ACCOUNT_ID}-us-east-1
```

---

## License

MIT
