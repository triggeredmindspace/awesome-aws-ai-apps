# Awesome AWS AI Apps

A curated collection of production-ready AI applications built on AWS. Each app ships with full source code, a CloudFormation template, a deploy script, and runnable examples.

## Categories

| Category | Apps | Key Services |
|---|---|---|
| [RAG on AWS](#-rag-on-aws) | 2 | Bedrock, OpenSearch Serverless, S3, DynamoDB |
| [Conversational AI](#-conversational-ai) | 1 | Bedrock, Transcribe, DynamoDB |
| [Serverless AI Apps](#-serverless-ai-apps) | 1 | Lambda, API Gateway, Textract, Bedrock |
| [Real-time AI Streaming](#-real-time-ai-streaming) | 1 | Kinesis, Lambda, Bedrock, OpenSearch |
| [SageMaker ML Apps](#-sagemaker-ml-apps) | 1 | SageMaker, Forecast, DynamoDB |
| [Bedrock AI Agents](#-bedrock-ai-agents) | coming soon | Bedrock Agents, Lambda, Knowledge Bases |

---

## 📚 RAG on AWS

Retrieval-Augmented Generation — give LLMs access to your own documents.

### [Enterprise Knowledge Base RAG](./rag_on_aws/enterprise-knowledge-base-rag/)
Query internal documents in plain English. Employees get accurate, cited answers from HR policies, legal docs, runbooks, and support wikis.

**Services:** Bedrock (Claude + Titan Embeddings), OpenSearch Serverless, S3, DynamoDB, Textract, Comprehend, SNS

### [Multi-Tenant Customer Support RAG](./rag_on_aws/customer-support-rag/)
Per-tenant isolated vector indices, multi-turn conversation memory, and automatic escalation to human agents after repeated low-confidence responses.

**Services:** Bedrock (Claude + Titan Embeddings), OpenSearch Serverless, S3, DynamoDB, Textract, SQS FIFO

---

## 💬 Conversational AI

### [Meeting Context Navigator](./conversational_ai/meeting-context-navigator/)
Index meeting transcripts and query them conversationally — retrieve decisions, action items, and key discussions without scrubbing through recordings.

**Services:** Bedrock, Transcribe, S3, DynamoDB, OpenSearch

---

## ⚡ Serverless AI Apps

### [Smart Invoice Intelligence Platform](./serverless_ai_apps/smart-invoice-intelligence-platform/)
Automated invoice processing — extract, validate, categorise, and route invoices with zero manual data entry. Detects anomalies and integrates with accounting systems.

**Services:** Lambda, API Gateway, Textract, Bedrock, DynamoDB, S3, SQS

---

## 🌊 Real-time AI Streaming

### [LiveStream Sentiment Pulse Analytics](./realtime_ai_streaming/livestream-sentiment-pulse-analytics/)
Analyse live video and social media feeds simultaneously to surface audience sentiment and engagement patterns in real time during broadcasts or product launches.

**Services:** Kinesis Data Streams, Lambda, Bedrock, OpenSearch, Rekognition, Comprehend

---

## 🧪 SageMaker ML Apps

### [Smart Inventory Demand Forecaster](./sagemaker_ml_apps/smart-inventory-demand-forecaster/)
Time-series demand forecasting across multiple warehouses. Analyses historical sales, seasonal trends, and external signals to optimise stock and prevent stockouts.

**Services:** SageMaker, Amazon Forecast, DynamoDB, S3, Lambda, QuickSight

---

## 🤖 Bedrock AI Agents

Coming soon — autonomous agents built with Amazon Bedrock Agents, Lambda action groups, and Knowledge Bases.

---

## Getting Started

Every app follows the same pattern:

```bash
cd <category>/<app-name>/aws
./deploy.sh            # provisions all AWS resources via CloudFormation
cd ..
pip install -r requirements.txt
python <app>.py        # starts the service
```

Detailed setup steps, environment variables, and API examples are in each app's own README.

## Prerequisites

- AWS account with permissions for the services listed per app
- Python 3.10+
- AWS CLI configured (`aws configure`)

## Cost

These apps use AWS services that incur costs. Review pricing for each service before deploying. Most apps are designed to stay within AWS Free Tier limits for light testing, but production workloads — especially OpenSearch Serverless and Bedrock inference — will accrue charges.

## License

MIT — see individual app directories for details.

---

*New applications are added regularly. Watch or star this repo to stay updated.*
