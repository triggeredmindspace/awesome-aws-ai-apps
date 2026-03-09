# Multi-Tenant Customer Support RAG

A production-ready, multi-tenant customer support assistant powered by **Amazon Bedrock (Claude)**, **Amazon OpenSearch Serverless**, and **Amazon SQS**. Each tenant gets a fully isolated vector index; conversations are stateful and auto-escalate to human agents when confidence drops.

## Architecture

```
Customer / App
      │
      ▼
 FastAPI Service
      │
      ├─► Amazon Bedrock (Claude) ──► answer generation
      ├─► Amazon Bedrock (Titan)  ──► text embeddings
      ├─► OpenSearch Serverless   ──► per-tenant KNN indices
      ├─► DynamoDB (sessions)     ──► conversation history + TTL
      ├─► DynamoDB (tenants)      ──► tenant registry
      ├─► S3                      ──► document storage
      ├─► Textract                ──► PDF / image text extraction
      └─► SQS FIFO               ──► escalation to human agents
```

### What makes this different from a standard RAG

| Feature | Standard RAG | This app |
|---|---|---|
| Tenancy | Single index | Per-tenant isolated OpenSearch index |
| Memory | Stateless | Multi-turn conversation history (DynamoDB) |
| Escalation | Manual | Auto-escalates after 3 low-confidence turns |
| Document ingestion | Text only | PDF, images, Markdown, JSON via Textract |
| Product scoping | Full-corpus search | Optional per-product filter at query time |

## AWS Services

| Service | Role |
|---|---|
| **Amazon Bedrock** (Claude Sonnet 4.6) | Answer generation |
| **Amazon Bedrock** (Titan Embed v2) | 1536-dim text embeddings |
| **Amazon OpenSearch Serverless** | Vector KNN search, per-tenant indices |
| **Amazon DynamoDB** | Session state, tenant registry |
| **Amazon S3** | Raw document storage |
| **Amazon Textract** | Text extraction from PDFs and images |
| **Amazon SQS FIFO** | Escalation event routing per tenant |

## Quick Start

### 1. Deploy infrastructure

```bash
cd aws
export AWS_REGION=us-east-1
export ENVIRONMENT=dev
./deploy.sh
```

This creates all AWS resources and writes a `.env.dev` file.

### 2. Run the service

```bash
source .env.dev
pip install -r requirements.txt
python customer_support_rag.py
```

### 3. Register a tenant

```bash
curl -X POST http://localhost:8000/tenants \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "acme-corp",
    "name": "Acme Corporation",
    "plan": "pro"
  }'
```

### 4. Upload and ingest a document

```bash
# Upload to S3
aws s3 cp product-manual.pdf s3://<DOCS_BUCKET>/acme-corp/product-manual.pdf

# Ingest via API
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "acme-corp",
    "doc_type": "manual",
    "s3_key": "acme-corp/product-manual.pdf",
    "title": "Product Manual v3.2",
    "product": "widget-pro",
    "version": "3.2",
    "tags": ["setup", "troubleshooting"]
  }'
```

### 5. Chat

```bash
# Start a new session
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "acme-corp",
    "user_id": "user-123",
    "message": "How do I reset the Widget Pro to factory settings?",
    "product": "widget-pro"
  }'

# Response includes session_id — pass it in subsequent messages
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<SESSION_ID>",
    "tenant_id": "acme-corp",
    "user_id": "user-123",
    "message": "What if it still doesn'\''t work after that?"
  }'
```

### 6. Manual escalation

```bash
curl -X POST http://localhost:8000/escalate \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<SESSION_ID>",
    "tenant_id": "acme-corp",
    "reason": "user_request",
    "notes": "Customer is frustrated"
  }'
```

## Tenant Isolation

Each tenant gets its own OpenSearch index (`support_<tenant_id>`). Queries, document ingestion, and session history are all strictly scoped — one tenant's documents are never visible to another.

## Escalation Logic

The service tracks a **miss counter** per session. When retrieved chunks score below `confidence_threshold` (default `0.72`), the counter increments. After **3 consecutive low-confidence turns**, the session is automatically escalated:

1. A structured event is sent to the **SQS FIFO queue** (grouped by `tenant_id`)
2. The session is marked `escalated = true`
3. Subsequent messages return a handoff message instead of an AI response

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/tenants` | Register a new tenant |
| `GET` | `/tenants/{id}` | Fetch tenant config |
| `POST` | `/ingest` | Ingest a document (async) |
| `POST` | `/chat` | Send a message, get an AI answer |
| `POST` | `/escalate` | Manually escalate a session |
| `GET` | `/sessions/{id}` | Retrieve session history |
| `GET` | `/health` | Health check |

## Configuration

All settings are available via environment variables or `config.yaml`:

| Variable | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.72` | Min cosine score to trust a retrieved chunk |
| `SESSION_TTL_HOURS` | `24` | Session auto-expiry |
| `MAX_CONVERSATION_TURNS` | `20` | Max turns kept in context |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |

## Use Cases

- **SaaS product support** — isolate each customer's documentation
- **E-commerce help desks** — FAQ + policy + order policy RAG
- **Telecom support** — device manuals + troubleshooting guides per product line
- **Financial services** — product terms, compliance FAQs per institution
