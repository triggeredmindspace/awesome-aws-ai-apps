```python
import os
import json
import uuid
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Environment configuration ───────────────────────────────────────────────
AWS_REGION             = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET              = os.getenv('S3_BUCKET', 'customer-support-docs')
OPENSEARCH_ENDPOINT    = os.getenv('OPENSEARCH_ENDPOINT', '')
BEDROCK_MODEL_ID       = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-sonnet-4-6')
BEDROCK_EMBED_MODEL_ID = os.getenv('BEDROCK_EMBED_MODEL_ID', 'amazon.titan-embed-text-v2:0')
DYNAMODB_SESSIONS      = os.getenv('DYNAMODB_SESSIONS', 'support-sessions')
DYNAMODB_TENANTS       = os.getenv('DYNAMODB_TENANTS', 'support-tenants')
SQS_ESCALATION_URL     = os.getenv('SQS_ESCALATION_URL', '')
ELASTICACHE_ENDPOINT   = os.getenv('ELASTICACHE_ENDPOINT', '')
TOP_K_RESULTS          = int(os.getenv('TOP_K_RESULTS', '5'))
CONFIDENCE_THRESHOLD   = float(os.getenv('CONFIDENCE_THRESHOLD', '0.72'))
SESSION_TTL_HOURS      = int(os.getenv('SESSION_TTL_HOURS', '24'))
MAX_CONVERSATION_TURNS = int(os.getenv('MAX_CONVERSATION_TURNS', '20'))
CHUNK_SIZE             = int(os.getenv('CHUNK_SIZE', '800'))
CHUNK_OVERLAP          = int(os.getenv('CHUNK_OVERLAP', '150'))

# ── AWS clients ─────────────────────────────────────────────────────────────
s3_client         = boto3.client('s3', region_name=AWS_REGION)
bedrock_client    = boto3.client('bedrock-runtime', region_name=AWS_REGION)
dynamodb          = boto3.resource('dynamodb', region_name=AWS_REGION)
sqs_client        = boto3.client('sqs', region_name=AWS_REGION)
textract_client   = boto3.client('textract', region_name=AWS_REGION)

sessions_table = dynamodb.Table(DYNAMODB_SESSIONS)
tenants_table  = dynamodb.Table(DYNAMODB_TENANTS)

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multi-Tenant Customer Support RAG",
    description="Conversational support powered by Amazon Bedrock + OpenSearch with per-tenant isolation",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Enums / constants ────────────────────────────────────────────────────────
class DocType(str, Enum):
    FAQ        = "faq"
    MANUAL     = "manual"
    RELEASE    = "release_notes"
    POLICY     = "policy"
    TICKET     = "resolved_ticket"

class EscalationReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    USER_REQUEST   = "user_request"
    REPEATED_MISS  = "repeated_miss"
    BILLING        = "billing_issue"

# ── Pydantic models ──────────────────────────────────────────────────────────
class TenantCreate(BaseModel):
    tenant_id:   str = Field(..., description="Unique slug, e.g. 'acme-corp'")
    name:        str
    plan:        str = Field(default="starter", description="starter | pro | enterprise")
    metadata:    Dict[str, Any] = {}

class DocumentIngest(BaseModel):
    tenant_id:   str
    doc_type:    DocType
    s3_key:      str = Field(..., description="S3 key of the source document")
    title:       str
    product:     Optional[str] = None
    version:     Optional[str] = None
    tags:        List[str] = []

class ChatMessage(BaseModel):
    session_id:  Optional[str] = None   # None → new session
    tenant_id:   str
    user_id:     str
    message:     str
    product:     Optional[str] = None   # narrows search scope

class EscalateRequest(BaseModel):
    session_id:  str
    tenant_id:   str
    reason:      EscalationReason = EscalationReason.USER_REQUEST
    notes:       Optional[str] = None

# ── Tenant management ────────────────────────────────────────────────────────

def get_tenant(tenant_id: str) -> Dict:
    """Fetch tenant config; raises 404 if not found."""
    resp = tenants_table.get_item(Key={"tenant_id": tenant_id})
    item = resp.get("Item")
    if not item:
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found")
    return item

def tenant_index(tenant_id: str) -> str:
    """OpenSearch index name for a tenant — enforces isolation."""
    safe = tenant_id.replace("-", "_").lower()
    return f"support_{safe}"

# ── Embedding ────────────────────────────────────────────────────────────────

def embed(text: str) -> List[float]:
    """Generate a vector embedding via Amazon Titan."""
    body = json.dumps({"inputText": text[:8192]})
    resp = bedrock_client.invoke_model(
        modelId=BEDROCK_EMBED_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(resp["body"].read())["embedding"]

# ── Text extraction ──────────────────────────────────────────────────────────

def extract_text_from_s3(bucket: str, key: str) -> str:
    """Use Textract for PDFs/images; direct read for .txt/.md."""
    if key.lower().endswith((".txt", ".md", ".json")):
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read().decode("utf-8")

    resp = textract_client.detect_document_text(
        Document={"S3Object": {"Bucket": bucket, "Name": key}}
    )
    lines = [
        block["Text"]
        for block in resp["Blocks"]
        if block["BlockType"] == "LINE"
    ]
    return "\n".join(lines)

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c for c in chunks if c.strip()]

# ── OpenSearch helpers ───────────────────────────────────────────────────────

def _os_request(method: str, path: str, body: Optional[Dict] = None) -> Dict:
    """Minimal OpenSearch Serverless request via boto3 (SigV4 signed)."""
    import urllib.request
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import get_credentials

    session   = boto3.Session()
    creds     = session.get_credentials()
    url       = f"https://{OPENSEARCH_ENDPOINT}/{path}"
    data      = json.dumps(body).encode() if body else b""
    headers   = {"Content-Type": "application/json"}

    aws_req   = AWSRequest(method=method, url=url, data=data, headers=headers)
    SigV4Auth(creds, "aoss", AWS_REGION).add_auth(aws_req)

    req = urllib.request.Request(url, data=data, headers=dict(aws_req.headers), method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())

def ensure_tenant_index(tenant_id: str, dims: int = 1536):
    """Create the per-tenant OpenSearch index if it doesn't exist."""
    index = tenant_index(tenant_id)
    try:
        _os_request("GET", index)
        logger.info("Index %s already exists", index)
    except Exception:
        mapping = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "tenant_id":  {"type": "keyword"},
                    "doc_id":     {"type": "keyword"},
                    "doc_type":   {"type": "keyword"},
                    "product":    {"type": "keyword"},
                    "version":    {"type": "keyword"},
                    "tags":       {"type": "keyword"},
                    "title":      {"type": "text"},
                    "chunk_text": {"type": "text"},
                    "embedding":  {
                        "type":      "knn_vector",
                        "dimension": dims,
                        "method": {
                            "name":       "hnsw",
                            "space_type": "cosinesimil",
                            "engine":     "faiss",
                        },
                    },
                    "ingested_at": {"type": "date"},
                }
            },
        }
        _os_request("PUT", index, mapping)
        logger.info("Created index %s", index)

def index_chunks(tenant_id: str, doc_meta: Dict, chunks: List[str]):
    """Index all chunks into the tenant's OpenSearch index."""
    index = tenant_index(tenant_id)
    for i, chunk in enumerate(chunks):
        vec = embed(chunk)
        doc = {
            **doc_meta,
            "chunk_index": i,
            "chunk_text":  chunk,
            "embedding":   vec,
            "ingested_at": datetime.utcnow().isoformat(),
        }
        _os_request("POST", f"{index}/_doc", doc)

def vector_search(
    tenant_id: str,
    query_vec: List[float],
    product: Optional[str] = None,
    top_k: int = TOP_K_RESULTS,
) -> List[Dict]:
    """KNN search within a tenant's index, optionally filtered by product."""
    index = tenant_index(tenant_id)
    knn_query: Dict[str, Any] = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {"vector": query_vec, "k": top_k}
            }
        },
        "_source": ["doc_id", "doc_type", "product", "title", "chunk_text", "tags"],
    }
    if product:
        knn_query["query"] = {
            "bool": {
                "must": [{"knn": {"embedding": {"vector": query_vec, "k": top_k}}}],
                "filter": [{"term": {"product": product}}],
            }
        }
    resp  = _os_request("GET", f"{index}/_search", knn_query)
    hits  = resp.get("hits", {}).get("hits", [])
    return [{"score": h["_score"], **h["_source"]} for h in hits]

# ── Session / conversation management ───────────────────────────────────────

def create_session(tenant_id: str, user_id: str) -> str:
    session_id  = str(uuid.uuid4())
    expires_at  = (datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS)).isoformat()
    sessions_table.put_item(Item={
        "session_id":   session_id,
        "tenant_id":    tenant_id,
        "user_id":      user_id,
        "history":      [],
        "miss_count":   0,
        "created_at":   datetime.utcnow().isoformat(),
        "expires_at":   expires_at,
        "escalated":    False,
    })
    return session_id

def get_session(session_id: str) -> Dict:
    resp = sessions_table.get_item(Key={"session_id": session_id})
    item = resp.get("Item")
    if not item:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return item

def append_turn(session_id: str, user_msg: str, assistant_msg: str, escalated: bool = False):
    """Append a conversation turn and cap history length."""
    session = get_session(session_id)
    history: List[Dict] = session.get("history", [])
    history.append({
        "role": "user",
        "content": user_msg,
        "ts": datetime.utcnow().isoformat(),
    })
    history.append({
        "role": "assistant",
        "content": assistant_msg,
        "ts": datetime.utcnow().isoformat(),
    })
    # Keep only the most recent N turns
    history = history[-(MAX_CONVERSATION_TURNS * 2):]

    miss_count = session.get("miss_count", 0)
    sessions_table.update_item(
        Key={"session_id": session_id},
        UpdateExpression="SET #h = :h, miss_count = :m, escalated = :e",
        ExpressionAttributeNames={"#h": "history"},
        ExpressionAttributeValues={
            ":h": history,
            ":m": miss_count,
            ":e": escalated,
        },
    )

def increment_miss(session_id: str) -> int:
    resp = sessions_table.update_item(
        Key={"session_id": session_id},
        UpdateExpression="ADD miss_count :one",
        ExpressionAttributeValues={":one": 1},
        ReturnValues="UPDATED_NEW",
    )
    return int(resp["Attributes"]["miss_count"])

# ── Answer generation ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful customer support assistant for {company_name}.
Answer the user's question using ONLY the provided knowledge base excerpts.
If the excerpts do not contain enough information to answer confidently, say so clearly and suggest escalating to a human agent.
Always be concise, friendly, and professional.
If you reference a specific document, mention its title.
Do not invent information that is not in the excerpts."""

def build_context(hits: List[Dict]) -> Tuple[str, float]:
    """Format retrieved chunks into a context block and compute avg confidence."""
    if not hits:
        return "", 0.0
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(
            f"[{i}] Title: {h.get('title','N/A')} | Type: {h.get('doc_type','')}\n"
            f"{h['chunk_text']}"
        )
    avg_score = sum(h["score"] for h in hits) / len(hits)
    return "\n\n---\n\n".join(parts), avg_score

def generate_answer(
    company_name: str,
    context: str,
    history: List[Dict],
    user_message: str,
) -> str:
    messages = []
    for turn in history[-10:]:          # last 5 turns for context
        messages.append({"role": turn["role"], "content": turn["content"]})

    user_content = (
        f"Knowledge base excerpts:\n\n{context}\n\n---\n\nUser question: {user_message}"
        if context
        else user_message
    )
    messages.append({"role": "user", "content": user_content})

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT.format(company_name=company_name),
        "messages": messages,
    })
    resp = bedrock_client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(resp["body"].read())
    return result["content"][0]["text"]

# ── Escalation ───────────────────────────────────────────────────────────────

def escalate_to_human(session: Dict, reason: EscalationReason, notes: Optional[str]):
    """Push escalation event to SQS for routing to human agent queue."""
    if not SQS_ESCALATION_URL:
        logger.warning("SQS_ESCALATION_URL not set — escalation skipped")
        return
    payload = {
        "event":      "escalation",
        "session_id": session["session_id"],
        "tenant_id":  session["tenant_id"],
        "user_id":    session["user_id"],
        "reason":     reason,
        "notes":      notes,
        "history":    session.get("history", [])[-10:],
        "timestamp":  datetime.utcnow().isoformat(),
    }
    sqs_client.send_message(
        QueueUrl=SQS_ESCALATION_URL,
        MessageBody=json.dumps(payload),
        MessageGroupId=session["tenant_id"],  # FIFO group by tenant
    )
    logger.info("Escalated session %s — reason: %s", session["session_id"], reason)

# ── API routes ───────────────────────────────────────────────────────────────

@app.post("/tenants", summary="Register a new tenant")
def register_tenant(body: TenantCreate):
    tenants_table.put_item(Item={
        "tenant_id":  body.tenant_id,
        "name":       body.name,
        "plan":       body.plan,
        "metadata":   body.metadata,
        "created_at": datetime.utcnow().isoformat(),
    })
    ensure_tenant_index(body.tenant_id)
    return {"tenant_id": body.tenant_id, "status": "created"}

@app.get("/tenants/{tenant_id}", summary="Fetch tenant config")
def fetch_tenant(tenant_id: str):
    return get_tenant(tenant_id)

@app.post("/ingest", summary="Ingest a support document into the knowledge base")
async def ingest_document(body: DocumentIngest, background_tasks: BackgroundTasks):
    """
    Fetches the document from S3, extracts text, chunks it, embeds each
    chunk, and indexes into the tenant-isolated OpenSearch index.
    """
    get_tenant(body.tenant_id)   # 404 guard

    def _ingest():
        try:
            raw_text = extract_text_from_s3(S3_BUCKET, body.s3_key)
            chunks   = chunk_text(raw_text)
            doc_id   = hashlib.sha256(body.s3_key.encode()).hexdigest()[:16]
            doc_meta = {
                "tenant_id": body.tenant_id,
                "doc_id":    doc_id,
                "doc_type":  body.doc_type,
                "product":   body.product,
                "version":   body.version,
                "tags":      body.tags,
                "title":     body.title,
                "s3_key":    body.s3_key,
            }
            index_chunks(body.tenant_id, doc_meta, chunks)
            logger.info(
                "Ingested %d chunks for doc '%s' (tenant=%s)",
                len(chunks), body.title, body.tenant_id,
            )
        except Exception as exc:
            logger.error("Ingest failed for %s: %s", body.s3_key, exc, exc_info=True)

    background_tasks.add_task(_ingest)
    return {"status": "ingestion_queued", "s3_key": body.s3_key}

@app.post("/chat", summary="Send a message and receive an AI-generated answer")
def chat(body: ChatMessage):
    """
    Core conversational RAG endpoint.
    1. Load or create session
    2. Embed user message
    3. KNN search in tenant index (optionally filtered by product)
    4. Generate answer with Claude
    5. Auto-escalate if confidence is too low or miss count is high
    """
    tenant = get_tenant(body.tenant_id)

    # Session management
    if body.session_id:
        session = get_session(body.session_id)
        if session["tenant_id"] != body.tenant_id:
            raise HTTPException(status_code=403, detail="Session belongs to different tenant")
        if session.get("escalated"):
            return {"session_id": body.session_id, "escalated": True,
                    "answer": "This conversation has been escalated to a human agent. They will be with you shortly."}
        session_id = body.session_id
    else:
        session_id = create_session(body.tenant_id, body.user_id)
        session    = get_session(session_id)

    # Retrieve relevant chunks
    query_vec = embed(body.message)
    hits      = vector_search(body.tenant_id, query_vec, product=body.product)
    context, confidence = build_context(hits)

    # Detect low confidence
    low_confidence = confidence < CONFIDENCE_THRESHOLD

    # Generate answer
    answer = generate_answer(
        company_name=tenant.get("name", "our company"),
        context=context,
        history=session.get("history", []),
        user_message=body.message,
    )

    # Track misses and check escalation threshold
    escalated = False
    if low_confidence or not hits:
        miss_count = increment_miss(session_id)
        if miss_count >= 3:
            escalate_to_human(session, EscalationReason.REPEATED_MISS, notes="3 consecutive low-confidence responses")
            escalated = True

    # Persist turn
    append_turn(session_id, body.message, answer, escalated=escalated)

    return {
        "session_id":  session_id,
        "answer":      answer,
        "confidence":  round(confidence, 4),
        "sources":     [{"title": h.get("title"), "doc_type": h.get("doc_type")} for h in hits],
        "escalated":   escalated,
    }

@app.post("/escalate", summary="Manually escalate a session to a human agent")
def manual_escalate(body: EscalateRequest):
    session = get_session(body.session_id)
    if session["tenant_id"] != body.tenant_id:
        raise HTTPException(status_code=403, detail="Session belongs to different tenant")
    escalate_to_human(session, body.reason, body.notes)
    sessions_table.update_item(
        Key={"session_id": body.session_id},
        UpdateExpression="SET escalated = :t",
        ExpressionAttributeValues={":t": True},
    )
    return {"session_id": body.session_id, "status": "escalated"}

@app.get("/sessions/{session_id}", summary="Retrieve session history")
def get_session_history(session_id: str, tenant_id: str):
    session = get_session(session_id)
    if session["tenant_id"] != tenant_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    return {
        "session_id": session_id,
        "user_id":    session["user_id"],
        "history":    session.get("history", []),
        "escalated":  session.get("escalated", False),
        "miss_count": session.get("miss_count", 0),
        "created_at": session.get("created_at"),
    }

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("customer_support_rag:app", host="0.0.0.0", port=8000, reload=False)
```
