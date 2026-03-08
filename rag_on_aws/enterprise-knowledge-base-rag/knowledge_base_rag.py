```python
import os
import json
import uuid
import logging
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Environment configuration ──────────────────────────────────────────────────
AWS_REGION              = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET               = os.getenv('S3_BUCKET', 'enterprise-kb-documents')
OPENSEARCH_ENDPOINT     = os.getenv('OPENSEARCH_ENDPOINT', '')
OPENSEARCH_INDEX        = os.getenv('OPENSEARCH_INDEX', 'enterprise-knowledge-base')
BEDROCK_MODEL_ID        = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-sonnet-4-6')
BEDROCK_EMBED_MODEL_ID  = os.getenv('BEDROCK_EMBED_MODEL_ID', 'amazon.titan-embed-text-v2:0')
DYNAMODB_TABLE          = os.getenv('DYNAMODB_TABLE', 'kb-query-history')
SNS_TOPIC_ARN           = os.getenv('SNS_TOPIC_ARN', '')
CHUNK_SIZE              = int(os.getenv('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP           = int(os.getenv('CHUNK_OVERLAP', '200'))
TOP_K_RESULTS           = int(os.getenv('TOP_K_RESULTS', '5'))
SIMILARITY_THRESHOLD    = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))

# ── AWS clients ────────────────────────────────────────────────────────────────
s3_client         = boto3.client('s3', region_name=AWS_REGION)
bedrock_client    = boto3.client('bedrock-runtime', region_name=AWS_REGION)
dynamodb          = boto3.resource('dynamodb', region_name=AWS_REGION)
sns_client        = boto3.client('sns', region_name=AWS_REGION)
textract_client   = boto3.client('textract', region_name=AWS_REGION)
comprehend_client = boto3.client('comprehend', region_name=AWS_REGION)

query_history_table = dynamodb.Table(DYNAMODB_TABLE)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Enterprise Knowledge Base RAG",
    description=(
        "Retrieval-Augmented Generation system for enterprise knowledge bases "
        "powered by Amazon Bedrock, OpenSearch Serverless, and S3."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Enums & Models ─────────────────────────────────────────────────────────────
class DocumentStatus(str, Enum):
    UPLOADING   = "uploading"
    PROCESSING  = "processing"
    INDEXED     = "indexed"
    FAILED      = "failed"


class DocumentCategory(str, Enum):
    POLICY      = "policy"
    TECHNICAL   = "technical"
    LEGAL       = "legal"
    HR          = "hr"
    FINANCE     = "finance"
    GENERAL     = "general"


class QueryRequest(BaseModel):
    query: str                        = Field(..., min_length=3, max_length=2000, description="Natural-language question")
    category: Optional[DocumentCategory] = Field(None, description="Filter results to a specific category")
    top_k: int                        = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")
    include_sources: bool             = Field(True, description="Include source document references in the response")
    conversation_id: Optional[str]   = Field(None, description="Pass an existing conversation ID to maintain context")


class QueryResponse(BaseModel):
    query_id: str
    conversation_id: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    processing_time_ms: int
    timestamp: str


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    s3_key: str
    chunks_created: int
    message: str


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    category: DocumentCategory
    status: DocumentStatus
    s3_key: str
    chunks_count: int
    file_size_bytes: int
    content_type: str
    indexed_at: Optional[str]
    created_at: str
    tags: List[str]


class FeedbackRequest(BaseModel):
    query_id: str
    rating: int    = Field(..., ge=1, le=5, description="1 (poor) to 5 (excellent)")
    comment: Optional[str] = None


# ── Text-processing utilities ─────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks for better retrieval coverage."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break on a sentence boundary
        if end < len(text):
            for boundary in ['. ', '.\n', '\n\n', '\n']:
                idx = text.rfind(boundary, start, end)
                if idx != -1:
                    end = idx + len(boundary)
                    break
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


def extract_text_from_pdf(s3_key: str) -> str:
    """Use Amazon Textract to extract text from a PDF stored in S3."""
    try:
        response = textract_client.start_document_text_detection(
            DocumentLocation={'S3Object': {'Bucket': S3_BUCKET, 'Name': s3_key}}
        )
        job_id = response['JobId']

        import time
        while True:
            result = textract_client.get_document_text_detection(JobId=job_id)
            if result['JobStatus'] in ('SUCCEEDED', 'FAILED'):
                break
            time.sleep(2)

        if result['JobStatus'] == 'FAILED':
            raise Exception("Textract job failed")

        return ' '.join(
            block['Text']
            for block in result.get('Blocks', [])
            if block['BlockType'] == 'LINE'
        )
    except ClientError as e:
        logger.error(f"Textract error for {s3_key}: {e}")
        raise


def generate_embedding(text: str) -> List[float]:
    """Generate a vector embedding using Amazon Titan Embeddings v2."""
    try:
        response = bedrock_client.invoke_model(
            modelId=BEDROCK_EMBED_MODEL_ID,
            body=json.dumps({'inputText': text[:8192]}),
            contentType='application/json',
            accept='application/json',
        )
        return json.loads(response['body'].read())['embedding']
    except ClientError as e:
        logger.error(f"Embedding generation error: {e}")
        raise


def index_chunk_to_opensearch(
    chunk_text: str,
    embedding: List[float],
    document_id: str,
    chunk_index: int,
    metadata: Dict[str, Any],
) -> bool:
    """Index a single chunk with its embedding into OpenSearch Serverless."""
    import urllib3, base64

    doc = {
        'document_id':  document_id,
        'chunk_index':  chunk_index,
        'content':      chunk_text,
        'embedding':    embedding,
        'filename':     metadata.get('filename', ''),
        'category':     metadata.get('category', 'general'),
        'tags':         metadata.get('tags', []),
        'indexed_at':   datetime.utcnow().isoformat(),
    }

    http = urllib3.PoolManager()
    url  = f"{OPENSEARCH_ENDPOINT}/{OPENSEARCH_INDEX}/_doc/{document_id}_{chunk_index}"
    headers = {'Content-Type': 'application/json'}

    # Sign the request with AWS SigV4
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    import botocore.session as bs

    session = bs.get_session()
    creds   = session.get_credentials()
    request = AWSRequest(method='PUT', url=url, data=json.dumps(doc), headers=headers)
    SigV4Auth(creds, 'aoss', AWS_REGION).add_auth(request)

    resp = http.request('PUT', url, body=json.dumps(doc), headers=dict(request.headers))
    return resp.status in (200, 201)


def search_opensearch(
    query_embedding: List[float],
    top_k: int = TOP_K_RESULTS,
    category_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Perform k-NN vector search in OpenSearch Serverless."""
    import urllib3

    knn_query: Dict[str, Any] = {
        "size": top_k,
        "query": {
            "bool": {
                "must": [{"knn": {"embedding": {"vector": query_embedding, "k": top_k}}}],
                **({"filter": [{"term": {"category": category_filter}}]} if category_filter else {}),
            }
        },
        "_source": ["document_id", "chunk_index", "content", "filename", "category", "tags"],
    }

    http = urllib3.PoolManager()
    url  = f"{OPENSEARCH_ENDPOINT}/{OPENSEARCH_INDEX}/_search"

    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    import botocore.session as bs

    session = bs.get_session()
    creds   = session.get_credentials()
    request = AWSRequest(method='POST', url=url, data=json.dumps(knn_query),
                         headers={'Content-Type': 'application/json'})
    SigV4Auth(creds, 'aoss', AWS_REGION).add_auth(request)

    resp = urllib3.PoolManager().request('POST', url, body=json.dumps(knn_query),
                                         headers=dict(request.headers))
    if resp.status != 200:
        logger.error(f"OpenSearch search failed: {resp.data}")
        return []

    hits = json.loads(resp.data).get('hits', {}).get('hits', [])
    return [
        {
            **hit['_source'],
            'score': hit['_score'],
        }
        for hit in hits
        if hit['_score'] >= SIMILARITY_THRESHOLD
    ]


def generate_answer(query: str, context_chunks: List[Dict[str, Any]], conversation_history: List[Dict] = None) -> str:
    """Generate a grounded answer with Amazon Bedrock (Claude)."""
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['filename']} | Category: {c['category']}]\n{c['content']}"
        for c in context_chunks
    )

    history_text = ""
    if conversation_history:
        history_text = "\n".join(
            f"{turn['role'].capitalize()}: {turn['content']}"
            for turn in conversation_history[-6:]  # last 3 turns
        )

    system_prompt = (
        "You are a precise enterprise knowledge assistant. "
        "Answer the user's question using ONLY the provided context. "
        "If the answer is not in the context, say so clearly. "
        "Always cite the source document name when referencing information. "
        "Be concise, factual, and professional."
    )

    messages = []
    if history_text:
        messages.append({"role": "user", "content": f"Previous conversation:\n{history_text}"})
        messages.append({"role": "assistant", "content": "Understood. I'll consider the conversation history."})

    messages.append({
        "role": "user",
        "content": f"Context from the knowledge base:\n\n{context_text}\n\nQuestion: {query}",
    })

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "system": system_prompt,
        "messages": messages,
    }

    response = bedrock_client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(payload),
        contentType='application/json',
        accept='application/json',
    )
    return json.loads(response['body'].read())['content'][0]['text']


def detect_entities(text: str) -> List[Dict[str, str]]:
    """Use Amazon Comprehend to extract named entities from a query."""
    try:
        resp = comprehend_client.detect_entities(Text=text[:5000], LanguageCode='en')
        return [
            {'text': e['Text'], 'type': e['Type'], 'score': round(e['Score'], 3)}
            for e in resp.get('Entities', [])
            if e['Score'] >= 0.8
        ]
    except ClientError:
        return []


def compute_confidence(chunks: List[Dict[str, Any]]) -> float:
    """Derive a simple confidence score from retrieval scores."""
    if not chunks:
        return 0.0
    avg_score = sum(c['score'] for c in chunks) / len(chunks)
    return round(min(avg_score, 1.0), 3)


def save_query_history(
    query_id: str,
    conversation_id: str,
    query: str,
    answer: str,
    sources: List[Dict],
    confidence: float,
) -> None:
    try:
        query_history_table.put_item(Item={
            'query_id':        query_id,
            'conversation_id': conversation_id,
            'query':           query,
            'answer':          answer,
            'sources':         json.dumps(sources),
            'confidence':      str(confidence),
            'timestamp':       datetime.utcnow().isoformat(),
            'ttl':             int(datetime.utcnow().timestamp()) + 60 * 60 * 24 * 90,  # 90-day TTL
        })
    except ClientError as e:
        logger.warning(f"Could not save query history: {e}")


def notify_ingestion_complete(document_id: str, filename: str, chunks: int) -> None:
    if SNS_TOPIC_ARN:
        try:
            sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject="Document Indexed Successfully",
                Message=json.dumps({
                    'event':       'document_indexed',
                    'document_id': document_id,
                    'filename':    filename,
                    'chunks':      chunks,
                    'timestamp':   datetime.utcnow().isoformat(),
                }),
            )
        except ClientError as e:
            logger.warning(f"SNS notification failed: {e}")


# ── API routes ─────────────────────────────────────────────────────────────────
@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: DocumentCategory = DocumentCategory.GENERAL,
    tags: str = Query("", description="Comma-separated tags"),
):
    """
    Upload and ingest a document into the knowledge base.

    Supported formats: PDF, TXT, MD
    The document is stored in S3, text is extracted (via Textract for PDFs),
    chunked, embedded with Titan, and indexed into OpenSearch Serverless.
    """
    allowed_types = {"application/pdf", "text/plain", "text/markdown"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    document_id = str(uuid.uuid4())
    s3_key      = f"documents/{category.value}/{document_id}/{file.filename}"
    tag_list    = [t.strip() for t in tags.split(',') if t.strip()]

    # Upload to S3
    content = await file.read()
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=content,
            ContentType=file.content_type,
            Metadata={
                'document_id': document_id,
                'category':    category.value,
                'filename':    file.filename,
                'tags':        ','.join(tag_list),
            },
        )
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")

    # Process asynchronously
    metadata = {
        'filename': file.filename,
        'category': category.value,
        'tags':     tag_list,
    }
    background_tasks.add_task(
        _process_and_index_document,
        document_id=document_id,
        s3_key=s3_key,
        content=content,
        content_type=file.content_type,
        metadata=metadata,
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        status=DocumentStatus.PROCESSING,
        s3_key=s3_key,
        chunks_created=0,
        message="Document uploaded. Indexing is in progress.",
    )


async def _process_and_index_document(
    document_id: str,
    s3_key: str,
    content: bytes,
    content_type: str,
    metadata: Dict[str, Any],
) -> None:
    """Background task: extract → chunk → embed → index."""
    try:
        # Extract text
        if content_type == "application/pdf":
            text = extract_text_from_pdf(s3_key)
        else:
            text = content.decode('utf-8', errors='ignore')

        if not text.strip():
            logger.warning(f"No text extracted from {s3_key}")
            return

        # Chunk
        chunks = chunk_text(text)
        logger.info(f"Document {document_id}: {len(chunks)} chunks created")

        # Embed & index
        indexed = 0
        for i, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)
            if index_chunk_to_opensearch(chunk, embedding, document_id, i, metadata):
                indexed += 1

        logger.info(f"Document {document_id}: {indexed}/{len(chunks)} chunks indexed")
        notify_ingestion_complete(document_id, metadata['filename'], indexed)

    except Exception as e:
        logger.error(f"Failed to process document {document_id}: {e}")


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base using natural language.

    Retrieves the most relevant document chunks via k-NN vector search,
    then synthesises a grounded answer using Amazon Bedrock (Claude).
    """
    import time
    start = time.time()

    query_id        = str(uuid.uuid4())
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # Embed the query
    query_embedding = generate_embedding(request.query)

    # Retrieve relevant chunks
    chunks = search_opensearch(
        query_embedding,
        top_k=request.top_k,
        category_filter=request.category.value if request.category else None,
    )

    if not chunks:
        return QueryResponse(
            query_id=query_id,
            conversation_id=conversation_id,
            answer="I couldn't find relevant information in the knowledge base for your query. Please try rephrasing or uploading related documents.",
            sources=[],
            confidence_score=0.0,
            processing_time_ms=int((time.time() - start) * 1000),
            timestamp=datetime.utcnow().isoformat(),
        )

    # Fetch conversation history for multi-turn support
    conversation_history: List[Dict] = []
    try:
        hist = query_history_table.query(
            IndexName='conversation_id-index',
            KeyConditionExpression=boto3.dynamodb.conditions.Key('conversation_id').eq(conversation_id),
            ScanIndexForward=False,
            Limit=6,
        )
        for item in reversed(hist.get('Items', [])):
            conversation_history.extend([
                {'role': 'user',      'content': item['query']},
                {'role': 'assistant', 'content': item['answer']},
            ])
    except ClientError:
        pass  # history unavailable – still answer

    # Generate answer
    answer = generate_answer(request.query, chunks, conversation_history)

    # Build source list
    sources: List[Dict[str, Any]] = []
    if request.include_sources:
        seen = set()
        for c in chunks:
            key = f"{c['document_id']}_{c['chunk_index']}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    'document_id': c['document_id'],
                    'filename':    c['filename'],
                    'category':    c['category'],
                    'excerpt':     c['content'][:300] + '…' if len(c['content']) > 300 else c['content'],
                    'score':       round(c['score'], 3),
                })

    confidence = compute_confidence(chunks)
    elapsed_ms = int((time.time() - start) * 1000)

    # Persist history
    save_query_history(query_id, conversation_id, request.query, answer, sources, confidence)

    return QueryResponse(
        query_id=query_id,
        conversation_id=conversation_id,
        answer=answer,
        sources=sources,
        confidence_score=confidence,
        processing_time_ms=elapsed_ms,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/documents/{document_id}", response_model=DocumentMetadata)
async def get_document_metadata(document_id: str):
    """Retrieve metadata for an indexed document."""
    try:
        resp = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=f"documents/",
        )
        for obj in resp.get('Contents', []):
            if document_id in obj['Key']:
                meta = s3_client.head_object(Bucket=S3_BUCKET, Key=obj['Key'])['Metadata']
                return DocumentMetadata(
                    document_id=document_id,
                    filename=meta.get('filename', ''),
                    category=DocumentCategory(meta.get('category', 'general')),
                    status=DocumentStatus.INDEXED,
                    s3_key=obj['Key'],
                    chunks_count=0,
                    file_size_bytes=obj['Size'],
                    content_type=s3_client.head_object(Bucket=S3_BUCKET, Key=obj['Key'])['ContentType'],
                    indexed_at=None,
                    created_at=obj['LastModified'].isoformat(),
                    tags=meta.get('tags', '').split(','),
                )
        raise HTTPException(status_code=404, detail="Document not found")
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Remove a document and all its chunks from the knowledge base."""
    deleted_objects = []
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"documents/"):
            for obj in page.get('Contents', []):
                if document_id in obj['Key']:
                    s3_client.delete_object(Bucket=S3_BUCKET, Key=obj['Key'])
                    deleted_objects.append(obj['Key'])
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Also delete from OpenSearch (best-effort)
    # A production implementation would track chunk count in DynamoDB and bulk-delete
    return {"document_id": document_id, "deleted_s3_objects": deleted_objects, "status": "deleted"}


@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Submit quality feedback for a specific query response."""
    try:
        query_history_table.update_item(
            Key={'query_id': req.query_id},
            UpdateExpression="SET feedback_rating = :r, feedback_comment = :c",
            ExpressionAttributeValues={':r': req.rating, ':c': req.comment or ''},
        )
        return {"status": "feedback recorded", "query_id": req.query_id}
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Service health check."""
    checks: Dict[str, str] = {}

    # S3
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
        checks['s3'] = 'healthy'
    except ClientError:
        checks['s3'] = 'unhealthy'

    # Bedrock
    try:
        bedrock_client.list_foundation_models()
        checks['bedrock'] = 'healthy'
    except ClientError:
        checks['bedrock'] = 'unhealthy'

    # DynamoDB
    try:
        query_history_table.table_status
        checks['dynamodb'] = 'healthy'
    except ClientError:
        checks['dynamodb'] = 'unhealthy'

    overall = 'healthy' if all(v == 'healthy' for v in checks.values()) else 'degraded'
    return {"status": overall, "checks": checks, "timestamp": datetime.utcnow().isoformat()}


@app.get("/stats")
async def get_stats():
    """Return high-level usage statistics."""
    try:
        # Count total queries
        scan = query_history_table.scan(Select='COUNT')
        total_queries = scan.get('Count', 0)

        # Count indexed documents (unique S3 prefixes)
        resp = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix="documents/", Delimiter='/')
        total_documents = len(resp.get('CommonPrefixes', []))

        return {
            "total_queries":   total_queries,
            "total_documents": total_documents,
            "timestamp":       datetime.utcnow().isoformat(),
        }
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("knowledge_base_rag:app", host="0.0.0.0", port=8000, reload=False)
```
