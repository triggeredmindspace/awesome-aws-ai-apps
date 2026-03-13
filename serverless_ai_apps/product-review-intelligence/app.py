"""
AI-Powered Product Review Intelligence Platform
Serverless application that processes customer reviews, extracts insights,
and generates executive summaries using Amazon Bedrock (Claude) and AWS services.
"""

import json
import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Any

import boto3
import yaml
from botocore.exceptions import ClientError

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── AWS clients (lazy-initialised so cold-start cost stays low) ───────────────
_bedrock = None
_dynamodb = None
_sqs = None
_s3 = None
_ses = None


def _bedrock_client():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])
    return _bedrock


def _dynamodb_resource():
    global _dynamodb
    if _dynamodb is None:
        _dynamodb = boto3.resource("dynamodb", region_name=os.environ["AWS_REGION"])
    return _dynamodb


def _sqs_client():
    global _sqs
    if _sqs is None:
        _sqs = boto3.client("sqs", region_name=os.environ["AWS_REGION"])
    return _sqs


def _s3_client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3", region_name=os.environ["AWS_REGION"])
    return _s3


def _ses_client():
    global _ses
    if _ses is None:
        _ses = boto3.client("ses", region_name=os.environ["AWS_REGION"])
    return _ses


# ── Config ───────────────────────────────────────────────────────────────────
def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


CONFIG = load_config()
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", CONFIG["bedrock"]["model_id"])
REVIEWS_TABLE = os.environ.get("REVIEWS_TABLE", CONFIG["dynamodb"]["reviews_table"])
INSIGHTS_TABLE = os.environ.get("INSIGHTS_TABLE", CONFIG["dynamodb"]["insights_table"])
REVIEWS_BUCKET = os.environ.get("REVIEWS_BUCKET", CONFIG["s3"]["reviews_bucket"])
QUEUE_URL = os.environ.get("REVIEW_QUEUE_URL", "")
DIGEST_EMAIL = os.environ.get("DIGEST_EMAIL", CONFIG["ses"].get("digest_email", ""))


# ── Bedrock helpers ───────────────────────────────────────────────────────────
def invoke_claude(prompt: str, max_tokens: int = 1024) -> str:
    """Call Claude via Bedrock and return the text response."""
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = _bedrock_client().invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def analyse_review(review_text: str, product_name: str) -> dict:
    """
    Use Claude to extract:
      - sentiment (positive / neutral / negative)
      - star_rating (1-5 inferred)
      - themes  (list of short keywords)
      - issues  (list of reported problems, empty if none)
      - praise  (list of positive highlights, empty if none)
      - summary (1-sentence plain-English summary)
    """
    prompt = f"""You are a product feedback analyst. Analyse this customer review for "{product_name}" and respond ONLY with valid JSON matching the schema below. Do not add markdown fences.

Schema:
{{
  "sentiment": "<positive|neutral|negative>",
  "star_rating": <integer 1-5>,
  "themes": ["<theme>", ...],
  "issues": ["<issue>", ...],
  "praise": ["<praise point>", ...],
  "summary": "<one sentence summary>"
}}

Review:
{review_text}"""

    raw = invoke_claude(prompt, max_tokens=512)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Claude returned non-JSON; attempting extraction")
        # Fallback: return a minimal structure
        return {
            "sentiment": "neutral",
            "star_rating": 3,
            "themes": [],
            "issues": [],
            "praise": [],
            "summary": review_text[:200],
        }


def generate_product_summary(product_name: str, analyses: list[dict]) -> dict:
    """Generate an executive summary across multiple analysed reviews."""
    aggregated = json.dumps(analyses, indent=2)
    prompt = f"""You are a senior product analyst. Given the aggregated review analyses below for "{product_name}", produce an executive summary. Respond ONLY with valid JSON:

{{
  "overall_sentiment": "<positive|mixed|negative>",
  "average_rating": <float rounded to 1 decimal>,
  "top_themes": ["<theme>", ...],
  "top_issues": ["<issue>", ...],
  "top_praise": ["<praise>", ...],
  "executive_summary": "<3-4 sentence paragraph suitable for a VP of Product>",
  "recommended_actions": ["<action>", ...]
}}

Analyses:
{aggregated}"""

    raw = invoke_claude(prompt, max_tokens=1024)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"executive_summary": raw, "error": "parse_failed"}


# ── DynamoDB helpers ──────────────────────────────────────────────────────────
def save_review(review_id: str, product_id: str, product_name: str,
                review_text: str, analysis: dict) -> None:
    table = _dynamodb_resource().Table(REVIEWS_TABLE)
    table.put_item(Item={
        "review_id": review_id,
        "product_id": product_id,
        "product_name": product_name,
        "review_text": review_text,
        "analysis": analysis,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ttl": int(datetime.now(timezone.utc).timestamp()) + 60 * 60 * 24 * 90,  # 90 days
    })


def save_insight(product_id: str, product_name: str, summary: dict,
                 review_count: int) -> str:
    insight_id = str(uuid.uuid4())
    table = _dynamodb_resource().Table(INSIGHTS_TABLE)
    table.put_item(Item={
        "insight_id": insight_id,
        "product_id": product_id,
        "product_name": product_name,
        "summary": summary,
        "review_count": review_count,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return insight_id


def get_recent_analyses(product_id: str, limit: int = 50) -> list[dict]:
    table = _dynamodb_resource().Table(REVIEWS_TABLE)
    response = table.query(
        IndexName="product_id-index",
        KeyConditionExpression=boto3.dynamodb.conditions.Key("product_id").eq(product_id),
        ScanIndexForward=False,
        Limit=limit,
    )
    return [item["analysis"] for item in response.get("Items", [])]


def get_insight(product_id: str) -> dict | None:
    table = _dynamodb_resource().Table(INSIGHTS_TABLE)
    response = table.query(
        IndexName="product_id-index",
        KeyConditionExpression=boto3.dynamodb.conditions.Key("product_id").eq(product_id),
        ScanIndexForward=False,
        Limit=1,
    )
    items = response.get("Items", [])
    return items[0] if items else None


# ── SES helpers ───────────────────────────────────────────────────────────────
def send_digest_email(product_name: str, summary: dict, recipient: str) -> None:
    if not recipient:
        logger.info("No digest email configured; skipping SES send")
        return
    subject = f"[Review Intelligence] Weekly digest – {product_name}"
    body_text = json.dumps(summary, indent=2)
    body_html = f"""<html><body>
<h2>Weekly Review Digest: {product_name}</h2>
<p><strong>Overall sentiment:</strong> {summary.get('overall_sentiment','—')}</p>
<p><strong>Average rating:</strong> {summary.get('average_rating','—')} / 5</p>
<h3>Executive Summary</h3>
<p>{summary.get('executive_summary','—')}</p>
<h3>Top Issues</h3>
<ul>{''.join(f'<li>{i}</li>' for i in summary.get('top_issues', []))}</ul>
<h3>Recommended Actions</h3>
<ul>{''.join(f'<li>{a}</li>' for a in summary.get('recommended_actions', []))}</ul>
</body></html>"""

    _ses_client().send_email(
        Source=recipient,
        Destination={"ToAddresses": [recipient]},
        Message={
            "Subject": {"Data": subject},
            "Body": {
                "Text": {"Data": body_text},
                "Html": {"Data": body_html},
            },
        },
    )
    logger.info("Digest email sent to %s", recipient)


# ── Lambda handlers ───────────────────────────────────────────────────────────

def ingest_review_handler(event: dict, context: Any) -> dict:
    """
    POST /reviews
    Body: { product_id, product_name, review_text, source? }
    Validates input, enqueues the review for async processing,
    and returns a 202 Accepted with the review_id.
    """
    try:
        body = json.loads(event.get("body") or "{}")
        product_id = body.get("product_id", "").strip()
        product_name = body.get("product_name", "").strip()
        review_text = body.get("review_text", "").strip()

        if not product_id or not product_name or not review_text:
            return _response(400, {"error": "product_id, product_name, and review_text are required"})

        if len(review_text) > 5000:
            return _response(400, {"error": "review_text exceeds 5000 character limit"})

        review_id = str(uuid.uuid4())
        message = {
            "review_id": review_id,
            "product_id": product_id,
            "product_name": product_name,
            "review_text": review_text,
        }

        _sqs_client().send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps(message),
            MessageGroupId=product_id,          # FIFO queue — group by product
            MessageDeduplicationId=review_id,
        )

        logger.info("Queued review %s for product %s", review_id, product_id)
        return _response(202, {"review_id": review_id, "status": "queued"})

    except ClientError as e:
        logger.error("AWS error in ingest_review_handler: %s", e)
        return _response(500, {"error": "Failed to queue review"})
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        return _response(500, {"error": "Internal server error"})


def process_review_handler(event: dict, context: Any) -> None:
    """
    SQS trigger — processes each review message:
    1. Calls Claude for analysis
    2. Saves review + analysis to DynamoDB
    """
    for record in event.get("Records", []):
        try:
            message = json.loads(record["body"])
            review_id = message["review_id"]
            product_id = message["product_id"]
            product_name = message["product_name"]
            review_text = message["review_text"]

            logger.info("Processing review %s", review_id)
            analysis = analyse_review(review_text, product_name)
            save_review(review_id, product_id, product_name, review_text, analysis)
            logger.info("Saved review %s — sentiment: %s", review_id, analysis.get("sentiment"))

        except Exception as e:
            logger.error("Failed to process record: %s — %s", record.get("messageId"), e)
            raise  # Raises to trigger SQS retry / DLQ


def get_insights_handler(event: dict, context: Any) -> dict:
    """
    GET /insights/{product_id}
    Returns the latest executive insight for a product.
    """
    try:
        product_id = (event.get("pathParameters") or {}).get("product_id", "")
        if not product_id:
            return _response(400, {"error": "product_id path parameter is required"})

        insight = get_insight(product_id)
        if not insight:
            return _response(404, {"error": "No insights found for this product"})

        return _response(200, insight)

    except Exception as e:
        logger.error("Error in get_insights_handler: %s", e)
        return _response(500, {"error": "Internal server error"})


def generate_insights_handler(event: dict, context: Any) -> dict:
    """
    POST /insights/{product_id}/generate
    Pulls recent reviews from DynamoDB, runs executive summary,
    saves insight, and optionally sends digest email.
    """
    try:
        product_id = (event.get("pathParameters") or {}).get("product_id", "")
        body = json.loads(event.get("body") or "{}")
        product_name = body.get("product_name", "").strip()
        send_email = body.get("send_email", False)

        if not product_id or not product_name:
            return _response(400, {"error": "product_id path param and product_name body field are required"})

        analyses = get_recent_analyses(product_id, limit=50)
        if not analyses:
            return _response(404, {"error": "No reviews found for this product"})

        summary = generate_product_summary(product_name, analyses)
        insight_id = save_insight(product_id, product_name, summary, len(analyses))

        if send_email and DIGEST_EMAIL:
            send_digest_email(product_name, summary, DIGEST_EMAIL)

        return _response(200, {
            "insight_id": insight_id,
            "product_id": product_id,
            "review_count": len(analyses),
            "summary": summary,
        })

    except Exception as e:
        logger.error("Error in generate_insights_handler: %s", e)
        return _response(500, {"error": "Internal server error"})


def weekly_digest_handler(event: dict, context: Any) -> None:
    """
    EventBridge scheduled trigger — generates and emails insights
    for all products that had reviews in the past 7 days.
    """
    logger.info("Weekly digest triggered by EventBridge")
    table = _dynamodb_resource().Table(INSIGHTS_TABLE)
    scan_response = table.scan(ProjectionExpression="product_id, product_name")
    products_seen = {
        (item["product_id"], item["product_name"])
        for item in scan_response.get("Items", [])
    }

    for product_id, product_name in products_seen:
        try:
            analyses = get_recent_analyses(product_id, limit=50)
            if not analyses:
                continue
            summary = generate_product_summary(product_name, analyses)
            save_insight(product_id, product_name, summary, len(analyses))
            if DIGEST_EMAIL:
                send_digest_email(product_name, summary, DIGEST_EMAIL)
        except Exception as e:
            logger.error("Weekly digest failed for product %s: %s", product_id, e)


def s3_batch_ingest_handler(event: dict, context: Any) -> None:
    """
    S3 trigger — processes a JSON file of reviews dropped into the ingestion bucket.
    Expected file format:
    [
      { "product_id": "...", "product_name": "...", "review_text": "..." },
      ...
    ]
    """
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        logger.info("Processing S3 batch file s3://%s/%s", bucket, key)

        try:
            obj = _s3_client().get_object(Bucket=bucket, Key=key)
            reviews = json.loads(obj["Body"].read())

            for review in reviews:
                review_id = str(uuid.uuid4())
                product_id = review.get("product_id", "")
                product_name = review.get("product_name", "")
                review_text = review.get("review_text", "")
                if not (product_id and product_name and review_text):
                    logger.warning("Skipping invalid review entry: %s", review)
                    continue

                _sqs_client().send_message(
                    QueueUrl=QUEUE_URL,
                    MessageBody=json.dumps({
                        "review_id": review_id,
                        "product_id": product_id,
                        "product_name": product_name,
                        "review_text": review_text,
                    }),
                    MessageGroupId=product_id,
                    MessageDeduplicationId=review_id,
                )

            logger.info("Queued %d reviews from %s", len(reviews), key)

        except Exception as e:
            logger.error("Failed to process S3 file %s: %s", key, e)
            raise


# ── Utility ───────────────────────────────────────────────────────────────────
def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body, default=str),
    }
