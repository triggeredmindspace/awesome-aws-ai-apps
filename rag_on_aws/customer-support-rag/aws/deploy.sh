#!/usr/bin/env bash
# deploy.sh — Deploy Multi-Tenant Customer Support RAG to AWS
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
STACK_NAME="${STACK_NAME:-customer-support-rag}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
AWS_REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET_NAME="${S3_BUCKET_NAME:-customer-support-docs}"
TEMPLATE="$(dirname "$0")/cloudformation/template.yaml"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[deploy]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $*"; }
err()  { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# ── Pre-flight ────────────────────────────────────────────────────────────────
command -v aws  &>/dev/null || err "AWS CLI not found"
command -v python3 &>/dev/null || err "python3 not found"
aws sts get-caller-identity --region "$AWS_REGION" &>/dev/null \
  || err "AWS credentials not configured or invalid"

AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
log "Account: ${AWS_ACCOUNT} | Region: ${AWS_REGION} | Env: ${ENVIRONMENT}"

# ── Deploy CloudFormation ─────────────────────────────────────────────────────
log "Deploying CloudFormation stack: ${STACK_NAME}-${ENVIRONMENT}"
aws cloudformation deploy \
  --template-file "$TEMPLATE" \
  --stack-name "${STACK_NAME}-${ENVIRONMENT}" \
  --region "$AWS_REGION" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
      EnvironmentName="$ENVIRONMENT" \
      S3BucketName="$S3_BUCKET_NAME" \
  --no-fail-on-empty-changeset

log "Stack deployed successfully"

# ── Read outputs ──────────────────────────────────────────────────────────────
get_output() {
  aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}-${ENVIRONMENT}" \
    --region "$AWS_REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" \
    --output text
}

DOCS_BUCKET=$(get_output DocumentsBucketName)
SESSIONS_TABLE=$(get_output SessionsTableName)
TENANTS_TABLE=$(get_output TenantsTableName)
ESCALATION_QUEUE=$(get_output EscalationQueueUrl)
OSS_ENDPOINT=$(get_output OSSCollectionEndpoint)
APP_ROLE_ARN=$(get_output AppRoleArn)

log "Resources:"
log "  S3 Bucket:         ${DOCS_BUCKET}"
log "  Sessions Table:    ${SESSIONS_TABLE}"
log "  Tenants Table:     ${TENANTS_TABLE}"
log "  Escalation Queue:  ${ESCALATION_QUEUE}"
log "  OpenSearch:        ${OSS_ENDPOINT}"
log "  App Role:          ${APP_ROLE_ARN}"

# ── Write .env file ───────────────────────────────────────────────────────────
ENV_FILE="$(dirname "$0")/../../.env.${ENVIRONMENT}"
cat > "$ENV_FILE" <<EOF
AWS_REGION=${AWS_REGION}
S3_BUCKET=${DOCS_BUCKET}
OPENSEARCH_ENDPOINT=${OSS_ENDPOINT}
DYNAMODB_SESSIONS=${SESSIONS_TABLE}
DYNAMODB_TENANTS=${TENANTS_TABLE}
SQS_ESCALATION_URL=${ESCALATION_QUEUE}
BEDROCK_MODEL_ID=anthropic.claude-sonnet-4-6
BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
CONFIDENCE_THRESHOLD=0.72
SESSION_TTL_HOURS=24
MAX_CONVERSATION_TURNS=20
EOF
log "Environment file written to: ${ENV_FILE}"

# ── Install Python dependencies ───────────────────────────────────────────────
REQUIREMENTS="$(dirname "$0")/../../requirements.txt"
if [[ -f "$REQUIREMENTS" ]]; then
  log "Installing Python dependencies..."
  python3 -m pip install -r "$REQUIREMENTS" --quiet
fi

# ── OpenSearch: wait for collection to be active ──────────────────────────────
log "Waiting for OpenSearch Serverless collection to become ACTIVE..."
for i in $(seq 1 30); do
  STATUS=$(aws opensearchserverless list-collections \
    --region "$AWS_REGION" \
    --query "collectionSummaries[?name=='support-${ENVIRONMENT}'].status" \
    --output text 2>/dev/null || echo "UNKNOWN")
  if [[ "$STATUS" == "ACTIVE" ]]; then
    log "Collection is ACTIVE"
    break
  fi
  warn "Collection status: ${STATUS} — waiting (${i}/30)..."
  sleep 10
done

[[ "$STATUS" == "ACTIVE" ]] || err "Collection did not become ACTIVE in time"

# ── Smoke test ────────────────────────────────────────────────────────────────
log "Running smoke test against /health endpoint..."
export $(grep -v '^#' "$ENV_FILE" | xargs)

python3 - <<'PYEOF'
import os, sys, subprocess, time

proc = subprocess.Popen(
    ["python3", "-m", "uvicorn", "customer_support_rag:app", "--host", "127.0.0.1", "--port", "18080"],
    cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."),
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
time.sleep(4)
try:
    import urllib.request
    with urllib.request.urlopen("http://127.0.0.1:18080/health", timeout=5) as r:
        body = r.read()
        assert b"ok" in body, f"Unexpected response: {body}"
    print("[smoke] /health OK")
finally:
    proc.terminate()
PYEOF

log "Deployment complete for environment: ${ENVIRONMENT}"
log ""
log "Next steps:"
log "  1. Register a tenant:  POST /tenants"
log "  2. Upload documents to s3://${DOCS_BUCKET}/<tenant_id>/"
log "  3. Ingest docs:        POST /ingest"
log "  4. Start chatting:     POST /chat"
log "  5. Monitor escalations in SQS: ${ESCALATION_QUEUE}"
