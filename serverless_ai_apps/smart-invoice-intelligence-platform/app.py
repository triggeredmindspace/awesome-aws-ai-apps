```python
import os
import json
import uuid
import logging
import hashlib
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any
from enum import Enum

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
from jose import JWTError, jwt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET = os.getenv('S3_BUCKET', 'invoice-intelligence-bucket')
DYNAMODB_INVOICES_TABLE = os.getenv('DYNAMODB_INVOICES_TABLE', 'invoices')
DYNAMODB_VENDORS_TABLE = os.getenv('DYNAMODB_VENDORS_TABLE', 'vendors')
DYNAMODB_PO_TABLE = os.getenv('DYNAMODB_PO_TABLE', 'purchase_orders')
DYNAMODB_AUDIT_TABLE = os.getenv('DYNAMODB_AUDIT_TABLE', 'audit_trail')
SNS_TOPIC_ARN = os.getenv('SNS_TOPIC_ARN', '')
COGNITO_USER_POOL_ID = os.getenv('COGNITO_USER_POOL_ID', '')
COGNITO_CLIENT_ID = os.getenv('COGNITO_CLIENT_ID', '')
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key')
JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
SAGEMAKER_ENDPOINT = os.getenv('SAGEMAKER_ENDPOINT', 'invoice-anomaly-detector')
EXCHANGE_RATE_API_KEY = os.getenv('EXCHANGE_RATE_API_KEY', '')
WEBHOOK_URLS = os.getenv('WEBHOOK_URLS', '').split(',')

# Initialize AWS clients
s3_client = boto3.client('s3', region_name=AWS_REGION)
textract_client = boto3.client('textract', region_name=AWS_REGION)
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
sns_client = boto3.client('sns', region_name=AWS_REGION)
sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
eventbridge_client = boto3.client('events', region_name=AWS_REGION)
cloudwatch_client = boto3.client('cloudwatch', region_name=AWS_REGION)
cognito_client = boto3.client('cognito-idp', region_name=AWS_REGION)

# Initialize DynamoDB tables
invoices_table = dynamodb.Table(DYNAMODB_INVOICES_TABLE)
vendors_table = dynamodb.Table(DYNAMODB_VENDORS_TABLE)
po_table = dynamodb.Table(DYNAMODB_PO_TABLE)
audit_table = dynamodb.Table(DYNAMODB_AUDIT_TABLE)

# FastAPI app initialization
app = FastAPI(
    title="Smart Invoice Intelligence Platform",
    description="AI-powered invoice processing and validation system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()


# Enums
class InvoiceStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    VALIDATED = "validated"
    APPROVED = "approved"
    REJECTED = "rejected"
    PAID = "paid"
    ANOMALY_DETECTED = "anomaly_detected"


class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"


# Pydantic models
class InvoiceLineItem(BaseModel):
    description: str
    quantity: float
    unit_price: Decimal
    total: Decimal
    tax_amount: Optional[Decimal] = None


class InvoiceData(BaseModel):
    invoice_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    invoice_number: str
    vendor_name: str
    vendor_id: Optional[str] = None
    invoice_date: str
    due_date: str
    currency: Currency
    subtotal: Decimal
    tax_amount: Decimal
    total_amount: Decimal
    line_items: List[InvoiceLineItem]
    po_number: Optional[str] = None
    status: InvoiceStatus = InvoiceStatus.PENDING
    confidence_score: Optional[float] = None
    anomaly_flags: List[str] = []
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    class Config:
        use_enum_values = True


class VendorData(BaseModel):
    vendor_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vendor_name: str
    tax_id: Optional[str] = None
    address: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    payment_terms: Optional[str] = None
    average_invoice_amount: Optional[Decimal] = None
    total_invoices: int = 0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class PurchaseOrder(BaseModel):
    po_number: str
    vendor_id: str
    po_date: str
    expected_amount: Decimal
    currency: Currency
    line_items: List[InvoiceLineItem]
    status: str = "open"
    matched_invoice_id: Optional[str] = None


class ApprovalWorkflow(BaseModel):
    invoice_id: str
    approver_email: str
    approval_status: str = "pending"
    comments: Optional[str] = None
    approved_at: Optional[str] = None


class AnalyticsQuery(BaseModel):
    start_date: str
    end_date: str
    vendor_id: Optional[str] = None
    currency: Optional[Currency] = None


# Helper functions
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Verify JWT token and return user information.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        Decoded token payload
        
    Raises:
        HTTPException: If token is invalid
    """
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        logger.error(f"Token verification failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")


def log_audit_trail(
    action: str,
    resource_type: str,
    resource_id: str,
    user_id: str,
    details: Dict[str, Any]
) -> None:
    """
    Log audit trail entry to DynamoDB.
    
    Args:
        action: Action performed
        resource_type: Type of resource
        resource_id: Resource identifier
        user_id: User who performed action
        details: Additional details
    """
    try:
        audit_entry = {
            'audit_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().