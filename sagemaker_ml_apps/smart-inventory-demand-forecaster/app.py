```python
import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from decimal import Decimal
from enum import Enum
import re

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import pandas as pd
import numpy as np
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET = os.getenv('S3_BUCKET', 'inventory-forecaster-data')
OPENSEARCH_ENDPOINT = os.getenv('OPENSEARCH_ENDPOINT', '')
SAGEMAKER_ROLE_ARN = os.getenv('SAGEMAKER_ROLE_ARN', '')
API_GATEWAY_URL = os.getenv('API_GATEWAY_URL', '')
EVENTBRIDGE_BUS_NAME = os.getenv('EVENTBRIDGE_BUS_NAME', 'inventory-events')
CLOUDWATCH_NAMESPACE = os.getenv('CLOUDWATCH_NAMESPACE', 'InventoryForecaster')
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '').split(',') if os.getenv('ALLOWED_ORIGINS') else []

# Initialize AWS clients
s3_client = boto3.client('s3', region_name=AWS_REGION)
sagemaker_client = boto3.client('sagemaker', region_name=AWS_REGION)
sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
lambda_client = boto3.client('lambda', region_name=AWS_REGION)
apigateway_client = boto3.client('apigatewayv2', region_name=AWS_REGION)
ecs_client = boto3.client('ecs', region_name=AWS_REGION)
eventbridge_client = boto3.client('events', region_name=AWS_REGION)
cloudwatch_client = boto3.client('cloudwatch', region_name=AWS_REGION)
secretsmanager_client = boto3.client('secretsmanager', region_name=AWS_REGION)

# Initialize FastAPI app
app = FastAPI(
    title="Smart Inventory Demand Forecaster",
    description="AI-powered inventory management and demand forecasting system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ForecastRequest(BaseModel):
    product_id: str = Field(..., description="Unique product identifier")
    warehouse_id: str = Field(..., description="Warehouse identifier")
    forecast_horizon: int = Field(30, ge=1, le=365, description="Days to forecast")
    include_confidence_intervals: bool = Field(True)


class InventoryAlert(BaseModel):
    product_id: str
    warehouse_id: str
    current_stock: int
    reorder_point: int
    severity: AlertSeverity
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReorderCalculation(BaseModel):
    product_id: str
    warehouse_id: str
    current_stock: int
    average_daily_demand: float
    lead_time_days: int
    safety_stock: int
    reorder_point: int
    economic_order_quantity: int


class WhatIfScenario(BaseModel):
    product_id: str
    warehouse_id: str
    scenario_name: str
    demand_multiplier: float = Field(1.0, ge=0.1, le=10.0)
    duration_days: int = Field(7, ge=1, le=90)
    start_date: datetime


class AnomalyDetection(BaseModel):
    product_id: str
    warehouse_id: str
    date: datetime
    expected_demand: float
    actual_demand: float
    anomaly_score: float
    is_anomaly: bool


# OpenSearch client initialization
def get_opensearch_client() -> Optional[OpenSearch]:
    """Initialize and return OpenSearch client with AWS authentication."""
    if not OPENSEARCH_ENDPOINT:
        logger.warning("OpenSearch endpoint not configured")
        return None
    
    try:
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials is None:
            logger.error("AWS credentials not found. Ensure IAM role is properly configured.")
            return None
        
        # Refresh credentials to ensure they're current
        frozen_credentials = credentials.get_frozen_credentials()
        
        if not frozen_credentials.access_key or not frozen_credentials.secret_key:
            logger.error("Invalid AWS credentials: missing access key or secret key")
            return None
        
        awsauth = AWS4Auth(
            frozen_credentials.access_key,
            frozen_credentials.secret_key,
            AWS_REGION,
            'es',
            session_token=frozen_credentials.token if frozen_credentials.token else None
        )
        
        client = OpenSearch(
            hosts=[{'host': OPENSEARCH_ENDPOINT, 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
        return client
    except NoCredentialsError as e:
        logger.error(f"No AWS credentials found: {str(e)}. Please configure IAM role or credentials.")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize OpenSearch client: {str(e)}")
        return None


opensearch_client = get_opensearch_client()


# Utility functions
def publish_cloudwatch_metric(metric_name: str, value: float, unit: str = 'Count', 
                              dimensions: Optional[List[Dict]] = None):
    """Publish custom metrics to CloudWatch."""
    try:
        metric_data = {
            'MetricName': metric_name,
            'Value': value,
            'Unit': unit,
            'Timestamp': datetime.utcnow()
        }
        
        if dimensions:
            metric_data['Dimensions'] = dimensions
        
        cloudwatch_client.put_metric_data(
            Namespace=CLOUDWATCH_NAMESPACE,
            MetricData=[metric_data]
        )
        logger.info(f"Published metric {metric_name}: {value}")
    except ClientError as e:
        logger.error(f"Failed to publish CloudWatch metric: {str(e)}")


def publish_event(detail_type: str, detail: Dict[str, Any]):
    """Publish event to EventBridge."""
    try:
        eventbridge_client.put_events(
            Entries=[{
                'Source': 'inventory.forecaster',
                'DetailType': detail_type,
                'Detail': json.dumps(detail, default=str),
                'EventBusName': EVENTBRIDGE_BUS_NAME
            }]
        )
        logger.info(f"Published event: {detail_type}")
    except ClientError as e:
        logger.error(f"Failed to publish event: {str(e)}")


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """Sanitize string values to prevent injection attacks."""
    if not isinstance(value, str):
        return str(value)
    
    # Remove control characters and limit length
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
    sanitized = sanitized[:max_length]
    
    # Remove potential script tags and SQL injection patterns
    sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
    
    return sanitized.strip()


def validate_and_sanitize_document(document: Dict[str, Any], allowed_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """Validate and sanitize document content before indexing."""
    if not isinstance(document, dict):
        raise ValueError("Document must be a dictionary")
    
    sanitized_doc = {}
    
    # Define allowed field types and their validators
    allowed_types = (str, int, float, bool, datetime, type(None))
    
    for key, value in document.items():
        # Sanitize key
        sanitized_key = sanitize_string(str(key), max_length=255)
        
        # Skip if key is empty after sanitization
        if not sanitized_key:
            logger.warning(f"Skipping invalid key: {key}")
            continue
        
        # Check if field is allowed (if whitelist provided)
        if allowed_fields and sanitized_key not in allowed_fields:
            logger.warning(f"Skipping non-whitelisted field: {sanitized_key}")
            continue
        
        # Sanitize value based on type
        if value is None:
            sanitized_doc[sanitized_key] = None
        elif isinstance(value, bool):
            sanitized_doc[sanitized_key] = value
        elif isinstance(value, (int, float)):
            # Validate numeric ranges
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                logger.warning(f"Skipping invalid numeric value for {sanitized_key}")
                continue
            sanitized_doc[sanitized_key] = value
        elif isinstance(value, str):
            sanitized_doc[sanitized_key] = sanitize_string(value)
        elif isinstance(value, datetime):
            sanitized_doc[sanitized_key] = value.isoformat()
        elif isinstance(value, (list, tuple)):
            # Recursively sanitize list items
            sanitized_list = []
            for item in value[:100]:  # Limit list size
                if isinstance(item, dict):
                    sanitized_list.append(validate_and_sanitize_document(item))
                elif isinstance(item, allowed_types):
                    if isinstance(item, str):
                        sanitized_list.append(sanitize_string(item))
                    else:
                        sanitized_list.append(item)
            sanitized_doc[sanitized_key] = sanitized_list
        elif isinstance(value, dict):
            # Recursively sanitize nested dictionaries
            sanitized_doc[sanitized_key] = validate_and_sanitize_document(value)
        else:
            # Convert other types to string and sanitize
            sanitized_doc[sanitized_key] = sanitize_string(str(value))
    
    return sanitized_doc


def store_in_opensearch(index: str, document: Dict[str, Any], doc_id: Optional[str] = None):
    """Store document in OpenSearch for analysis and visualization."""
    if not opensearch_client:
        logger.warning("OpenSearch client not available")
        return
    
    try:
        # Validate and sanitize the document
        sanitized_document = validate_and_sanitize_document(document)
        
        # Sanitize index name
        sanitized_index = re.sub(r'[^a-z0-9_\-]', '', index.lower())
        if not sanitized_index:
            raise ValueError("Invalid index name")
        
        # Sanitize doc_id if provided
        if doc_id:
            sanitized_doc_id = sanitize_string(str(doc_id), max_length=512)
            opensearch_client.index(index=sanitized_index, body=sanitized_document, id=sanitized_doc_id)
        else:
            opensearch_client.index(index=sanitized_index, body=sanitized_document)
        
        logger.info(f"Stored document in OpenSearch index: {sanitized_index}")
    except ValueError as e:
        logger.error(f"Validation error storing in OpenSearch: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to store in OpenSearch: {str(e)}")


def load_historical_data(product_id: str, warehouse_id: str, days: int = 365) -> pd.DataFrame:
    """Load historical sales data from S3."""
    try:
        key = f"historical_data/{warehouse_id}/{product_id}.csv"
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        
        # Define expected schema
        expected_columns = ['date', 'demand', 'price', 'promotions']
        expected_dtypes = {
            'demand': 'float64',
            'price': 'float64',
            'promotions': 'float64'
        }
        
        try:
            # Read CSV with error handling
            df = pd.read_csv(
                obj['Body'],
                parse_dates=['date'],
                on_bad_lines='skip',
                encoding='utf-8'
            )
            
            # Validate required columns exist
            missing_columns = set(expected_columns) - set(df.columns)
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                raise ValueError(f"CSV missing required columns: {missing_columns}")
            
            # Validate and convert data types
            for col, dtype in expected_dtypes.items():
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as e:
                        logger.error(f"Failed to convert column {col} to numeric: {str(e)}")
                        raise ValueError(f"Invalid data type in column {col}")
            
            # Validate date column
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                try:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                except Exception as e:
                    logger.error(f"Failed to parse date column: {str(e)}")
                    raise ValueError("Invalid date format in CSV")
            
            # Remove rows with invalid data
            initial_rows = len(df)
            df = df.dropna(subset=['date', 'demand'])
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                logger.warning(f"Dropped {dropped_rows} rows with invalid data")
            
            # Validate data ranges
            if (df['demand'] < 0).any():
                logger.warning("Negative demand values found, setting to 0")
                df.loc[df['demand'] < 0, 'demand'] = 0
            
            if (df['price'] < 0).any():
                logger.warning("Negative price values found, setting to 0")
                df.loc[df['price'] < 0, 'price'] = 0
            
            # Filter to requested time period
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            df = df[df['date'] >= cutoff_date]
            
            if df.empty:
                logger.warning(f"No valid data found for {product_id} in {warehouse_id} after filtering")
                return pd.DataFrame(columns=expected_columns)
            
            return df.sort_values('date')
            
        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing error for {product_id}: {str(e)}")