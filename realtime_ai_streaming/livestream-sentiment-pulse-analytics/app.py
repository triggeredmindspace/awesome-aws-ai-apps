```python
import asyncio
import base64
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== Configuration ====================
class Config:
    """Application configuration from environment variables."""
    
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    KINESIS_VIDEO_STREAM = os.getenv("KINESIS_VIDEO_STREAM", "sentiment-video-stream")
    KINESIS_DATA_STREAM = os.getenv("KINESIS_DATA_STREAM", "sentiment-data-stream")
    DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "sentiment-analytics")
    DYNAMODB_EVENTS_TABLE = os.getenv("DYNAMODB_EVENTS_TABLE", "live-events")
    DYNAMODB_HIGHLIGHTS_TABLE = os.getenv("DYNAMODB_HIGHLIGHTS_TABLE", "event-highlights")
    S3_BUCKET = os.getenv("S3_BUCKET", "sentiment-analytics-data")
    BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
    ALERT_SNS_TOPIC = os.getenv("ALERT_SNS_TOPIC", "")
    CLOUDWATCH_NAMESPACE = os.getenv("CLOUDWATCH_NAMESPACE", "SentimentAnalytics")
    SENTIMENT_THRESHOLD_NEGATIVE = float(os.getenv("SENTIMENT_THRESHOLD_NEGATIVE", "-0.5"))
    SENTIMENT_THRESHOLD_POSITIVE = float(os.getenv("SENTIMENT_THRESHOLD_POSITIVE", "0.5"))
    VIRAL_MOMENT_THRESHOLD = int(os.getenv("VIRAL_MOMENT_THRESHOLD", "100"))
    S3_ENCRYPTION_TYPE = os.getenv("S3_ENCRYPTION_TYPE", "AES256")  # AES256 or aws:kms
    S3_KMS_KEY_ID = os.getenv("S3_KMS_KEY_ID", "")  # Optional KMS key ID for aws:kms encryption
    MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "10000"))  # Maximum input text length
    
    # API_KEY must be set - no default value
    API_KEY = os.environ.get("API_KEY")
    if not API_KEY:
        raise ValueError("API_KEY environment variable must be set")


# ==================== Input Sanitization ====================
class InputSanitizer:
    """Sanitizes user input to prevent prompt injection and other attacks."""
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = Config.MAX_INPUT_LENGTH) -> str:
        """
        Sanitize text input to prevent prompt injection.
        
        Args:
            text: Raw text input
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
            
        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Limit length
        if len(text) > max_length:
            logger.warning(f"Input text truncated from {len(text)} to {max_length} characters")
            text = text[:max_length]
        
        # Remove control characters except common whitespace
        text = ''.join(char for char in text if char.isprintable() or char in '\n\r\t ')
        
        # Escape special characters that could be used for injection
        # Remove or escape common prompt injection patterns
        injection_patterns = [
            r'<\|.*?\|>',  # Special tokens
            r'\[INST\].*?\[/INST\]',  # Instruction markers
            r'###\s*System:',  # System prompts
            r'###\s*Assistant:',  # Assistant markers
            r'Human:.*?Assistant:',  # Conversation markers
        ]
        
        for pattern in injection_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Final validation
        if not text or len(text.strip()) == 0:
            raise ValueError("Input text is empty after sanitization")
        
        return text.strip()
    
    @staticmethod
    def validate_and_sanitize(text: str) -> str:
        """
        Validate and sanitize text input with comprehensive checks.
        
        Args:
            text: Raw text input
            
        Returns:
            Sanitized and validated text
            
        Raises:
            ValueError: If input fails validation
        """
        if text is None:
            raise ValueError("Input text cannot be None")
        
        # Sanitize
        sanitized = InputSanitizer.sanitize_text(text)
        
        # Additional validation
        if len(sanitized) < 1:
            raise ValueError("Input text too short after sanitization")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'ignore\s+previous\s+instructions',
            r'disregard\s+all\s+prior',
            r'forget\s+everything',
            r'new\s+instructions:',
            r'system\s+override',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                logger.warning(f"Suspicious pattern detected in input: {pattern}")
                # Remove the suspicious content
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized


# ==================== API Key Authentication ====================
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def validate_api_key(api_key: str = Security(api_key_header)):
    """
    Validate API key from request header.
    
    Args:
        api_key: API key from X-API-Key header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    if api_key != Config.API_KEY:
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key


# ==================== Data Models ====================
class SentimentType(str, Enum):
    """Sentiment classification types."""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"


class EmotionType(str, Enum):
    """Emotion classification types."""
    HAPPY = "HAPPY"
    SAD = "SAD"
    ANGRY = "ANGRY"
    CONFUSED = "CONFUSED"
    DISGUSTED = "DISGUSTED"
    SURPRISED = "SURPRISED"
    CALM = "CALM"
    FEAR = "FEAR"


class StreamSource(str, Enum):
    """Supported streaming sources."""
    YOUTUBE = "YOUTUBE"
    TWITCH = "TWITCH"
    RTMP = "RTMP"
    CUSTOM = "CUSTOM"


class LiveEvent(BaseModel):
    """Live event configuration."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    stream_source: StreamSource
    stream_url: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "active"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SentimentData(BaseModel):
    """Sentiment analysis result."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_id: str
    sentiment: SentimentType
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    emotions: Dict[EmotionType, float] = Field(default_factory=dict)
    text_content: Optional[str] = None
    source_type: str = "video"  # video, audio, text, combined
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EngagementMetrics(BaseModel):
    """Real-time engagement metrics."""
    event_id: str
    timestamp: datetime
    viewer_count: int = 0
    comment_rate: float = 0.0
    reaction_count: int = 0
    average_sentiment: float = 0.0
    dominant_emotion: Optional[EmotionType] = None
    engagement_score: float = 0.0


class Alert(BaseModel):
    """Sentiment alert configuration."""
    alert_id: str = Field(default_factory=lambda: str(uuid4()))
    event_id: str
    alert_type: str  # sentiment_shift, viral_moment, low_engagement
    severity: str  # low, medium, high, critical
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(default_factory=dict)


class Highlight(BaseModel):
    """Event highlight based on emotional peaks."""
    highlight_id: str = Field(default_factory=lambda: str(uuid4()))
    event_id: str
    start_time: datetime
    end_time: datetime
    peak_emotion: EmotionType
    peak_sentiment_score: float
    engagement_score: float
    description: str
    video_url: Optional[str] = None


# ==================== AWS Service Clients ====================
class AWSServiceManager:
    """Manages AWS service clients with proper error handling."""
    
    def __init__(self):
        """Initialize AWS service clients."""
        self.region = Config.AWS_REGION
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize clients with error handling
        try:
            self.kinesis_video = boto3.client('kinesisvideo', region_name=self.region)
            logger.info("Kinesis Video client initialized successfully")
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error(f"Failed to initialize Kinesis Video client - missing or invalid AWS credentials: {str(e)}")
            raise
        except ClientError as e:
            logger.error(f"Failed to initialize Kinesis Video client - insufficient permissions or service error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Kinesis Video client - unexpected error: {str(e)}")
            raise
        
        try:
            self.kinesis_data = boto3.client('kinesis', region_name=self.region)
            logger.info("Kinesis Data client initialized successfully")
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error(f"Failed to initialize Kinesis Data client - missing or invalid AWS credentials: {str(e)}")
            raise
        except ClientError as e:
            logger.error(f"Failed to initialize Kinesis Data client - insufficient permissions or service error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Kinesis Data client - unexpected error: {str(e)}")
            raise
        
        try:
            self.dynamodb = boto3.resource('dynamodb', region_name=self.region)
            logger.info("DynamoDB resource initialized successfully")
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error(f"Failed to initialize DynamoDB resource - missing or invalid AWS credentials: {str(e)}")
            raise
        except ClientError as e:
            logger.error(f"Failed to initialize DynamoDB resource - insufficient permissions or service error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize DynamoDB resource - unexpected error: {str(e)}")
            raise
        
        try:
            self.rekognition = boto3.client('rekognition', region_name=self.region)
            logger.info("Rekognition client initialized successfully")
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error(f"Failed to initialize Rekognition client - missing or invalid AWS credentials: {str(e)}")
            raise
        except ClientError as e:
            logger.error(f"Failed to initialize Rekognition client - insufficient permissions or service error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Rekognition client - unexpected error: {str(e)}")
            raise
        
        try:
            self.transcribe = boto3.client('transcribe', region_name=self.region)
            logger.info("Transcribe client initialized successfully")
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error(f"Failed to initialize Transcribe client - missing or invalid AWS credentials: {str(e)}")
            raise
        except ClientError as e:
            logger.error(f"Failed to initialize Transcribe client - insufficient permissions or service error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Transcribe client - unexpected error: {str(e)}")
            raise
        
        try:
            self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=self.region)
            logger.info("Bedrock Runtime client initialized successfully")
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error(f"Failed to initialize Bedrock Runtime client - missing or invalid AWS credentials: {str(e)}")
            raise
        except ClientError as e:
            logger.error(f"Failed to initialize Bedrock Runtime client - insufficient permissions or service error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock Runtime client - unexpected error: {str(e)}")
            raise
        
        try:
            self.s3 = boto3.client('s3', region_name=self.region)
            logger.info("S3 client initialized successfully")
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error(f"Failed to initialize S3 client - missing or invalid AWS credentials: {str(e)}")
            raise
        except ClientError as e:
            logger.error(f"Failed to initialize S3 client - insufficient permissions or service error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize S3 client - unexpected error: {str(e)}")
            raise
        
        try:
            self.cloudwatch = boto3.client('cloudwatch', region_name=self.region)
            logger.info("CloudWatch client initialized successfully")
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error(f"Failed to initialize CloudWatch client - missing or invalid AWS credentials: {str(e)}")
            raise
        except ClientError as e:
            logger.error(f"Failed to initialize CloudWatch client - insufficient permissions or service error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize CloudWatch client - unexpected error: {str(e)}")
            raise
        
        try:
            self.