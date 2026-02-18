```python
import asyncio
import base64
import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from contextlib import asynccontextmanager

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
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))  # Requests per window
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # Window in seconds
    
    # API_KEY must be set - no default value
    API_KEY = os.environ.get("API_KEY")
    if not API_KEY:
        raise ValueError("API_KEY environment variable must be set")


# ==================== Rate Limiter ====================
class RateLimiter:
    """Rate limiting and anomaly detection for API requests."""
    
    def __init__(self):
        self.request_counts = defaultdict(list)
        self.blocked_ips = {}
        
    def check_rate_limit(self, client_id: str) -> bool:
        """
        Check if client has exceeded rate limit.
        
        Args:
            client_id: Client identifier (IP address or API key)
            
        Returns:
            True if request is allowed, False if rate limited
        """
        now = time.time()
        window_start = now - Config.RATE_LIMIT_WINDOW
        
        # Check if client is blocked
        if client_id in self.blocked_ips:
            block_until = self.blocked_ips[client_id]
            if now < block_until:
                logger.warning(f"Blocked client {client_id} attempted request")
                return False
            else:
                del self.blocked_ips[client_id]
        
        # Clean old requests
        self.request_counts[client_id] = [
            req_time for req_time in self.request_counts[client_id]
            if req_time > window_start
        ]
        
        # Check rate limit
        if len(self.request_counts[client_id]) >= Config.RATE_LIMIT_REQUESTS:
            logger.warning(f"Rate limit exceeded for client {client_id}")
            # Block for 5 minutes
            self.blocked_ips[client_id] = now + 300
            return False
        
        # Add current request
        self.request_counts[client_id].append(now)
        return True
    
    def detect_anomaly(self, client_id: str, text: str) -> bool:
        """
        Detect anomalous behavior patterns.
        
        Args:
            client_id: Client identifier
            text: Input text to analyze
            
        Returns:
            True if anomaly detected, False otherwise
        """
        # Check for repeated identical requests (potential attack)
        recent_requests = self.request_counts[client_id][-10:]
        if len(recent_requests) >= 5:
            # If more than 5 requests in last 10, check for suspicious patterns
            if len(text) > Config.MAX_INPUT_LENGTH * 0.9:
                logger.warning(f"Anomaly detected: Large input from {client_id}")
                return True
        
        return False


# ==================== Input Sanitization ====================
class InputSanitizer:
    """Sanitizes user input to prevent prompt injection and other attacks."""
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize Unicode characters to prevent Unicode-based bypasses.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Normalize to NFKC form (compatibility decomposition + canonical composition)
        text = unicodedata.normalize('NFKC', text)
        
        # Remove zero-width characters and other invisible Unicode
        invisible_chars = [
            '\u200b',  # Zero-width space
            '\u200c',  # Zero-width non-joiner
            '\u200d',  # Zero-width joiner
            '\ufeff',  # Zero-width no-break space
            '\u2060',  # Word joiner
            '\u180e',  # Mongolian vowel separator
        ]
        for char in invisible_chars:
            text = text.replace(char, '')
        
        return text
    
    @staticmethod
    def decode_common_encodings(text: str) -> str:
        """
        Detect and decode common encoding attempts.
        
        Args:
            text: Input text
            
        Returns:
            Decoded text
        """
        # Try to detect base64 encoding
        try:
            # Check if text looks like base64
            if re.match(r'^[A-Za-z0-9+/]+=*$', text.replace('\n', '').replace('\r', '')):
                decoded = base64.b64decode(text).decode('utf-8', errors='ignore')
                if decoded and len(decoded) > 0:
                    logger.warning("Base64 encoded input detected and decoded")
                    text = decoded
        except Exception:
            pass
        
        # Decode URL encoding
        try:
            import urllib.parse
            decoded = urllib.parse.unquote(text)
            if decoded != text:
                logger.warning("URL encoded input detected and decoded")
                text = decoded
        except Exception:
            pass
        
        # Decode HTML entities
        try:
            import html
            decoded = html.unescape(text)
            if decoded != text:
                logger.warning("HTML encoded input detected and decoded")
                text = decoded
        except Exception:
            pass
        
        return text
    
    @staticmethod
    def detect_prompt_injection(text: str) -> bool:
        """
        Detect prompt injection attempts using multiple techniques.
        
        Args:
            text: Input text to check
            
        Returns:
            True if injection detected, False otherwise
        """
        # Normalize text for detection (lowercase, remove extra spaces)
        normalized = ' '.join(text.lower().split())
        
        # Comprehensive injection patterns
        injection_patterns = [
            # Instruction override attempts
            r'ignore\s+(?:previous|prior|all|above)\s+(?:instructions?|prompts?|commands?)',
            r'disregard\s+(?:previous|prior|all|above)',
            r'forget\s+(?:previous|prior|all|above|everything)',
            r'new\s+(?:instructions?|prompts?|commands?)\s*:',
            r'system\s+(?:override|prompt|instruction)',
            
            # Role manipulation
            r'you\s+are\s+(?:now|a)\s+(?:developer|admin|root|system)',
            r'act\s+as\s+(?:a\s+)?(?:developer|admin|root|system)',
            r'pretend\s+(?:to\s+be|you\s+are)',
            
            # Delimiter injection
            r'<\|.*?\|>',
            r'\[(?:INST|SYS|SYSTEM)\]',
            r'###\s*(?:system|assistant|human|user)\s*:',
            r'human\s*:\s*.*?\s*assistant\s*:',
            
            # Jailbreak attempts
            r'jailbreak',
            r'dan\s+mode',
            r'developer\s+mode',
            
            # Prompt leaking
            r'(?:show|reveal|display|print)\s+(?:your|the)\s+(?:prompt|instructions?|system\s+message)',
            r'what\s+(?:are|is)\s+your\s+(?:instructions?|prompts?|rules)',
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, normalized):
                logger.warning(f"Prompt injection pattern detected: {pattern}")
                return True
        
        # Check for excessive special characters (potential obfuscation)
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        if special_char_ratio > 0.3:
            logger.warning(f"High special character ratio detected: {special_char_ratio}")
            return True
        
        return False
    
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
        
        # Normalize Unicode
        text = InputSanitizer.normalize_unicode(text)
        
        # Decode common encodings
        text = InputSanitizer.decode_common_encodings(text)
        
        # Limit length
        if len(text) > max_length:
            logger.warning(f"Input text truncated from {len(text)} to {max_length} characters")
            text = text[:max_length]
        
        # Remove control characters except common whitespace
        text = ''.join(char for char in text if char.isprintable() or char in '\n\r\t ')
        
        # Check for prompt injection
        if InputSanitizer.detect_prompt_injection(text):
            raise ValueError("Potential prompt injection detected in input")
        
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
        
        return sanitized


# ==================== Content Moderation with AWS Comprehend ====================
class ContentModerator:
    """Content moderation using AWS Comprehend."""
    
    def __init__(self, aws_manager):
        """Initialize content moderator with AWS services."""
        try:
            self.comprehend = boto3.client('comprehend', region_name=Config.AWS_REGION)
            logger.info("Comprehend client initialized for content moderation")
        except Exception as e:
            logger.warning(f"Failed to initialize Comprehend client: {e}")
            self.comprehend = None
    
    async def check_content(self, text: str) -> Dict[str, Any]:
        """
        Check content for toxic or inappropriate material.
        
        Args:
            text: Text to check
            
        Returns:
            Dictionary with moderation results
        """
        if not self.comprehend:
            return {"safe": True, "reason": "Moderation unavailable"}
        
        try:
            # Use Comprehend to detect toxic content
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.comprehend.detect_toxic_content(
                    TextSegments=[{'Text': text[:5000]}],  # Limit to 5000 chars
                    LanguageCode='en'
                )
            )
            
            # Check toxicity scores
            if response.get('ResultList'):
                result = response['ResultList'][0]
                labels = result.get('Labels', [])
                
                for label in labels:
                    if label.get('Score', 0) > 0.7:  # High confidence toxic content
                        logger.warning(f"Toxic content detected: {label.get('Name')}")
                        return {
                            "safe": False,
                            "reason": f"Toxic content detected: {label.get('Name')}",
                            "score": label.get('Score')
                        }
            
            return {"safe": True}
            
        except Exception as e:
            logger.error(f"Content moderation error: {e}")
            # Fail open but log the error
            return {"safe": True, "error": str(e)}


# ==================== AWS Service Manager ====================
class AWSServiceManager:
    """Manages AWS service connections and lifecycle."""
    
    def __init__(self):
        """Initialize AWS service manager with ThreadPoolExecutor."""
        self.executor = ThreadPoolExecutor(max_workers=10)
        logger.info("AWSServiceManager initialized with ThreadPoolExecutor")
    
    def shutdown(self):
        """Shutdown the ThreadPoolExecutor and cleanup resources."""
        logger.info("Shutting down AWSServiceManager")
        self.executor.shutdown(wait=True)
        logger.info("ThreadPoolExecutor shutdown complete")


# Global AWS service manager instance
aws_service_manager = None


# ==================== Application Lifecycle ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    global aws_service_manager
    
    # Startup