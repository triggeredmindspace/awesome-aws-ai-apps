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


# ==================== Sensitive Data Filter ====================
class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive data from logs using pattern matching."""
    
    # Regex patterns to detect sensitive data formats
    PATTERNS = {
        'aws_access_key': re.compile(r'AKIA[0-9A-Z]{16}'),
        'aws_secret_key': re.compile(r'(?i)aws_secret[_\s]*(?:access[_\s]*)?key[_\s]*[=:]\s*["\']?([A-Za-z0-9/+=]{40})["\']?'),
        'api_key': re.compile(r'(?i)api[_\s]*key[_\s]*[=:]\s*["\']?([A-Za-z0-9_\-]{20,})["\']?'),
        'bearer_token': re.compile(r'Bearer\s+[A-Za-z0-9\-._~+/]+=*'),
        'password': re.compile(r'(?i)password[_\s]*[=:]\s*["\']?([^\s"\']{8,})["\']?'),
        'secret': re.compile(r'(?i)secret[_\s]*[=:]\s*["\']?([A-Za-z0-9/+=]{16,})["\']?'),
        'token': re.compile(r'(?i)token[_\s]*[=:]\s*["\']?([A-Za-z0-9_\-\.]{20,})["\']?'),
        'jwt': re.compile(r'eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*'),
        'private_key': re.compile(r'-----BEGIN (?:RSA |EC )?PRIVATE KEY-----'),
        'session_token': re.compile(r'(?i)session[_\s]*token[_\s]*[=:]\s*["\']?([A-Za-z0-9/+=]{20,})["\']?'),
    }
    
    def filter(self, record):
        """Redact sensitive information from log records using pattern matching."""
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            
            # Apply each pattern to redact sensitive data
            for pattern_name, pattern in self.PATTERNS.items():
                msg = pattern.sub('[REDACTED]', msg)
            
            record.msg = msg
        
        # Also check args if present
        if hasattr(record, 'args') and record.args:
            try:
                sanitized_args = []
                for arg in record.args:
                    arg_str = str(arg)
                    for pattern_name, pattern in self.PATTERNS.items():
                        arg_str = pattern.sub('[REDACTED]', arg_str)
                    sanitized_args.append(arg_str)
                record.args = tuple(sanitized_args)
            except Exception:
                # If sanitization fails, keep original to avoid breaking logging
                pass
        
        return True


# Configure logging with sensitive data filter
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.addFilter(SensitiveDataFilter())

# Create a separate debug logger for internal use
debug_logger = logging.getLogger(__name__ + '.debug')
debug_logger.setLevel(logging.DEBUG if os.getenv('DEBUG_MODE', 'false').lower() == 'true' else logging.CRITICAL)


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
    # Retrieve from AWS Secrets Manager if available, otherwise fall back to environment variable
    _api_key = None
    
    @classmethod
    def get_api_key(cls) -> str:
        """
        Retrieve API key from AWS Secrets Manager or environment variable.
        Never logs the actual key value.
        """
        if cls._api_key:
            return cls._api_key
            
        # Try AWS Secrets Manager first
        try:
            secrets_client = boto3.client('secretsmanager', region_name=cls.AWS_REGION)
            response = secrets_client.get_secret_value(SecretId='sentiment-analytics/api-key')
            secret_data = json.loads(response['SecretString'])
            cls._api_key = secret_data.get('API_KEY')
            if cls._api_key:
                logger.info("API key retrieved from AWS Secrets Manager")
                return cls._api_key
        except Exception as e:
            logger.info("AWS Secrets Manager not available, falling back to environment variable")
        
        # Fall back to environment variable
        cls._api_key = os.environ.get("API_KEY")
        if not cls._api_key:
            logger.error("API_KEY not found in Secrets Manager or environment variables")
            raise ValueError("API_KEY must be set in AWS Secrets Manager or environment variable")
        
        logger.info("API key retrieved from environment variable")
        return cls._api_key
    
    @property
    def API_KEY(self) -> str:
        """Property to access API key safely."""
        return self.get_api_key()


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
            client_id: Client identifier (IP address or API key hash)
            
        Returns:
            True if request is allowed, False if rate limited
        """
        now = time.time()
        window_start = now - Config.RATE_LIMIT_WINDOW
        
        # Check if client is blocked
        if client_id in self.blocked_ips:
            block_until = self.blocked_ips[client_id]
            if now < block_until:
                logger.warning(f"Blocked client attempted request")
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
            logger.warning(f"Rate limit exceeded for client")
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
                logger.warning(f"Anomaly detected: Large input from client")
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
    def is_valid_base64(text: str) -> bool:
        """
        Check if text is valid base64 with reasonable structure.
        
        Args:
            text: Text to check
            
        Returns:
            True if valid base64, False otherwise
        """
        # Remove whitespace
        text = text.strip().replace('\n', '').replace('\r', '').replace(' ', '')
        
        # Length limits - must be between 16 and 10000 characters
        if len(text) < 16 or len(text) > 10000:
            return False
        
        # Must be multiple of 4 (with padding)
        if len(text) % 4 != 0:
            return False
        
        # Check character set - only valid base64 characters
        if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', text):
            return False
        
        # Padding must be at the end only
        if '=' in text and not text.endswith('==') and not text.endswith('='):
            return False
        
        # Check for reasonable entropy (not all same character)
        unique_chars = len(set(text.replace('=', '')))
        if unique_chars < 4:
            return False
        
        return True
    
    @staticmethod
    def is_valid_decoded_content(decoded: str) -> bool:
        """
        Validate decoded content is reasonable text.
        
        Args:
            decoded: Decoded text
            
        Returns:
            True if content is valid, False otherwise
        """
        # Must not be empty
        if not decoded or len(decoded) == 0:
            return False
        
        # Must not be too long
        if len(decoded) > Config.MAX_INPUT_LENGTH:
            return False
        
        # Check for reasonable printable character ratio
        printable_ratio = sum(1 for c in decoded if c.isprintable() or c.isspace()) / len(decoded)
        if printable_ratio < 0.8:
            return False
        
        # Must not contain excessive null bytes or control characters
        control_chars = sum(1 for c in decoded if ord(c) < 32 and c not in '\n\r\t')
        if control_chars > len(decoded) * 0.1:
            return False
        
        return True
    
    @staticmethod
    def decode_common_encodings(text: str) -> str:
        """
        Detect and decode common encoding attempts with validation.
        
        Args:
            text: Input text
            
        Returns:
            Decoded text
        """
        original_text = text
        
        # Try to detect base64 encoding with strict validation
        try:
            # Only attempt decode if it looks like valid base64
            if InputSanitizer.is_valid_base64(text):
                decoded_bytes = base64.b64decode(text, validate=True)
                
                # Limit decoded size
                if len(decoded_bytes) > Config.MAX_INPUT_LENGTH:
                    logger.warning("Base64 decoded content exceeds size limit")
                    return original_text
                
                # Try to decode as UTF-8
                decoded = decoded_bytes.decode('utf-8', errors='strict')
                
                # Validate decoded content
                if InputSanitizer.is_valid_decoded_content(decoded):
                    logger.warning("Valid base64 encoded input detected and decoded")
                    text = decoded
                else:
                    logger.warning("Base64 decoded content failed validation")
                    return original_text
        except (base64.binascii.Error, UnicodeDecodeError, ValueError) as e:
            # Not valid base64 or not valid UTF-8, keep original
            pass
        except Exception as e:
            logger.warning(f"Unexpected error during base64 decode: {type(e).__name__}")
            return original_text
        
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
                logger.warning("HTML