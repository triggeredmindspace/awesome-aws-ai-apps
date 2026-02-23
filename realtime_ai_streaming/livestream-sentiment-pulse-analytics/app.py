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
from collections import OrderedDict
from contextlib import asynccontextmanager
import hashlib

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Security, Depends, Request, status
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
    RATE_LIMITER_MAX_CLIENTS = int(os.getenv("RATE_LIMITER_MAX_CLIENTS", "10000"))  # Maximum tracked clients
    MAX_DECODE_DEPTH = int(os.getenv("MAX_DECODE_DEPTH", "2"))  # Maximum recursive decode depth
    MAX_DECODE_TIME_MS = int(os.getenv("MAX_DECODE_TIME_MS", "100"))  # Maximum decode processing time
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else []
    WEBSOCKET_MAX_CONNECTIONS = int(os.getenv("WEBSOCKET_MAX_CONNECTIONS", "1000"))
    
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
    """Rate limiting and anomaly detection for API requests with bounded memory."""
    
    # Injection attack patterns
    SQL_KEYWORDS = [
        'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'EXEC', 'EXECUTE',
        'UNION', 'JOIN', 'WHERE', 'FROM', 'TABLE', '--', ';--', '/*', '*/', 'xp_', 'sp_',
        'INFORMATION_SCHEMA', 'SYSOBJECTS', 'SYSCOLUMNS'
    ]
    
    SCRIPT_PATTERNS = [
        re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'on\w+\s*=', re.IGNORECASE),  # Event handlers like onclick=
        re.compile(r'<iframe[^>]*>', re.IGNORECASE),
        re.compile(r'<object[^>]*>', re.IGNORECASE),
        re.compile(r'<embed[^>]*>', re.IGNORECASE),
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        re.compile(r'[;&|`$]'),  # Shell metacharacters
        re.compile(r'\$\(.*?\)'),  # Command substitution
        re.compile(r'`.*?`'),  # Backtick command execution
        re.compile(r'\|\s*\w+'),  # Pipe to command
        re.compile(r'&&|\|\|'),  # Command chaining
    ]
    
    def __init__(self):
        self.request_counts = OrderedDict()
        self.blocked_ips = OrderedDict()
        self.max_clients = Config.RATE_LIMITER_MAX_CLIENTS
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # Cleanup every 5 minutes
        
    def _cleanup_old_entries(self):
        """Periodically cleanup old entries to prevent memory exhaustion."""
        now = time.time()
        
        # Only cleanup if interval has passed
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        self.last_cleanup = now
        window_start = now - Config.RATE_LIMIT_WINDOW
        
        # Clean old request counts
        clients_to_remove = []
        for client_id, timestamps in self.request_counts.items():
            # Remove old timestamps
            self.request_counts[client_id] = [
                ts for ts in timestamps if ts > window_start
            ]
            # Mark empty entries for removal
            if not self.request_counts[client_id]:
                clients_to_remove.append(client_id)
        
        for client_id in clients_to_remove:
            del self.request_counts[client_id]
        
        # Clean expired blocks
        expired_blocks = [
            client_id for client_id, block_until in self.blocked_ips.items()
            if now >= block_until
        ]
        for client_id in expired_blocks:
            del self.blocked_ips[client_id]
        
        logger.info(f"Rate limiter cleanup: {len(clients_to_remove)} inactive clients, {len(expired_blocks)} expired blocks removed")
    
    def _enforce_size_limit(self):
        """Enforce maximum number of tracked clients using LRU eviction."""
        # Remove oldest entries if we exceed the limit
        while len(self.request_counts) > self.max_clients:
            # Remove oldest (first) entry
            self.request_counts.popitem(last=False)
            logger.warning("Rate limiter at capacity, evicting oldest client")
        
        while len(self.blocked_ips) > self.max_clients:
            self.blocked_ips.popitem(last=False)
    
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
        
        # Periodic cleanup
        self._cleanup_old_entries()
        
        # Check if client is blocked
        if client_id in self.blocked_ips:
            block_until = self.blocked_ips[client_id]
            if now < block_until:
                logger.warning(f"Blocked client attempted request")
                # Move to end (most recently used)
                self.blocked_ips.move_to_end(client_id)
                return False
            else:
                del self.blocked_ips[client_id]
        
        # Initialize or get existing request list
        if client_id not in self.request_counts:
            self.request_counts[client_id] = []
        
        # Move to end (most recently used)
        self.request_counts.move_to_end(client_id)
        
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
            self._enforce_size_limit()
            return False
        
        # Add current request
        self.request_counts[client_id].append(now)
        
        # Enforce size limits
        self._enforce_size_limit()
        
        return True
    
    def _check_sql_injection(self, text: str) -> bool:
        """Check for SQL injection patterns."""
        text_upper = text.upper()
        
        # Check for SQL keywords
        keyword_count = sum(1 for keyword in self.SQL_KEYWORDS if keyword in text_upper)
        if keyword_count >= 2:  # Multiple SQL keywords suggest injection attempt
            logger.warning(f"SQL injection pattern detected: {keyword_count} SQL keywords found")
            return True
        
        # Check for SQL comment patterns
        if '--' in text or '/*' in text or '*/' in text:
            logger.warning("SQL injection pattern detected: SQL comment syntax")
            return True
        
        return False
    
    def _check_xss(self, text: str) -> bool:
        """Check for XSS (Cross-Site Scripting) patterns."""
        for pattern in self.SCRIPT_PATTERNS:
            if pattern.search(text):
                logger.warning("XSS pattern detected: script or event handler")
                return True
        return False