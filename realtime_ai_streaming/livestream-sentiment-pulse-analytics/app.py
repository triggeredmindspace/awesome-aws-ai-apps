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
import bleach


# ==================== Input Validation Framework ====================
class InputValidator:
    """Centralized input validation and sanitization framework."""
    
    # Allowed HTML tags and attributes for sanitization
    ALLOWED_TAGS = ['p', 'br', 'strong', 'em', 'u', 'a', 'ul', 'ol', 'li']
    ALLOWED_ATTRIBUTES = {'a': ['href', 'title']}
    ALLOWED_PROTOCOLS = ['http', 'https']
    
    # Validation patterns
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    ALPHANUMERIC_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
    UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    
    # Dangerous patterns
    SQL_INJECTION_PATTERNS = [
        re.compile(r"(\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b|\bCREATE\b|\bALTER\b|\bEXEC\b|\bUNION\b)", re.IGNORECASE),
        re.compile(r"(--|;--|/\*|\*/|xp_|sp_)", re.IGNORECASE),
        re.compile(r"(\bOR\b\s+\d+\s*=\s*\d+|\bAND\b\s+\d+\s*=\s*\d+)", re.IGNORECASE),
    ]
    
    XSS_PATTERNS = [
        re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'on\w+\s*=', re.IGNORECASE),
        re.compile(r'<iframe[^>]*>', re.IGNORECASE),
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        re.compile(r'[;&|`$]'),
        re.compile(r'\$\(.*?\)'),
        re.compile(r'`.*?`'),
    ]
    
    @classmethod
    def sanitize_html(cls, text: str) -> str:
        """Sanitize HTML content using bleach library."""
        if not text:
            return ""
        
        return bleach.clean(
            text,
            tags=cls.ALLOWED_TAGS,
            attributes=cls.ALLOWED_ATTRIBUTES,
            protocols=cls.ALLOWED_PROTOCOLS,
            strip=True
        )
    
    @classmethod
    def validate_string(cls, value: str, field_name: str, max_length: int = None, 
                       min_length: int = 0, pattern: re.Pattern = None,
                       allow_empty: bool = False) -> str:
        """
        Validate and sanitize string input.
        
        Args:
            value: Input string to validate
            field_name: Name of the field for error messages
            max_length: Maximum allowed length
            min_length: Minimum required length
            pattern: Regex pattern to match
            allow_empty: Whether empty strings are allowed
            
        Returns:
            Validated and sanitized string
            
        Raises:
            ValueError: If validation fails
        """
        if value is None:
            if allow_empty:
                return ""
            raise ValueError(f"{field_name} cannot be None")
        
        # Convert to string and normalize
        value = str(value).strip()
        
        # Check empty
        if not value and not allow_empty:
            raise ValueError(f"{field_name} cannot be empty")
        
        # Check length
        if min_length and len(value) < min_length:
            raise ValueError(f"{field_name} must be at least {min_length} characters")
        
        if max_length and len(value) > max_length:
            raise ValueError(f"{field_name} exceeds maximum length of {max_length}")
        
        # Check pattern
        if pattern and not pattern.match(value):
            raise ValueError(f"{field_name} format is invalid")
        
        # Check for injection attacks
        cls.check_injection_attacks(value, field_name)
        
        return value
    
    @classmethod
    def check_injection_attacks(cls, value: str, field_name: str):
        """
        Check for common injection attack patterns.
        
        Args:
            value: Input to check
            field_name: Field name for error messages
            
        Raises:
            ValueError: If injection pattern detected
        """
        # SQL Injection
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if pattern.search(value):
                raise ValueError(f"{field_name} contains potentially malicious SQL patterns")
        
        # XSS
        for pattern in cls.XSS_PATTERNS:
            if pattern.search(value):
                raise ValueError(f"{field_name} contains potentially malicious script patterns")
        
        # Command Injection
        for pattern in cls.COMMAND_INJECTION_PATTERNS:
            if pattern.search(value):
                raise ValueError(f"{field_name} contains potentially malicious command patterns")
    
    @classmethod
    def validate_integer(cls, value: Any, field_name: str, min_value: int = None, 
                        max_value: int = None) -> int:
        """
        Validate integer input.
        
        Args:
            value: Input to validate
            field_name: Field name for error messages
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            Validated integer
            
        Raises:
            ValueError: If validation fails
        """
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field_name} must be a valid integer")
        
        if min_value is not None and int_value < min_value:
            raise ValueError(f"{field_name} must be at least {min_value}")
        
        if max_value is not None and int_value > max_value:
            raise ValueError(f"{field_name} must be at most {max_value}")
        
        return int_value
    
    @classmethod
    def validate_float(cls, value: Any, field_name: str, min_value: float = None, 
                      max_value: float = None) -> float:
        """
        Validate float input.
        
        Args:
            value: Input to validate
            field_name: Field name for error messages
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            Validated float
            
        Raises:
            ValueError: If validation fails
        """
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field_name} must be a valid number")
        
        if min_value is not None and float_value < min_value:
            raise ValueError(f"{field_name} must be at least {min_value}")
        
        if max_value is not None and float_value > max_value:
            raise ValueError(f"{field_name} must be at most {max_value}")
        
        return float_value
    
    @classmethod
    def validate_email(cls, email: str) -> str:
        """
        Validate email address format.
        
        Args:
            email: Email address to validate
            
        Returns:
            Validated email
            
        Raises:
            ValueError: If email format is invalid
        """
        email = cls.validate_string(email, "email", max_length=254)
        
        if not cls.EMAIL_PATTERN.match(email):
            raise ValueError("Invalid email format")
        
        return email.lower()
    
    @classmethod
    def validate_uuid(cls, uuid_str: str, field_name: str = "UUID") -> str:
        """
        Validate UUID format.
        
        Args:
            uuid_str: UUID string to validate
            field_name: Field name for error messages
            
        Returns:
            Validated UUID string
            
        Raises:
            ValueError: If UUID format is invalid
        """
        uuid_str = cls.validate_string(uuid_str, field_name, max_length=36)
        
        if not cls.UUID_PATTERN.match(uuid_str):
            raise ValueError(f"{field_name} must be a valid UUID")
        
        return uuid_str.lower()
    
    @classmethod
    def validate_dict(cls, data: Any, field_name: str, required_keys: List[str] = None,
                     max_keys: int = 100) -> Dict:
        """
        Validate dictionary input.
        
        Args:
            data: Input to validate
            field_name: Field name for error messages
            required_keys: List of required keys
            max_keys: Maximum number of keys allowed
            
        Returns:
            Validated dictionary
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(data, dict):
            raise ValueError(f"{field_name} must be a dictionary")
        
        if len(data) > max_keys:
            raise ValueError(f"{field_name} exceeds maximum of {max_keys} keys")
        
        if required_keys:
            missing_keys = set(required_keys) - set(data.keys())
            if missing_keys:
                raise ValueError(f"{field_name} missing required keys: {missing_keys}")
        
        return data
    
    @classmethod
    def validate_list(cls, data: Any, field_name: str, max_items: int = 1000,
                     item_validator: callable = None) -> List:
        """
        Validate list input.
        
        Args:
            data: Input to validate
            field_name: Field name for error messages
            max_items: Maximum number of items allowed
            item_validator: Optional function to validate each item
            
        Returns:
            Validated list
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(data, list):
            raise ValueError(f"{field_name} must be a list")
        
        if len(data) > max_items:
            raise ValueError(f"{field_name} exceeds maximum of {max_items} items")
        
        if item_validator:
            validated_items = []
            for i, item in enumerate(data):
                try:
                    validated_items.append(item_validator(item))
                except ValueError as e:
                    raise ValueError(f"{field_name}[{i}]: {str(e)}")
            return validated_items
        
        return data


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
    DYNAMODB_HIGHLIGHTS_TABLE = os.getenv("DYNAMODB_