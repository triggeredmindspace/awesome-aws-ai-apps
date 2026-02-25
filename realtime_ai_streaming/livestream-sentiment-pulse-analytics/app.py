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
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError, BotoCoreError
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Security, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
import uvicorn
import bleach


# ==================== AWS Configuration and Validation ====================
class AWSClientManager:
    """Manages AWS client configuration with proper credential validation, retries, and error handling."""
    
    # Boto3 client configuration with retries and timeouts
    BOTO3_CONFIG = Config(
        retries={
            'max_attempts': 5,
            'mode': 'adaptive'  # Uses exponential backoff
        },
        connect_timeout=5,
        read_timeout=60,
        max_pool_connections=50
    )
    
    _clients = {}
    _credentials_validated = False
    
    @classmethod
    def validate_credentials(cls) -> bool:
        """
        Validate AWS credentials on startup.
        
        Returns:
            bool: True if credentials are valid
            
        Raises:
            NoCredentialsError: If no credentials are found
            PartialCredentialsError: If credentials are incomplete
            ClientError: If credentials are invalid
        """
        if cls._credentials_validated:
            return True
        
        try:
            # Use STS to validate credentials
            sts_client = boto3.client('sts', config=cls.BOTO3_CONFIG)
            response = sts_client.get_caller_identity()
            
            logger.info(f"AWS credentials validated successfully. Account: {response.get('Account', 'Unknown')}")
            cls._credentials_validated = True
            return True
            
        except NoCredentialsError as e:
            logger.error("AWS credentials not found. Please configure credentials.")
            raise
        except PartialCredentialsError as e:
            logger.error("AWS credentials are incomplete.")
            raise
        except ClientError as e:
            logger.error(f"AWS credentials validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during AWS credential validation: {e}")
            raise
    
    @classmethod
    def get_client(cls, service_name: str, region_name: Optional[str] = None):
        """
        Get or create a boto3 client with proper configuration.
        
        Args:
            service_name: AWS service name (e.g., 's3', 'dynamodb')
            region_name: AWS region name (optional)
            
        Returns:
            Configured boto3 client
        """
        # Validate credentials first
        if not cls._credentials_validated:
            cls.validate_credentials()
        
        # Create cache key
        cache_key = f"{service_name}:{region_name or 'default'}"
        
        # Return cached client if available
        if cache_key in cls._clients:
            return cls._clients[cache_key]
        
        # Create new client with configuration
        try:
            client_kwargs = {'config': cls.BOTO3_CONFIG}
            if region_name:
                client_kwargs['region_name'] = region_name
            
            client = boto3.client(service_name, **client_kwargs)
            cls._clients[cache_key] = client
            
            logger.info(f"Created AWS {service_name} client for region {region_name or 'default'}")
            return client
            
        except Exception as e:
            logger.error(f"Failed to create AWS {service_name} client: {e}")
            raise
    
    @classmethod
    async def execute_with_backoff(cls, func, *args, max_retries: int = 5, **kwargs):
        """
        Execute AWS API call with exponential backoff.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            max_retries: Maximum number of retry attempts
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            Exception: If all retries are exhausted
        """
        base_delay = 1  # Start with 1 second
        max_delay = 60  # Cap at 60 seconds
        
        for attempt in range(max_retries):
            try:
                # Execute in thread pool for sync boto3 calls
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
                return result
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                
                # Don't retry on certain errors
                non_retryable_errors = [
                    'AccessDenied',
                    'InvalidAccessKeyId',
                    'SignatureDoesNotMatch',
                    'InvalidClientTokenId'
                ]
                
                if error_code in non_retryable_errors:
                    logger.error(f"Non-retryable AWS error: {error_code}")
                    raise
                
                # Retry on throttling and server errors
                if attempt < max_retries - 1:
                    # Calculate exponential backoff with jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = delay * 0.1 * (0.5 - asyncio.get_event_loop().time() % 1)
                    total_delay = delay + jitter
                    
                    logger.warning(f"AWS API call failed (attempt {attempt + 1}/{max_retries}): {error_code}. Retrying in {total_delay:.2f}s")
                    await asyncio.sleep(total_delay)
                else:
                    logger.error(f"AWS API call failed after {max_retries} attempts: {error_code}")
                    raise
                    
            except BotoCoreError as e:
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"BotoCore error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"BotoCore error after {max_retries} attempts: {e}")
                    raise
                    
            except Exception as e:
                logger.error(f"Unexpected error during AWS API call: {e}")
                raise


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
            required_keys: List of required