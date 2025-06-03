"""
Security utilities for HED MCP Server.

This module provides comprehensive security measures including:
- Rate limiting and resource management
- Path validation and sanitization
- MCP protocol-compliant error handling
- Security audit logging
- Input/output validation and scrubbing
"""

import os
import re
import time
import json
import hashlib
import logging
import asyncio
import psutil
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Dict, Any, List, Optional

# Replace the magic import with optional import
try:
    import magic

    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    magic = None


class MCPErrorCode(Enum):
    """MCP protocol error codes with security-focused extensions."""

    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    # Custom security error codes
    RATE_LIMITED = -32000
    SECURITY_VIOLATION = -32001
    TIMEOUT_ERROR = -32002
    FILE_ACCESS_DENIED = -32003
    RESOURCE_EXHAUSTED = -32004


class SecurityError(Exception):
    """Base exception for security-related errors."""

    def __init__(
        self, message: str, error_code: MCPErrorCode = MCPErrorCode.SECURITY_VIOLATION
    ):
        super().__init__(message)
        self.error_code = error_code


class RateLimitError(SecurityError):
    """Exception raised when rate limits are exceeded."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, MCPErrorCode.RATE_LIMITED)


class TimeoutError(SecurityError):
    """Exception raised when operations timeout."""

    def __init__(self, message: str = "Operation timeout"):
        super().__init__(message, MCPErrorCode.TIMEOUT_ERROR)


class SecureConfig:
    """Environment-based configuration with secure defaults."""

    def __init__(self):
        self.config = {
            "max_file_size": int(
                os.getenv("HED_MAX_FILE_SIZE", "104857600")
            ),  # 100MB default
            "rate_limit_window": int(os.getenv("HED_RATE_LIMIT_WINDOW", "60")),
            "max_concurrent_requests": int(os.getenv("HED_MAX_CONCURRENT", "10")),
            "allowed_base_paths": self._parse_allowed_paths(),
            "security_log_level": os.getenv("HED_SECURITY_LOG_LEVEL", "INFO"),
            "enable_audit_logging": os.getenv("HED_ENABLE_AUDIT", "true").lower()
            == "true",
            "max_memory_mb": int(os.getenv("HED_MAX_MEMORY_MB", "512")),
            "max_execution_time": int(
                os.getenv("HED_MAX_EXECUTION_TIME", "300")
            ),  # 5 minutes
            "temp_dir": os.getenv("HED_TEMP_DIR", "/tmp"),
        }

    def _parse_allowed_paths(self) -> List[str]:
        """Parse allowed base paths from environment."""
        paths_str = os.getenv("HED_ALLOWED_PATHS", "/tmp,/data,./data")
        paths = [p.strip() for p in paths_str.split(",") if p.strip()]

        # Resolve relative paths and ensure they exist or can be created
        resolved_paths = []
        for path in paths:
            try:
                resolved_path = Path(path).resolve()
                resolved_paths.append(str(resolved_path))
            except Exception:
                # Skip invalid paths
                continue

        return resolved_paths if resolved_paths else ["/tmp"]

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)


# Global configuration instance
config = SecureConfig()


class RateLimiter:
    """Per-tool rate limiting with configurable windows and limits."""

    def __init__(self):
        self.requests = defaultdict(list)
        self.limits = {
            "generate_hed_sidecar": {"calls": 5, "window": 60},
            "validate_hed_string": {"calls": 50, "window": 60},
            "validate_hed_file": {"calls": 10, "window": 60},
            "search_hed_schema": {"calls": 100, "window": 60},
            "list_hed_schemas": {"calls": 200, "window": 60},
            "server_health": {"calls": 20, "window": 60},
        }
        self.global_limit = {"calls": 100, "window": 60}

    def check_limit(self, tool_name: str, client_id: str = "default") -> bool:
        """Check if request is within rate limits."""
        now = time.time()

        # Check tool-specific limits
        tool_limits = self.limits.get(tool_name, self.global_limit)
        window = tool_limits["window"]
        limit = tool_limits["calls"]

        key = f"{tool_name}:{client_id}"

        # Clean old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key] if now - req_time < window
        ]

        # Check if limit exceeded
        if len(self.requests[key]) >= limit:
            return False

        # Record this request
        self.requests[key].append(now)
        return True

    def get_remaining_requests(self, tool_name: str, client_id: str = "default") -> int:
        """Get number of remaining requests in current window."""
        tool_limits = self.limits.get(tool_name, self.global_limit)
        key = f"{tool_name}:{client_id}"

        # Clean old requests first
        now = time.time()
        window = tool_limits["window"]
        self.requests[key] = [
            req_time for req_time in self.requests[key] if now - req_time < window
        ]

        return max(0, tool_limits["calls"] - len(self.requests[key]))


# Global rate limiter instance
rate_limiter = RateLimiter()


class SecurityAuditor:
    """Security audit logging with sanitization."""

    def __init__(self, log_file: str = "security_audit.log"):
        self.logger = logging.getLogger("security_audit")

        # Only add handler if not already present
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)
            self.logger.setLevel(
                getattr(logging, config.get("security_log_level", "INFO"))
            )

    def log_security_event(
        self, event_type: str, details: Dict[str, Any], severity: str = "INFO"
    ):
        """Log security events with sanitization."""
        if not config.get("enable_audit_logging", True):
            return

        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "severity": severity,
            "details": self._sanitize_audit_data(details),
        }

        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method(json.dumps(audit_entry))

    def _sanitize_audit_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive fields from audit logs."""
        sensitive_keys = ["password", "token", "key", "secret", "auth"]
        sanitized = {}

        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            else:
                # Truncate long values and convert to string
                sanitized[key] = (
                    str(value)[:200] if len(str(value)) > 200 else str(value)
                )

        return sanitized


# Global security auditor instance
security_auditor = SecurityAuditor()


class SecurityAwareErrorHandler:
    """MCP protocol-compliant error handler with security focus."""

    def __init__(self):
        self.logger = logging.getLogger("error_handler")

    def format_mcp_error(
        self, code: MCPErrorCode, message: str, data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Format errors according to MCP protocol with security sanitization."""
        # Sanitize error messages to prevent information leakage
        sanitized_message = self._sanitize_error_message(message)

        error_response = {
            "error": {
                "code": code.value,
                "message": sanitized_message,
            }
        }

        # Only include data if it doesn't contain sensitive information
        if data and not self._contains_sensitive_data(data):
            error_response["error"]["data"] = self._sanitize_error_data(data)

        # Log security-related errors
        if code in [
            MCPErrorCode.SECURITY_VIOLATION,
            MCPErrorCode.RATE_LIMITED,
            MCPErrorCode.FILE_ACCESS_DENIED,
        ]:
            security_auditor.log_security_event(
                f"mcp_error_{code.name.lower()}",
                {"message": message, "code": code.value},
                "WARNING",
            )

        return error_response

    def _sanitize_error_message(self, message: str) -> str:
        """Remove sensitive information from error messages."""
        # Remove file paths
        message = re.sub(r"/[^\s]*", "[PATH_REDACTED]", message)
        # Remove IP addresses
        message = re.sub(
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP_REDACTED]", message
        )
        # Remove potential secret patterns
        message = re.sub(r"[a-zA-Z0-9]{32,}", "[TOKEN_REDACTED]", message)
        return message

    def _contains_sensitive_data(self, data: Dict[str, Any]) -> bool:
        """Check if data contains sensitive information."""
        sensitive_patterns = [
            "password",
            "token",
            "key",
            "secret",
            "auth",
            "credential",
        ]

        for key in data.keys():
            if any(pattern in key.lower() for pattern in sensitive_patterns):
                return True
        return False

    def _sanitize_error_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize error data while preserving useful information."""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 100:
                sanitized[key] = value[:100] + "..."
            else:
                sanitized[key] = value
        return sanitized


# Global error handler instance
error_handler = SecurityAwareErrorHandler()


class ResourceManager:
    """Manage execution resources with timeouts and memory limits."""

    def __init__(
        self,
        max_memory_mb: Optional[int] = None,
        max_execution_time: Optional[int] = None,
    ):
        self.max_memory_mb = max_memory_mb or config.get("max_memory_mb", 512)
        self.max_execution_time = max_execution_time or config.get(
            "max_execution_time", 300
        )
        self.logger = logging.getLogger("resource_manager")

    @asynccontextmanager
    async def managed_execution(self, operation_name: str):
        """Context manager for resource-controlled execution."""
        start_memory = self._get_memory_usage()
        start_time = time.time()

        try:
            async with asyncio.timeout(self.max_execution_time):
                yield
        except asyncio.TimeoutError:
            security_auditor.log_security_event(
                "operation_timeout",
                {
                    "operation": operation_name,
                    "timeout_seconds": self.max_execution_time,
                    "execution_time": time.time() - start_time,
                },
                "WARNING",
            )
            raise TimeoutError(
                f"Operation {operation_name} exceeded {self.max_execution_time}s timeout"
            )
        finally:
            # Check memory usage
            end_memory = self._get_memory_usage()
            memory_used = end_memory - start_memory

            if memory_used > self.max_memory_mb:
                security_auditor.log_security_event(
                    "high_memory_usage",
                    {
                        "operation": operation_name,
                        "memory_used_mb": memory_used,
                        "limit_mb": self.max_memory_mb,
                    },
                    "WARNING",
                )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0


class PathValidator:
    """Secure path validation with allowlist checking."""

    def __init__(self, allowed_base_paths: Optional[List[str]] = None):
        self.allowed_base_paths = allowed_base_paths or config.get(
            "allowed_base_paths", ["/tmp"]
        )
        self.logger = logging.getLogger("path_validator")

    def validate_path(self, path: str, operation: str = "access") -> Path:
        """Validate and resolve path with security checks."""
        if not path:
            raise SecurityError("Empty path provided")

        # Basic security checks
        if ".." in path:
            security_auditor.log_security_event(
                "path_traversal_attempt",
                {"path": path, "operation": operation},
                "WARNING",
            )
            raise SecurityError("Path traversal attempt detected")

        # Resolve path
        try:
            resolved_path = Path(path).resolve()
        except Exception as e:
            raise SecurityError(f"Invalid path: {e}")

        # Check against allowlist
        path_str = str(resolved_path)
        allowed = False

        for base_path in self.allowed_base_paths:
            try:
                base_resolved = str(Path(base_path).resolve())
                if path_str.startswith(base_resolved):
                    allowed = True
                    break
            except Exception:
                continue

        if not allowed:
            security_auditor.log_security_event(
                "path_access_denied",
                {
                    "path": path_str,
                    "operation": operation,
                    "allowed_paths": self.allowed_base_paths,
                },
                "WARNING",
            )
            raise SecurityError(f"Path access denied: {path}")

        return resolved_path

    def validate_file_type(self, file_path: Path, expected_types: List[str]) -> bool:
        """Validate file type using magic numbers, not just extensions."""
        try:
            # Check if file exists
            if not file_path.exists():
                return False

            # Get MIME type using magic
            mime_type = magic.from_file(str(file_path), mime=True)
            return mime_type in expected_types
        except Exception:
            return False


# Global path validator instance
path_validator = PathValidator()


# File type mappings
ALLOWED_MIME_TYPES = {
    "tsv": ["text/tab-separated-values", "text/plain", "text/csv"],
    "csv": ["text/csv", "text/plain", "application/csv"],
    "json": ["application/json", "text/plain"],
    "txt": ["text/plain", "text/csv"],
}


def with_rate_limiting(tool_name: str):
    """Decorator for rate limiting tool calls."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract client ID from context if available
            client_id = "default"
            for arg in args:
                if hasattr(arg, "session") and hasattr(arg.session, "client_id"):
                    client_id = arg.session.client_id
                    break

            # Check rate limit
            if not rate_limiter.check_limit(tool_name, client_id):
                remaining = rate_limiter.get_remaining_requests(tool_name, client_id)
                security_auditor.log_security_event(
                    "rate_limit_exceeded",
                    {"tool": tool_name, "client_id": client_id, "remaining": remaining},
                    "WARNING",
                )
                raise RateLimitError(f"Rate limit exceeded for {tool_name}")

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def with_resource_management(operation_name: str):
    """Decorator for resource-managed execution."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            resource_manager = ResourceManager()
            async with resource_manager.managed_execution(operation_name):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def with_security_validation(allowed_base_paths: Optional[List[str]] = None):
    """Decorator for comprehensive security validation."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            validator = PathValidator(allowed_base_paths)

            # Validate path arguments
            for key, value in kwargs.items():
                if "path" in key.lower() and isinstance(value, str):
                    try:
                        kwargs[key] = str(validator.validate_path(value, func.__name__))
                    except SecurityError:
                        # Re-raise with tool context
                        raise SecurityError(
                            f"Security validation failed for {key} in {func.__name__}"
                        )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def create_secure_temp_file(suffix: str = ".tmp", prefix: str = "hed_") -> Path:
    """Create a secure temporary file in the configured temp directory."""
    import tempfile

    temp_dir = config.get("temp_dir", "/tmp")

    # Ensure temp directory exists and is writable
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)

    if not os.access(temp_path, os.W_OK):
        raise SecurityError(f"Temp directory not writable: {temp_dir}")

    # Create secure temporary file
    fd, temp_file = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=temp_dir)
    os.close(fd)

    return Path(temp_file)


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """Sanitize text input to prevent injection attacks."""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    # Length check
    if len(text) > max_length:
        raise ValueError(f"Input too long: {len(text)} > {max_length}")

    # Remove potential script injections and dangerous patterns
    dangerous_patterns = [
        r"<script[^>]*>.*?</script>",  # Script tags
        r"javascript:",  # JavaScript URLs
        r"data:",  # Data URLs
        r"vbscript:",  # VBScript URLs
        r"on\w+\s*=",  # Event handlers
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            security_auditor.log_security_event(
                "dangerous_input_detected",
                {"pattern": pattern, "text_length": len(text)},
                "WARNING",
            )
            raise SecurityError("Potentially dangerous content detected in input")

    return text


def hash_sensitive_data(data: str) -> str:
    """Create a hash of sensitive data for logging purposes."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]
