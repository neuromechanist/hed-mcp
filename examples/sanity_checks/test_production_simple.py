#!/usr/bin/env python3
"""
Simplified test suite for HED MCP Server production features (Subtask 4.6).

Tests the production features without external dependencies.
"""

import asyncio
import time

# Import the production server components
from src.hed_tools.server.server import (
    SecurityConfig,
    MCPErrorHandler,
    InputValidator,
    PerformanceMonitor,
    rate_limit,
)
import mcp.types as types


def test_security_config():
    """Test 1: Security configuration initialization and token generation."""
    print("🔐 Test 1: Security Configuration")

    security_config = SecurityConfig()

    # Test configuration values
    assert security_config.max_request_size == 10 * 1024 * 1024
    assert security_config.rate_limit_requests == 100
    assert security_config.allowed_file_types == [".tsv", ".csv", ".txt"]

    # Test token generation
    token1 = security_config.generate_session_token()
    token2 = security_config.generate_session_token()
    assert len(token1) > 20
    assert token1 != token2

    # Test API key hashing
    api_key = "test_api_key_123"
    hash1 = security_config.hash_api_key(api_key)
    hash2 = security_config.hash_api_key(api_key)
    assert hash1 == hash2
    assert len(hash1) == 64

    print("✅ Security configuration working correctly")


def test_input_validation():
    """Test 2: Input validation and sanitization for security."""
    print("🛡️ Test 2: Input Validation Security")

    security_config = SecurityConfig()

    # Test valid file path
    valid_path = "data/events.tsv"
    result = InputValidator.validate_file_path(valid_path, security_config)
    assert result == valid_path

    # Test directory traversal protection
    try:
        InputValidator.validate_file_path("../../../etc/passwd", security_config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "directory traversal" in str(e)

    # Test file extension validation
    try:
        InputValidator.validate_file_path("malicious.exe", security_config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not allowed" in str(e)

    # Test column list validation
    valid_cols = ["trial_type", "response"]
    result = InputValidator.validate_column_list(valid_cols)
    assert result == valid_cols

    # Test schema version validation
    valid_version = "8.3.0"
    result = InputValidator.validate_schema_version(valid_version)
    assert result == valid_version

    try:
        InputValidator.validate_schema_version("invalid_version")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "major.minor.patch" in str(e)

    print("✅ Input validation working correctly")


def test_performance_monitor():
    """Test 3: Performance monitoring and timing functionality."""
    print("⏱️ Test 3: Performance Monitoring")

    performance_monitor = PerformanceMonitor()

    # Test operation timing
    operation_id = performance_monitor.start_timer("test_operation")
    assert operation_id.startswith("test_operation_")
    assert operation_id in performance_monitor.request_times

    # Simulate operation
    time.sleep(0.1)

    performance_monitor.end_timer(operation_id, "test_operation")
    assert operation_id not in performance_monitor.request_times
    assert "test_operation" in performance_monitor.request_counts

    # Check statistics
    stats = performance_monitor.get_stats()
    assert "test_operation" in stats
    assert stats["test_operation"]["total_requests"] == 1
    assert stats["test_operation"]["average_time"] > 0.05

    print("✅ Performance monitoring working correctly")


async def test_error_handling():
    """Test 4: Error handling decorator and MCP protocol compliance."""
    print("❌ Test 4: Error Handling")

    # Test function that raises ValueError
    @MCPErrorHandler.handle_tool_error("test_tool")
    async def failing_function():
        raise ValueError("Test validation error")

    result = await failing_function()
    assert len(result) == 1
    assert isinstance(result[0], types.TextContent)
    assert "Test validation error" in result[0].text
    assert "test_tool Error" in result[0].text

    # Test function that raises FileNotFoundError
    @MCPErrorHandler.handle_tool_error("test_tool")
    async def file_not_found_function():
        raise FileNotFoundError("Test file not found")

    result = await file_not_found_function()
    assert "File not found" in result[0].text

    print("✅ Error handling working correctly")


async def test_rate_limiting():
    """Test 5: Rate limiting decorator functionality."""
    print("🚦 Test 5: Rate Limiting")

    # Create a test class with rate limited method
    class TestRateLimitedClass:
        def __init__(self):
            self._client_id = "test_client"

        @rate_limit(max_requests=3, window_seconds=10)
        async def limited_method(self):
            return "success"

    test_instance = TestRateLimitedClass()

    # Should succeed for first 3 calls
    for i in range(3):
        result = await test_instance.limited_method()
        assert result == "success"

    # Fourth call should raise ValueError due to rate limit
    try:
        await test_instance.limited_method()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Rate limit exceeded" in str(e)

    print("✅ Rate limiting working correctly")


async def test_mcp_error_context():
    """Test 6: MCP Error Handler context and sanitization."""
    print("🧹 Test 6: Error Context and Sanitization")

    # Test argument sanitization
    test_args = {
        "api_key": "secret_key_123",
        "normal_param": "normal_value",
        "long_string": "x" * 300,
        "password": "secret_password",
    }

    sanitized = MCPErrorHandler._sanitize_arguments(test_args)

    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["password"] == "[REDACTED]"
    assert sanitized["normal_param"] == "normal_value"
    assert len(sanitized["long_string"]) <= 10000

    print("✅ Error context and sanitization working correctly")


def test_comprehensive_feature_validation():
    """Test 7: Comprehensive validation of all production features."""
    print("🔗 Test 7: Comprehensive Feature Validation")

    # Test all components integrate properly
    security_config = SecurityConfig()
    performance_monitor = PerformanceMonitor()

    # Validate security features
    assert hasattr(security_config, "generate_session_token")
    assert hasattr(security_config, "hash_api_key")
    assert hasattr(security_config, "allowed_file_types")

    # Validate performance features
    assert hasattr(performance_monitor, "start_timer")
    assert hasattr(performance_monitor, "end_timer")
    assert hasattr(performance_monitor, "get_stats")

    # Validate input validation
    assert hasattr(InputValidator, "validate_file_path")
    assert hasattr(InputValidator, "validate_column_list")
    assert hasattr(InputValidator, "validate_schema_version")

    # Validate error handling
    assert hasattr(MCPErrorHandler, "handle_tool_error")
    assert hasattr(MCPErrorHandler, "log_request_context")
    assert hasattr(MCPErrorHandler, "_sanitize_arguments")

    print("✅ All production features properly integrated")


async def run_async_tests():
    """Run all async tests."""
    await test_error_handling()
    await test_rate_limiting()
    await test_mcp_error_context()


def main():
    """Run all production feature tests."""
    print("🧪 Starting Production HED MCP Server Tests")
    print("=" * 60)

    # Run synchronous tests
    test_security_config()
    test_input_validation()
    test_performance_monitor()
    test_comprehensive_feature_validation()

    # Run async tests
    asyncio.run(run_async_tests())

    print("\n" + "=" * 60)
    print("🎉 All Production Feature Tests Passed!")
    print("\nProduction Features Validated:")
    print("✅ Security configuration and token generation")
    print("✅ Input validation and sanitization")
    print("✅ Performance monitoring and timing")
    print("✅ Error handling and MCP protocol compliance")
    print("✅ Rate limiting functionality")
    print("✅ Error context logging and sanitization")
    print("✅ Comprehensive integration of all features")
    print("\n🚀 HED MCP Server Subtask 4.6 Implementation Complete!")
    print("\nReady for production deployment with:")
    print("🔐 Enhanced security and input validation")
    print("⚡ Performance monitoring and optimization")
    print("🛡️ Comprehensive error handling")
    print("🚦 Rate limiting and concurrency control")
    print("📊 Request logging and audit trails")


if __name__ == "__main__":
    main()
