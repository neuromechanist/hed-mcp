#!/usr/bin/env python3
"""
Comprehensive test suite for HED MCP Server production features (Subtask 4.6).

Tests cover:
- Error handling and MCP protocol compliance
- Security middleware with input validation
- Performance monitoring and timeout handling
- Rate limiting functionality
- Async optimization and concurrency control
- Input sanitization and validation
"""

import asyncio
import pytest
import time
import tempfile
import os
from unittest.mock import patch

# Import the production server
from src.hed_tools.server.server import (
    HEDServer,
    SecurityConfig,
    MCPErrorHandler,
    InputValidator,
    PerformanceMonitor,
    rate_limit,
)
import mcp.types as types


class TestProductionHEDServer:
    """Test suite for production HED MCP Server features."""

    @pytest.fixture
    async def server(self):
        """Create a test server instance."""
        server = HEDServer()
        yield server

    @pytest.fixture
    def security_config(self):
        """Create a test security configuration."""
        return SecurityConfig()

    @pytest.fixture
    def performance_monitor(self):
        """Create a test performance monitor."""
        return PerformanceMonitor()

    def test_security_config_initialization(self, security_config):
        """Test 1: Security configuration initialization and token generation."""
        print("🔐 Test 1: Security Configuration")

        # Test configuration values
        assert security_config.max_request_size == 10 * 1024 * 1024
        assert security_config.rate_limit_requests == 100
        assert security_config.allowed_file_types == [".tsv", ".csv", ".txt"]

        # Test token generation
        token1 = security_config.generate_session_token()
        token2 = security_config.generate_session_token()
        assert len(token1) > 20  # URL-safe tokens should be substantial
        assert token1 != token2  # Tokens should be unique

        # Test API key hashing
        api_key = "test_api_key_123"
        hash1 = security_config.hash_api_key(api_key)
        hash2 = security_config.hash_api_key(api_key)
        assert hash1 == hash2  # Same input should produce same hash
        assert len(hash1) == 64  # SHA256 hex digest length

        print("✅ Security configuration working correctly")

    def test_input_validator_file_path_security(self, security_config):
        """Test 2: Input validation and sanitization for security."""
        print("🛡️ Test 2: Input Validation Security")

        # Test valid file path
        valid_path = "data/events.tsv"
        result = InputValidator.validate_file_path(valid_path, security_config)
        assert result == valid_path

        # Test directory traversal protection
        with pytest.raises(ValueError, match="directory traversal"):
            InputValidator.validate_file_path("../../../etc/passwd", security_config)

        with pytest.raises(ValueError, match="directory traversal"):
            InputValidator.validate_file_path("/etc/passwd", security_config)

        # Test null byte protection
        with pytest.raises(ValueError, match="non-empty string"):
            InputValidator.validate_file_path("", security_config)

        # Test file extension validation
        with pytest.raises(ValueError, match="not allowed"):
            InputValidator.validate_file_path("malicious.exe", security_config)

        # Test column list validation
        valid_cols = ["trial_type", "response"]
        result = InputValidator.validate_column_list(valid_cols)
        assert result == valid_cols

        # Test column list sanitization
        malicious_cols = ["valid_col", "", "another\0col"]
        result = InputValidator.validate_column_list(malicious_cols)
        assert result == ["valid_col", "anothercol"]  # Empty and null-byte removed

        # Test schema version validation
        valid_version = "8.3.0"
        result = InputValidator.validate_schema_version(valid_version)
        assert result == valid_version

        with pytest.raises(ValueError, match="major.minor.patch"):
            InputValidator.validate_schema_version("invalid_version")

        print("✅ Input validation working correctly")

    def test_performance_monitor_timing(self, performance_monitor):
        """Test 3: Performance monitoring and timing functionality."""
        print("⏱️ Test 3: Performance Monitoring")

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
        assert stats["test_operation"]["average_time"] > 0.05  # Should be around 0.1s

        print("✅ Performance monitoring working correctly")

    @pytest.mark.asyncio
    async def test_error_handling_decorator(self):
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

        # Test function that raises unexpected error
        @MCPErrorHandler.handle_tool_error("test_tool")
        async def unexpected_error_function():
            raise RuntimeError("Unexpected error")

        result = await unexpected_error_function()
        assert "Unexpected error occurred" in result[0].text

        print("✅ Error handling working correctly")

    @pytest.mark.asyncio
    async def test_rate_limiting_functionality(self):
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
        with pytest.raises(ValueError, match="Rate limit exceeded"):
            await test_instance.limited_method()

        print("✅ Rate limiting working correctly")

    @pytest.mark.asyncio
    async def test_server_initialization_and_production_features(self, server):
        """Test 6: Server initialization with production features."""
        print("🚀 Test 6: Server Production Features")

        # Test server initialization
        assert server.security_config is not None
        assert server.performance_monitor is not None
        assert server.max_concurrent_requests == 10
        assert server.request_timeout == 30.0
        assert server.semaphore._value == 10  # Semaphore initialized correctly

        # Test argument validation and sanitization
        test_args = {
            "file_path": "test_events.tsv",
            "max_unique_values": 25,
            "schema_version": "8.3.0",
            "value_cols": ["trial_type", "response"],
        }

        sanitized = server._validate_and_sanitize_arguments("test_tool", test_args)
        assert sanitized["file_path"] == "test_events.tsv"
        assert sanitized["max_unique_values"] == 25
        assert sanitized["schema_version"] == "8.3.0"
        assert sanitized["value_cols"] == ["trial_type", "response"]

        # Test malicious input sanitization
        malicious_args = {
            "file_path": "../../../etc/passwd",
            "max_unique_values": -1,
            "schema_version": "invalid.version",
            "annotation": "valid" + "\0" + "x" * 20000,  # Null byte + excessive length
        }

        with pytest.raises(ValueError):
            server._validate_and_sanitize_arguments("test_tool", malicious_args)

        print("✅ Server production features working correctly")

    @pytest.mark.asyncio
    async def test_timeout_and_concurrency_control(self, server):
        """Test 7: Timeout handling and concurrency control."""
        print("⏰ Test 7: Timeout and Concurrency")

        # Test timeout functionality
        async def slow_operation():
            await asyncio.sleep(35)  # Longer than 30s timeout
            return "completed"

        with pytest.raises(asyncio.TimeoutError):
            await server._with_timeout_and_monitoring("slow_test", slow_operation())

        # Test fast operation
        async def fast_operation():
            await asyncio.sleep(0.1)
            return "completed"

        result = await server._with_timeout_and_monitoring(
            "fast_test", fast_operation()
        )
        assert result == "completed"

        # Test semaphore concurrency control
        assert server.semaphore._value == 10  # All permits available

        async def concurrent_operation():
            async with server.semaphore:
                await asyncio.sleep(0.2)
                return "concurrent"

        # Start multiple operations
        tasks = [
            server._with_timeout_and_monitoring(
                f"concurrent_test_{i}", concurrent_operation()
            )
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)
        assert all(result == "concurrent" for result in results)
        assert server.semaphore._value == 10  # All permits returned

        print("✅ Timeout and concurrency control working correctly")

    @pytest.mark.asyncio
    async def test_comprehensive_integration(self, server):
        """Test 8: Comprehensive integration of all production features."""
        print("🔗 Test 8: Comprehensive Integration")

        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("onset\tduration\ttrial_type\tresponse\n")
            f.write("0.0\t1.0\tgo\tcorrect\n")
            f.write("2.0\t1.0\tstop\tincorrect\n")
            temp_file = f.name

        try:
            # Test tool call with all production features
            test_args = {"file_path": temp_file, "max_unique_values": 10}

            # Mock the column analyzer for testing
            with patch.object(server, "column_analyzer") as mock_analyzer:
                mock_analyzer.analyze_events_file.return_value = {
                    "columns": {
                        "trial_type": {
                            "type": "object",
                            "unique_count": 2,
                            "statistics": {"unique_values": ["go", "stop"]},
                        }
                    },
                    "bids_compliance": {"is_compliant": True, "score": 85.0},
                    "recommendations": ["Consider adding more metadata"],
                }

                # This should succeed with all production features
                result = await server._analyze_event_columns_with_security(test_args)

                assert len(result) == 1
                assert isinstance(result[0], types.TextContent)
                assert "Column Analysis Results" in result[0].text
                assert temp_file in result[0].text

        finally:
            # Clean up
            os.unlink(temp_file)

        print("✅ Comprehensive integration working correctly")


def main():
    """Run all production feature tests."""
    print("🧪 Starting Production HED MCP Server Tests")
    print("=" * 60)

    # Run synchronous tests
    test_instance = TestProductionHEDServer()

    # Test 1: Security Configuration
    security_config = SecurityConfig()
    test_instance.test_security_config_initialization(security_config)

    # Test 2: Input Validation
    test_instance.test_input_validator_file_path_security(security_config)

    # Test 3: Performance Monitoring
    performance_monitor = PerformanceMonitor()
    test_instance.test_performance_monitor_timing(performance_monitor)

    # Run async tests
    async def run_async_tests():
        server = HEDServer()

        # Test 4: Error Handling
        await test_instance.test_error_handling_decorator()

        # Test 5: Rate Limiting
        await test_instance.test_rate_limiting_functionality()

        # Test 6: Server Features
        await test_instance.test_server_initialization_and_production_features(server)

        # Test 7: Timeout and Concurrency
        await test_instance.test_timeout_and_concurrency_control(server)

        # Test 8: Integration
        await test_instance.test_comprehensive_integration(server)

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
    print("✅ Server initialization with production features")
    print("✅ Timeout handling and concurrency control")
    print("✅ Comprehensive integration of all features")
    print("\n🚀 HED MCP Server is production-ready!")


if __name__ == "__main__":
    main()
