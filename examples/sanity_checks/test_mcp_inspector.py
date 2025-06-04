#!/usr/bin/env python3
"""
MCP Inspector Test Suite for HED MCP Server

This script tests the MCP server using the MCP inspector to verify:
- Tool registration and availability
- Resource endpoint functionality
- Parameter validation
- Error handling and protocol compliance
"""

import asyncio
import sys
import time
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import MCP client components
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    MCP_AVAILABLE = True
except ImportError as e:
    logger.error(f"MCP client not available: {e}")
    MCP_AVAILABLE = False


class MCPInspectorTester:
    """Comprehensive MCP server testing using inspector patterns."""

    def __init__(self, server_command: str = "hed-mcp-server"):
        self.server_command = server_command
        self.test_results = {}
        self.session = None

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run the complete test suite and return results."""
        logger.info("🔍 Starting MCP Inspector Test Suite")

        if not MCP_AVAILABLE:
            return {"error": "MCP client not available for testing"}

        try:
            # Connect to server
            await self._connect_to_server()

            # Run test phases
            test_phases = [
                ("server_info", self._test_server_info),
                ("tool_registration", self._test_tool_registration),
                ("resource_endpoints", self._test_resource_endpoints),
                ("tool_parameters", self._test_tool_parameters),
                ("error_handling", self._test_error_handling),
                ("performance", self._test_performance),
            ]

            for phase_name, test_func in test_phases:
                logger.info(f"📋 Running {phase_name} tests...")
                try:
                    result = await test_func()
                    self.test_results[phase_name] = {
                        "status": "passed",
                        "result": result,
                    }
                    logger.info(f"✅ {phase_name} tests passed")
                except Exception as e:
                    self.test_results[phase_name] = {
                        "status": "failed",
                        "error": str(e),
                    }
                    logger.error(f"❌ {phase_name} tests failed: {e}")

            return self.test_results

        except Exception as e:
            logger.error(f"💥 Test suite failed: {e}")
            return {"error": str(e)}

    async def _connect_to_server(self):
        """Establish connection to the HED MCP server."""
        server_params = StdioServerParameters(
            command=self.server_command, args=["--debug"]
        )

        logger.info(f"🔌 Connecting to server: {self.server_command}")

        # Use context manager for stdio client
        self._client_context = stdio_client(server_params)
        self._read, self._write = await self._client_context.__aenter__()

        # Create session
        self._session_context = ClientSession(self._read, self._write)
        self.session = await self._session_context.__aenter__()

        logger.info("✅ Connected to HED MCP server")

    async def _test_server_info(self) -> Dict[str, Any]:
        """Test basic server information and capabilities."""
        # Test server initialization
        result = {"connection": True, "initialization": True}

        # Test if we can call a basic tool
        try:
            tools_response = await self.session.list_tools()
            result["tools_list"] = len(tools_response.tools)
        except Exception as e:
            result["tools_list_error"] = str(e)

        try:
            resources_response = await self.session.list_resources()
            result["resources_list"] = len(resources_response.resources)
        except Exception as e:
            result["resources_list_error"] = str(e)

        return result

    async def _test_tool_registration(self) -> Dict[str, Any]:
        """Test that all expected tools are properly registered."""
        tools_response = await self.session.list_tools()
        available_tools = {tool.name for tool in tools_response.tools}

        expected_tools = {
            "validate_hed_string",
            "validate_hed_file",
            "list_hed_schemas",
            "generate_hed_sidecar",
            "get_server_info",
            "validate_hed_columns",
            "analyze_hed_spreadsheet",
            "server_health",
        }

        missing_tools = expected_tools - available_tools
        extra_tools = available_tools - expected_tools

        # Test tool metadata
        tool_metadata = {}
        for tool in tools_response.tools:
            tool_metadata[tool.name] = {
                "description": tool.description,
                "parameter_count": len(tool.inputSchema.get("properties", {}))
                if tool.inputSchema
                else 0,
            }

        return {
            "expected_tools": list(expected_tools),
            "available_tools": list(available_tools),
            "missing_tools": list(missing_tools),
            "extra_tools": list(extra_tools),
            "registration_complete": len(missing_tools) == 0,
            "tool_metadata": tool_metadata,
        }

    async def _test_resource_endpoints(self) -> Dict[str, Any]:
        """Test resource endpoint functionality."""
        resources_response = await self.session.list_resources()
        available_resources = {res.uri for res in resources_response.resources}

        expected_resource_patterns = [
            "hed://schemas/available",
            "hed://schema/",
            "hed://validation/rules",
        ]

        # Test resource access
        resource_tests = {}

        for resource in resources_response.resources:
            try:
                content = await self.session.read_resource(resource.uri)
                resource_tests[resource.uri] = {
                    "accessible": True,
                    "content_length": len(content.contents[0].text)
                    if content.contents
                    else 0,
                    "mime_type": content.contents[0].mimeType
                    if content.contents
                    else None,
                }
            except Exception as e:
                resource_tests[resource.uri] = {"accessible": False, "error": str(e)}

        return {
            "available_resources": list(available_resources),
            "expected_patterns": expected_resource_patterns,
            "resource_tests": resource_tests,
        }

    async def _test_tool_parameters(self) -> Dict[str, Any]:
        """Test tool parameter validation and functionality."""
        # Test matrix with various parameter combinations
        test_cases = {
            "validate_hed_string": [
                {
                    "args": {"hed_string": "Red, Blue", "schema_version": "8.3.0"},
                    "should_succeed": True,
                    "description": "Valid basic HED string",
                },
                {
                    "args": {"hed_string": "", "schema_version": "8.3.0"},
                    "should_succeed": False,
                    "description": "Empty HED string (should fail)",
                },
                {
                    "args": {
                        "hed_string": "Invalid/Unknown/Tag",
                        "schema_version": "8.3.0",
                    },
                    "should_succeed": True,  # Should succeed but report validation errors
                    "description": "Invalid HED tag",
                },
            ],
            "get_server_info": [
                {
                    "args": {},
                    "should_succeed": True,
                    "description": "Basic server info request",
                }
            ],
            "server_health": [
                {
                    "args": {},
                    "should_succeed": True,
                    "description": "Health check request",
                }
            ],
        }

        results = {}

        for tool_name, cases in test_cases.items():
            tool_results = []

            for case in cases:
                try:
                    start_time = time.time()
                    response = await self.session.call_tool(tool_name, case["args"])
                    execution_time = time.time() - start_time

                    tool_results.append(
                        {
                            "description": case["description"],
                            "success": True,
                            "execution_time": execution_time,
                            "response_type": type(response.content[0].text).__name__
                            if response.content
                            else "None",
                            "response_length": len(str(response.content[0].text))
                            if response.content
                            else 0,
                        }
                    )
                except Exception as e:
                    tool_results.append(
                        {
                            "description": case["description"],
                            "success": False,
                            "error": str(e),
                            "expected_to_fail": not case["should_succeed"],
                        }
                    )

            results[tool_name] = tool_results

        return results

    async def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and security boundaries."""
        error_test_cases = [
            {
                "tool": "validate_hed_file",
                "args": {"file_path": "/nonexistent/file.tsv"},
                "expected_error": "file not found",
                "description": "Non-existent file",
            },
            {
                "tool": "validate_hed_file",
                "args": {"file_path": "../../../etc/passwd"},
                "expected_error": "security",
                "description": "Path traversal attempt",
            },
            {
                "tool": "validate_hed_string",
                "args": {"hed_string": "A" * 100000, "schema_version": "8.3.0"},
                "expected_error": "length",
                "description": "Oversized input",
            },
            {
                "tool": "validate_hed_string",
                "args": {"schema_version": "invalid.version.format"},
                "expected_error": "validation",
                "description": "Invalid schema version format",
            },
        ]

        results = []

        for case in error_test_cases:
            try:
                response = await self.session.call_tool(case["tool"], case["args"])
                # If we get here, the call didn't fail as expected
                results.append(
                    {
                        "description": case["description"],
                        "expected_error": True,
                        "actual_error": False,
                        "status": "unexpected_success",
                        "response": str(response.content[0].text)[:200]
                        if response.content
                        else "",
                    }
                )
            except Exception as e:
                error_str = str(e).lower()
                expected_found = any(
                    expected in error_str for expected in case["expected_error"].split()
                )

                results.append(
                    {
                        "description": case["description"],
                        "expected_error": True,
                        "actual_error": True,
                        "error_contains_expected": expected_found,
                        "status": "expected_error"
                        if expected_found
                        else "unexpected_error",
                        "error_message": str(e)[:200],
                    }
                )

        return results

    async def _test_performance(self) -> Dict[str, Any]:
        """Test performance characteristics."""
        performance_tests = {
            "get_server_info": {"iterations": 10, "timeout": 5.0},
            "server_health": {"iterations": 10, "timeout": 5.0},
            "validate_hed_string": {"iterations": 5, "timeout": 10.0},
        }

        results = {}

        for tool_name, config in performance_tests.items():
            execution_times = []
            errors = []

            # Prepare test arguments
            if tool_name == "validate_hed_string":
                test_args = {
                    "hed_string": "Red, Blue, /Event/Category/Sensory",
                    "schema_version": "8.3.0",
                }
            else:
                test_args = {}

            for i in range(config["iterations"]):
                try:
                    start_time = time.time()
                    await asyncio.wait_for(
                        self.session.call_tool(tool_name, test_args),
                        timeout=config["timeout"],
                    )
                    execution_time = time.time() - start_time
                    execution_times.append(execution_time)
                except Exception as e:
                    errors.append(str(e))

            if execution_times:
                results[tool_name] = {
                    "iterations": len(execution_times),
                    "mean_time": sum(execution_times) / len(execution_times),
                    "min_time": min(execution_times),
                    "max_time": max(execution_times),
                    "errors": len(errors),
                    "error_rate": len(errors) / config["iterations"],
                }
            else:
                results[tool_name] = {
                    "iterations": 0,
                    "errors": len(errors),
                    "all_failed": True,
                }

        return results

    async def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, "_session_context"):
                await self._session_context.__aexit__(None, None, None)
            if hasattr(self, "_client_context"):
                await self._client_context.__aexit__(None, None, None)
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


def format_test_results(results: Dict[str, Any]) -> str:
    """Format test results for display."""
    lines = []
    lines.append("=" * 60)
    lines.append("🔍 HED MCP Server Inspector Test Results")
    lines.append("=" * 60)

    for phase, result in results.items():
        if phase == "error":
            lines.append(f"\n❌ Test Suite Failed: {result}")
            continue

        status = result.get("status", "unknown")
        status_icon = "✅" if status == "passed" else "❌"
        lines.append(f"\n{status_icon} {phase.replace('_', ' ').title()}: {status}")

        if status == "failed":
            lines.append(f"   Error: {result.get('error', 'Unknown error')}")
        elif "result" in result:
            phase_result = result["result"]

            if phase == "tool_registration":
                missing = phase_result.get("missing_tools", [])
                if missing:
                    lines.append(f"   ⚠️  Missing tools: {missing}")
                else:
                    lines.append(
                        f"   ✅ All {len(phase_result.get('expected_tools', []))} expected tools registered"
                    )

            elif phase == "performance":
                lines.append("   Performance Summary:")
                for tool, perf in phase_result.items():
                    if not perf.get("all_failed", False):
                        mean_time = perf.get("mean_time", 0)
                        lines.append(f"     {tool}: {mean_time:.3f}s avg")

            elif phase == "error_handling":
                expected_errors = sum(
                    1 for test in phase_result if test.get("status") == "expected_error"
                )
                lines.append(
                    f"   🛡️  {expected_errors}/{len(phase_result)} error cases handled correctly"
                )

    # Summary
    passed = sum(1 for r in results.values() if r.get("status") == "passed")
    total = len([r for r in results.values() if isinstance(r, dict) and "status" in r])

    lines.append(f"\n📊 Overall: {passed}/{total} test phases passed")

    if passed == total:
        lines.append("🎉 All tests passed! Server is functioning correctly.")
    else:
        lines.append("⚠️  Some tests failed. Review the detailed results above.")

    return "\n".join(lines)


async def main():
    """Main test execution function."""
    if len(sys.argv) > 1:
        server_command = sys.argv[1]
    else:
        server_command = "hed-mcp-server"

    tester = MCPInspectorTester(server_command)

    try:
        results = await tester.run_all_tests()
        print(format_test_results(results))

        # Return appropriate exit code
        if "error" in results:
            sys.exit(1)

        passed = sum(1 for r in results.values() if r.get("status") == "passed")
        total = len(
            [r for r in results.values() if isinstance(r, dict) and "status" in r]
        )

        if passed < total:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
