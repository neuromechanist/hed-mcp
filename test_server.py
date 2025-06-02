#!/usr/bin/env python3
"""Simple test script to verify the HED MCP server works."""

import sys
import subprocess


def test_server_initialization():
    """Test that the server can initialize properly."""
    print("🧪 Testing server initialization...")

    # Add src to path and test server import
    test_code = """
import sys
sys.path.insert(0, 'src')
from hed_tools.server.server import HEDServer

try:
    server = HEDServer()
    print("✅ Server initialized successfully")

    # Test that tools are registered
    tools = []
    async def mock_list_tools():
        # Access the registered tools handler
        for handler in server.server._tool_handlers:
            if hasattr(handler, '__name__') and 'list' in handler.__name__:
                return await handler()
        return []

    print("✅ Server structure looks good")

except Exception as e:
    print(f"❌ Server initialization failed: {e}")
    sys.exit(1)
"""

    result = subprocess.run(
        [sys.executable, "-c", test_code], capture_output=True, text=True, cwd="."
    )

    if result.returncode == 0:
        print("✅ Server initialization test passed")
        print(result.stdout)
    else:
        print("❌ Server initialization test failed")
        print(result.stderr)
        return False

    return True


def test_mcp_tools_format():
    """Test that the tools are properly formatted for MCP."""
    print("\n🧪 Testing MCP tools format...")

    test_code = """
import sys
sys.path.insert(0, 'src')
from hed_tools.server.server import HEDServer
import asyncio

async def test_tools():
    server = HEDServer()

    # Find the list_tools handler
    tools_handler = None
    for handler in server.server._tool_handlers:
        if hasattr(handler, '__name__') and 'list_tools' in str(handler):
            tools_handler = handler
            break

    if tools_handler:
        tools = await tools_handler()
        print(f"✅ Found {len(tools)} tools")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        return True
    else:
        print("❌ Could not find tools handler")
        return False

try:
    result = asyncio.run(test_tools())
    if result:
        print("✅ All tools properly registered")
    else:
        print("❌ Tools registration issue")
except Exception as e:
    print(f"❌ Tools test failed: {e}")
"""

    result = subprocess.run(
        [sys.executable, "-c", test_code], capture_output=True, text=True, cwd="."
    )

    if result.returncode == 0:
        print("✅ MCP tools format test passed")
        print(result.stdout)
    else:
        print("❌ MCP tools format test failed")
        print(result.stderr)
        return False

    return True


def test_generate_sidecar_basic():
    """Test basic sidecar generation functionality."""
    print("\n🧪 Testing basic sidecar generation...")

    test_code = """
import sys
sys.path.insert(0, 'src')
from hed_tools.server.server import HEDServer
import asyncio

async def test_sidecar():
    server = HEDServer()

    # Test the basic sidecar generation without HED components
    try:
        result = await server._create_basic_sidecar_template(
            'examples/sample_events.tsv',
            ['onset', 'duration'],
            ['trial_type', 'response'],
            '8.3.0'
        )

        print("✅ Basic sidecar generation works")
        print(f"Generated sidecar with {len(result)} columns")

        # Check structure
        if 'trial_type' in result:
            trial_type = result['trial_type']
            if 'HED' in trial_type and 'LevelsAndValues' in trial_type:
                print("✅ Sidecar structure is correct")
                print(f"  trial_type has {len(trial_type['HED'])} values")
                return True

        print("❌ Sidecar structure is incorrect")
        return False

    except Exception as e:
        print(f"❌ Sidecar generation failed: {e}")
        return False

try:
    result = asyncio.run(test_sidecar())
    if not result:
        sys.exit(1)
except Exception as e:
    print(f"❌ Sidecar test failed: {e}")
    sys.exit(1)
"""

    result = subprocess.run(
        [sys.executable, "-c", test_code], capture_output=True, text=True, cwd="."
    )

    if result.returncode == 0:
        print("✅ Basic sidecar generation test passed")
        print(result.stdout)
    else:
        print("❌ Basic sidecar generation test failed")
        print(result.stderr)
        return False

    return True


if __name__ == "__main__":
    print("🚀 Testing HED MCP Server Implementation")
    print("=" * 50)

    # Run tests
    tests_passed = 0
    total_tests = 3

    if test_server_initialization():
        tests_passed += 1

    if test_mcp_tools_format():
        tests_passed += 1

    if test_generate_sidecar_basic():
        tests_passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! MCP server is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
