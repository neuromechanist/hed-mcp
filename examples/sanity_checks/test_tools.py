#!/usr/bin/env python3
"""Test script to verify MCP tools work end-to-end."""

import sys
import asyncio

sys.path.insert(0, "src")

from hed_tools.server.server import HEDServer


async def test_analyze_columns():
    """Test the analyze_event_columns tool."""
    print("🧪 Testing analyze_event_columns tool...")

    server = HEDServer()

    try:
        # Test with our sample events file
        result = await server._analyze_event_columns(
            {"file_path": "examples/sample_events.tsv", "max_unique_values": 10}
        )

        print("✅ analyze_event_columns completed")
        print("Response:")
        print(result[0].text)
        return True

    except Exception as e:
        print(f"❌ analyze_event_columns failed: {e}")
        return False


async def test_generate_sidecar():
    """Test the generate_hed_sidecar tool."""
    print("\n🧪 Testing generate_hed_sidecar tool...")

    server = HEDServer()

    try:
        # Test with our sample events file
        result = await server._generate_hed_sidecar(
            {
                "file_path": "examples/sample_events.tsv",
                "skip_cols": ["onset", "duration"],
                "value_cols": ["trial_type", "response", "stimulus"],
                "schema_version": "8.3.0",
            }
        )

        print("✅ generate_hed_sidecar completed")
        print("Response preview:")
        response_text = result[0].text
        print(
            response_text[:500] + "..." if len(response_text) > 500 else response_text
        )
        return True

    except Exception as e:
        print(f"❌ generate_hed_sidecar failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Testing MCP Tools End-to-End")
    print("=" * 50)

    tests_passed = 0
    total_tests = 2

    if await test_analyze_columns():
        tests_passed += 1

    if await test_generate_sidecar():
        tests_passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All MCP tools are working!")
    else:
        print("❌ Some tools need fixes.")


if __name__ == "__main__":
    asyncio.run(main())
