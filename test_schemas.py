#!/usr/bin/env python3
"""Test script to verify hed_schemas resource implementation."""

import sys
import asyncio
import json

sys.path.insert(0, "src")

from hed_tools.server.server import HEDServer


async def test_hed_schemas_resource():
    """Test the comprehensive hed_schemas resource."""
    print("🧪 Testing comprehensive hed_schemas resource...")

    server = HEDServer()

    try:
        # Test schema info retrieval
        schema_info_str = await server._get_schema_info()
        schema_info = json.loads(schema_info_str)

        print("✅ Schema info retrieved successfully")

        # Test metadata structure
        assert "meta" in schema_info, "Missing meta section"
        assert "schemas" in schema_info, "Missing schemas section"
        assert "recommendations" in schema_info, "Missing recommendations section"
        assert "version_comparison" in schema_info, "Missing version comparison"
        assert "validation" in schema_info, "Missing validation info"
        assert "caching" in schema_info, "Missing caching info"

        print("✅ All required sections present")

        # Test schema catalog
        schemas = schema_info["schemas"]
        expected_versions = ["8.3.0", "8.2.0", "8.1.0", "8.0.0"]

        for version in expected_versions:
            assert version in schemas, f"Missing schema version {version}"
            schema = schemas[version]

            # Check required metadata fields
            required_fields = [
                "version",
                "publication_date",
                "status",
                "features",
                "schema_url",
            ]
            for field in required_fields:
                assert field in schema, f"Missing {field} in schema {version}"

        print(f"✅ All {len(expected_versions)} schemas have complete metadata")

        # Test version comparison functionality
        version_comp = schema_info["version_comparison"]
        assert version_comp["newest_first"][0] == "8.3.0", "Latest version not first"
        assert (
            "8.3.0" in version_comp["by_stability"]["stable"]
        ), "8.3.0 not marked as stable"

        print("✅ Version comparison functionality working")

        # Test recommendations
        recommendations = schema_info["recommendations"]
        assert recommendations["latest_stable"] == "8.3.0", "Incorrect latest stable"
        assert "upgrade_path" in recommendations, "Missing upgrade paths"

        print("✅ Recommendations system working")

        # Test validation info
        validation = schema_info["validation"]
        assert validation["all_schemas_validated"], "Schemas not validated"
        assert len(validation["validation_criteria"]) > 0, "No validation criteria"

        print("✅ Validation system working")

        # Show sample output
        print("\n📋 Sample Schema Info:")
        print(f"  • Total schemas: {schema_info['meta']['total_schemas']}")
        print(f"  • Current schema: {schema_info['meta']['current_schema']}")
        print(f"  • Latest stable: {schema_info['recommendations']['latest_stable']}")
        print(f"  • Manager status: {schema_info['meta']['manager_status']}")

        # Show schema features for latest version
        latest_schema = schemas["8.3.0"]
        print("\n🏷️  HED 8.3.0 Features:")
        for feature in latest_schema["features"][:3]:
            print(f"    • {feature}")

        return True

    except Exception as e:
        print(f"❌ Schema resource test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_resource_access():
    """Test accessing the resource through MCP protocol simulation."""
    print("\n🧪 Testing MCP resource access...")

    server = HEDServer()

    try:
        # Simulate MCP resource access
        resource_content = await server._get_schema_info()
        schema_data = json.loads(resource_content)

        # Verify we can access specific schema information
        schema_8_3 = schema_data["schemas"]["8.3.0"]
        print("✅ Successfully accessed HED 8.3.0 schema info")
        print(f"   Published: {schema_8_3['publication_date']}")
        print(f"   Status: {schema_8_3['status']}")
        print(f"   Documentation: {schema_8_3['documentation']}")

        return True

    except Exception as e:
        print(f"❌ MCP resource access test failed: {e}")
        return False


async def main():
    """Run comprehensive schema resource tests."""
    print("🚀 Testing HED Schemas Resource - Subtask 4.3")
    print("=" * 60)

    tests_passed = 0
    total_tests = 2

    if await test_hed_schemas_resource():
        tests_passed += 1

    if await test_resource_access():
        tests_passed += 1

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 HED schemas resource fully implements subtask 4.3 requirements!")
        print("\n✅ Implemented features:")
        print("   • Comprehensive schema catalog with metadata")
        print("   • Publication dates and feature documentation")
        print("   • Version comparison and compatibility matrix")
        print("   • Validation status and caching information")
        print("   • Upgrade recommendations and paths")
    else:
        print("❌ Some features need additional work.")


if __name__ == "__main__":
    asyncio.run(main())
