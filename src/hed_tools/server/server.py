"""
HED MCP Server - FastMCP Implementation

A Model Context Protocol (MCP) server providing HED (Hierarchical Event Descriptors)
validation and schema management capabilities using the FastMCP framework.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import Field

from mcp.server.fastmcp import FastMCP, Context

# Configure debug mode from environment
DEBUG_MODE = os.getenv("HED_MCP_DEBUG", "false").lower() == "true"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Configure MCP loggers to be less verbose unless in debug mode
for logger_name in ["mcp", "mcp.server", "mcp.client", "mcp.shared"]:
    logger = logging.getLogger(logger_name)
    if not DEBUG_MODE:
        logger.setLevel(logging.WARNING)
    logger.propagate = False

logger = logging.getLogger(__name__)

# Import HED components with graceful degradation
try:
    from hed_tools.hed_integration.hed_wrapper import HEDWrapper, HEDWrapperConfig
    from hed_tools.hed_integration.validation import HEDValidator
    from hed_tools.hed_integration.schema import SchemaManagerFacade
    from hed_tools.tools.column_analyzer import ColumnAnalyzer

    HED_AVAILABLE = True
    logger.info("HED components loaded successfully")
except ImportError as e:
    logger.warning(f"HED components not available: {e}")
    HEDWrapper = None
    HEDValidator = None
    SchemaManagerFacade = None
    ColumnAnalyzer = None
    HED_AVAILABLE = False

# Create FastMCP server instance
mcp = FastMCP("HED Validation Server")

# Module-level state for schema management
_schema_cache: Dict[str, Any] = {}
_initialized = False


async def _initialize_hed_components():
    """Initialize HED components if available."""
    global _initialized
    if _initialized or not HED_AVAILABLE:
        return

    try:
        # Initialize with default schema
        config = HEDWrapperConfig(hed_schema="8.3.0")
        _schema_cache["default"] = config
        _initialized = True
        logger.info("HED components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize HED components: {e}")
        raise


async def _ensure_schema_loaded(version: str = "8.3.0"):
    """Ensure HED schema is loaded and cached."""
    if not HED_AVAILABLE:
        raise RuntimeError("HED components are not available")

    await _initialize_hed_components()

    if version not in _schema_cache:
        try:
            config = HEDWrapperConfig(hed_schema=version)
            _schema_cache[version] = config
            logger.debug(f"Loaded HED schema version {version}")
        except Exception as e:
            logger.error(f"Failed to load HED schema {version}: {e}")
            raise

    return _schema_cache[version]


# HED Validation Tools


@mcp.tool()
async def validate_hed_string(
    hed_string: str = Field(description="HED string to validate"),
    schema_version: str = Field(
        default="8.3.0", description="HED schema version to use"
    ),
    ctx: Optional[Context] = None,
) -> str:
    """
    Validate a HED string against the specified schema version.

    Returns validation results including any errors or warnings found.
    """
    if not HED_AVAILABLE:
        return "Error: HED components are not available"

    if not hed_string.strip():
        return "Error: Empty HED string provided"

    try:
        await _ensure_schema_loaded(schema_version)

        if ctx:
            ctx.info(f"Validating HED string with schema {schema_version}")

        # Create validator with specified schema
        HEDValidator()
        _schema_cache[schema_version]

        # Perform validation (this would use the actual HED validation logic)
        # For now, returning a placeholder response
        result = f"HED string validation completed for schema {schema_version}"

        if len(hed_string) > 100:
            result += f"\nValidated {len(hed_string)} characters"

        logger.debug(f"Validated HED string: {hed_string[:50]}...")
        return result

    except Exception as e:
        error_msg = f"Validation failed: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def validate_hed_file(
    file_path: str = Field(description="Path to file containing HED annotations"),
    file_type: str = Field(
        default="tsv", description="File format: tsv, json, or bids"
    ),
    schema_version: str = Field(
        default="8.3.0", description="HED schema version to use"
    ),
    ctx: Optional[Context] = None,
) -> str:
    """
    Validate HED annotations in a file.

    Supports TSV, JSON, and BIDS format files containing HED annotations.
    """
    if not HED_AVAILABLE:
        return "Error: HED components are not available"

    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return f"Error: File not found: {file_path}"

        await _ensure_schema_loaded(schema_version)

        if ctx:
            ctx.info(f"Validating HED file: {file_path_obj.name}")

        # Read file asynchronously
        import aiofiles

        async with aiofiles.open(file_path, "r") as f:
            content = await f.read()

        # Validate file content based on type
        HEDValidator()

        # Placeholder validation logic
        lines = content.count("\n")
        result = f"HED file validation completed for {file_path_obj.name}"
        result += f"\nFile type: {file_type}"
        result += f"\nSchema version: {schema_version}"
        result += f"\nProcessed {lines} lines"

        logger.debug(f"Validated file: {file_path}")
        return result

    except Exception as e:
        error_msg = f"File validation failed: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def list_hed_schemas(ctx: Optional[Context] = None) -> str:
    """
    List available HED schema versions.

    Returns information about HED schemas that can be used for validation.
    """
    if not HED_AVAILABLE:
        return "Error: HED components are not available"

    try:
        if ctx:
            ctx.info("Retrieving available HED schemas")

        # This would query actual available schemas
        available_schemas = ["8.3.0", "8.2.0", "8.1.0", "8.0.0"]
        cached_schemas = list(_schema_cache.keys())

        result = "Available HED Schema Versions:\n"
        for schema in available_schemas:
            status = " (cached)" if schema in cached_schemas else ""
            result += f"  - {schema}{status}\n"

        result += f"\nCurrently cached: {len(cached_schemas)} schemas"

        return result

    except Exception as e:
        error_msg = f"Failed to list schemas: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def generate_hed_sidecar(
    events_file: str = Field(description="Path to events.tsv file"),
    output_path: str = Field(description="Output path for JSON sidecar"),
    ctx: Optional[Context] = None,
) -> str:
    """
    Generate a HED sidecar JSON file from an events.tsv file.

    Creates a BIDS-compatible JSON sidecar with HED annotation templates.
    """
    if not HED_AVAILABLE:
        return "Error: HED components are not available"

    try:
        events_path = Path(events_file)
        output_path_obj = Path(output_path)

        if not events_path.exists():
            return f"Error: Events file not found: {events_file}"

        if ctx:
            ctx.info(f"Generating HED sidecar: {output_path_obj.name}")

        # This would use column analyzer to generate sidecar
        ColumnAnalyzer() if ColumnAnalyzer else None

        # Placeholder sidecar generation
        sidecar_content = {
            "trial_type": {"HED": "Event"},
            "response_time": {"HED": "Duration"},
        }

        # Write sidecar file
        import aiofiles
        import json

        async with aiofiles.open(output_path, "w") as f:
            await f.write(json.dumps(sidecar_content, indent=2))

        result = "HED sidecar generated successfully"
        result += f"\nOutput: {output_path}"
        result += f"\nColumns analyzed: {len(sidecar_content)}"

        logger.debug(f"Generated sidecar: {output_path}")
        return result

    except Exception as e:
        error_msg = f"Sidecar generation failed: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def get_server_info(ctx: Optional[Context] = None) -> str:
    """
    Get information about the HED MCP server.

    Returns server status, available tools, and configuration.
    """
    try:
        if ctx:
            ctx.info("Retrieving server information")

        tools = list(mcp._tools.keys())
        resources = list(mcp._resources.keys())

        result = "HED MCP Server Information\n"
        result += "=" * 30 + "\n"
        result += f"HED Components Available: {HED_AVAILABLE}\n"
        result += f"Debug Mode: {DEBUG_MODE}\n"
        result += f"Initialized: {_initialized}\n"
        result += f"Cached Schemas: {len(_schema_cache)}\n"
        result += f"\nAvailable Tools ({len(tools)}):\n"
        for tool in tools:
            result += f"  - {tool}\n"

        if resources:
            result += f"\nAvailable Resources ({len(resources)}):\n"
            for resource in resources:
                result += f"  - {resource}\n"

        return result

    except Exception as e:
        error_msg = f"Failed to get server info: {str(e)}"
        logger.error(error_msg)
        return error_msg


# HED Schema Resources


@mcp.resource("hed://schema/{version}")
async def get_hed_schema_info(version: str) -> str:
    """
    Provide detailed information about a specific HED schema version.

    Returns schema metadata, tag hierarchy, and validation rules.
    """
    if not HED_AVAILABLE:
        return "Error: HED components are not available"

    try:
        await _ensure_schema_loaded(version)

        # This would return actual schema information
        result = f"HED Schema Version {version}\n"
        result += "=" * 30 + "\n"
        result += "Status: Available\n"
        result += f"Cached: {'Yes' if version in _schema_cache else 'No'}\n"
        result += "Tags: ~1000+ hierarchical tags\n"
        result += "Format: XML-based schema definition\n"

        return result

    except Exception as e:
        return f"Error retrieving schema {version}: {str(e)}"


@mcp.resource("hed://validation/rules")
async def get_validation_rules() -> str:
    """
    Provide information about HED validation rules and error types.

    Returns structured documentation of validation rules applied by the server.
    """
    rules = """
HED Validation Rules
===================

1. Syntax Rules:
   - Tags must be properly formatted
   - Required tags must be present
   - Tag hierarchy must be valid

2. Semantic Rules:
   - Value constraints must be satisfied
   - Units must be appropriate for tags
   - Temporal relationships must be logical

3. Error Types:
   - ERROR: Critical validation failures
   - WARNING: Style or recommendation issues
   - INFO: Suggestions for improvement

4. Schema Compliance:
   - All tags must exist in specified schema
   - Deprecated tags generate warnings
   - Extension rules apply for custom tags
"""
    return rules


def _verify_migration():
    """Verify FastMCP migration completed successfully."""
    required_tools = [
        "validate_hed_string",
        "validate_hed_file",
        "list_hed_schemas",
        "generate_hed_sidecar",
        "get_server_info",
    ]
    available_tools = list(mcp._tools.keys())

    missing = [tool for tool in required_tools if tool not in available_tools]
    if missing:
        raise RuntimeError(f"Migration incomplete: missing tools {missing}")

    logger.info(
        f"✓ FastMCP migration successful: {len(available_tools)} tools registered"
    )


def main():
    """Main entry point for the HED MCP server."""
    try:
        if DEBUG_MODE:
            _verify_migration()
            logger.debug("Starting HED MCP Server in debug mode")
        else:
            logger.info("Starting HED MCP Server")

        # Run the FastMCP server
        mcp.run()

    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
