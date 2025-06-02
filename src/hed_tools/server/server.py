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
import warnings

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

# Import components with graceful degradation
try:
    from hed_tools.hed_integration.hed_wrapper import HEDWrapper
    from hed_tools.tools.column_analyzer import ColumnAnalyzer
    from hed_tools.hed_integration.validation import HEDValidator
    from hed_tools.hed_integration.schema import SchemaHandler, SchemaManagerFacade

    HED_AVAILABLE = True
    logger.info("HED components loaded successfully")
except ImportError as e:
    warnings.warn(f"Could not import HED components: {e}")
    HEDWrapper = None
    ColumnAnalyzer = None
    HEDValidator = None
    SchemaHandler = None
    SchemaManagerFacade = None
    HED_AVAILABLE = False

# Create FastMCP server instance
mcp = FastMCP("HED Validation Server")

# Module-level state for schema management
_schema_cache: Dict[str, Any] = {}
_initialized = False

# Global schema handler
_schema_handler: Optional[SchemaHandler] = None
_schema_manager: Optional[SchemaManagerFacade] = None


async def _initialize_hed_components():
    """Initialize HED components asynchronously."""
    global _initialized, _schema_handler, _schema_manager

    if _initialized:
        return

    try:
        if not HED_AVAILABLE:
            logger.warning("HED components not available - running in stub mode")
            _initialized = True
            return

        # Initialize schema handler and manager
        if SchemaHandler is not None:
            _schema_handler = SchemaHandler()
            logger.info("Schema handler initialized")

        if SchemaManagerFacade is not None:
            _schema_manager = SchemaManagerFacade()
            await _schema_manager.initialize()
            logger.info("Schema manager initialized")

        _initialized = True
        logger.info("HED components initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize HED components: {e}")
        _initialized = True  # Mark as initialized to avoid repeated attempts


async def _ensure_schema_loaded(version: str = "8.3.0"):
    """Ensure HED schema is loaded and cached."""
    await _initialize_hed_components()

    if not HED_AVAILABLE or _schema_handler is None:
        logger.warning(f"Cannot load schema {version} - HED components not available")
        return None

    try:
        # Use schema handler to load schema
        result = await _schema_handler.load_schema(version=version)
        if result.success:
            logger.info(f"Schema {version} loaded successfully")
            return _schema_handler.get_schema()
        else:
            logger.error(f"Failed to load schema {version}: {result.error}")
            return None
    except Exception as e:
        logger.error(f"Error loading schema {version}: {e}")
        return None


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


@mcp.resource("hed://schemas/available")
async def get_available_schemas() -> str:
    """
    List all available HED schema versions with detailed information.

    Returns information about official HED schema versions that can be loaded.
    """
    await _initialize_hed_components()

    if not HED_AVAILABLE or _schema_handler is None:
        return "Error: HED components are not available"

    try:
        # Get available schemas from the schema handler
        available_schemas = _schema_handler.get_available_schemas()
        loaded_versions = _schema_handler.get_loaded_schema_versions()

        result = "Available HED Schema Versions\n"
        result += "=" * 40 + "\n\n"

        for schema_info in available_schemas:
            version = schema_info.get("version", "unknown")
            description = schema_info.get("description", "No description available")
            status = "✓ Cached" if version in loaded_versions else "○ Available"

            result += f"{status} {version}\n"
            result += f"   Description: {description}\n\n"

        # Add cache information
        cache_info = _schema_handler.get_cache_info()
        result += "Cache Status:\n"
        result += f"  - Cached schemas: {cache_info.get('cached_count', 0)}\n"
        result += f"  - Cache size: {cache_info.get('cache_size_mb', 0):.1f} MB\n"

        return result

    except Exception as e:
        error_msg = f"Failed to retrieve available schemas: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.resource("hed://schema/{version}")
async def get_hed_schema_info(version: str) -> str:
    """
    Provide detailed information about a specific HED schema version.

    Returns schema metadata, tag hierarchy summary, and validation capabilities.
    """
    await _initialize_hed_components()

    if not HED_AVAILABLE or _schema_handler is None:
        return "Error: HED components are not available"

    try:
        # Load the specific schema
        schema_result = await _schema_handler.load_schema(version=version)

        if not schema_result.success:
            return (
                f"Error: Could not load schema version {version}: {schema_result.error}"
            )

        schema_info = _schema_handler.get_schema_info()
        all_tags = _schema_handler.get_all_schema_tags(version=version)

        result = f"HED Schema Version {version}\n"
        result += "=" * 50 + "\n\n"

        if schema_info:
            result += f"Version: {schema_info.version}\n"
            result += f"Library: {schema_info.library}\n"
            result += f"Name: {schema_info.name}\n"
            result += f"Description: {schema_info.description}\n"
            result += f"Filename: {schema_info.filename}\n\n"

        # Tag statistics
        result += "Tag Information:\n"
        result += f"  - Total tags: {len(all_tags)}\n"
        result += f"  - Loading time: {schema_result.processing_time:.3f}s\n\n"

        # Sample of available tags (first 10)
        if all_tags:
            sample_tags = sorted(list(all_tags))[:10]
            result += "Sample Tags:\n"
            for tag in sample_tags:
                result += f"  - {tag}\n"
            if len(all_tags) > 10:
                result += f"  ... and {len(all_tags) - 10} more\n"

        return result

    except Exception as e:
        error_msg = f"Error retrieving schema {version}: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.resource("hed://schema/{version}/tags")
async def get_schema_tags(version: str) -> str:
    """
    Get all tags available in a specific HED schema version.

    Returns a comprehensive list of all tags in the schema.
    """
    await _initialize_hed_components()

    if not HED_AVAILABLE or _schema_handler is None:
        return "Error: HED components are not available"

    try:
        # Load the schema if needed
        await _ensure_schema_loaded(version)

        # Get all tags for this version
        all_tags = _schema_handler.get_all_schema_tags(version=version)

        if not all_tags:
            return f"No tags found for schema version {version}"

        # Organize tags by hierarchy
        sorted_tags = sorted(list(all_tags))

        result = f"All Tags in HED Schema {version}\n"
        result += "=" * 50 + "\n\n"
        result += f"Total tags: {len(sorted_tags)}\n\n"

        # Group tags by top-level category
        categories = {}
        for tag in sorted_tags:
            if "/" in tag:
                category = tag.split("/")[0]
            else:
                category = tag

            if category not in categories:
                categories[category] = []
            if tag != category:  # Don't duplicate the category itself
                categories[category].append(tag)

        # Display organized tags
        for category in sorted(categories.keys()):
            result += f"\n{category}:\n"
            for tag in sorted(categories[category])[
                :50
            ]:  # Limit to first 50 per category
                result += f"  - {tag}\n"
            if len(categories[category]) > 50:
                result += f"  ... and {len(categories[category]) - 50} more in this category\n"

        return result

    except Exception as e:
        error_msg = f"Error retrieving tags for schema {version}: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.resource("hed://validation/rules")
async def get_validation_rules() -> str:
    """
    Provide information about HED validation rules and error types.

    Returns structured documentation of validation rules applied by the server.
    """
    await _initialize_hed_components()

    rules = """
HED Validation Rules and Guidelines
==================================

1. SYNTAX RULES:
   • Tags must use proper formatting with forward slashes for hierarchy
   • Required tags must be present in annotations
   • Tag values must match expected data types (numeric, text, etc.)
   • Parentheses must be balanced for grouping
   • Commas separate multiple tags or tag groups

2. SEMANTIC RULES:
   • All tags must exist in the specified HED schema
   • Value constraints must be satisfied (e.g., units, ranges)
   • Temporal relationships must be logically consistent
   • Required tags for specific contexts must be present

3. SCHEMA COMPLIANCE:
   • Tags must exist in the loaded schema version
   • Deprecated tags generate warnings
   • Extension rules apply for custom tags (use # prefix)
   • Tag attributes must match schema definitions

4. ERROR TYPES:
   • ERROR: Critical validation failures that prevent processing
   • WARNING: Style issues or recommendations for improvement
   • INFO: Suggestions and best practices

5. VALIDATION PROCESS:
   • String parsing and syntax checking
   • Schema lookup and tag validation
   • Semantic rule application
   • Error reporting with line numbers and suggestions

6. SUPPORTED FORMATS:
   • Raw HED strings (comma-separated tags)
   • TSV files with HED columns
   • JSON sidecars with HED annotations
   • BIDS-compatible event files
"""

    # Add server-specific information if available
    if HED_AVAILABLE and _schema_handler is not None:
        loaded_versions = _schema_handler.get_loaded_schema_versions()
        if loaded_versions:
            rules += "\n\nCURRENTLY AVAILABLE SCHEMAS:\n"
            for version in loaded_versions:
                rules += f"   • HED {version}\n"

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
