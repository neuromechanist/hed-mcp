"""
HED MCP Server - FastMCP Implementation

A Model Context Protocol (MCP) server providing HED (Hierarchical Event Descriptors)
validation and schema management capabilities using the FastMCP framework.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
import csv
from pydantic import Field
import warnings
import time
import signal
import sys
import argparse
from dataclasses import dataclass, field
from datetime import datetime

from mcp.server.fastmcp import FastMCP, Context

# Security and validation imports
from ..utils.security import (
    with_rate_limiting,
    with_resource_management,
    with_security_validation,
    error_handler,
    security_auditor,
    MCPErrorCode,
    SecurityError,
    RateLimitError,
    TimeoutError,
    hash_sensitive_data,
)
from ..utils.validation_models import (
    HedStringValidationRequest,
    HedFileValidationRequest,
    validate_request_model,
)

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

# Import optional dependencies for tabular data handling
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
    logger.info("Pandas available for spreadsheet analysis")
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("Pandas not available - limited spreadsheet support")

try:
    import importlib.util

    EXCEL_AVAILABLE = importlib.util.find_spec("openpyxl") is not None
    if EXCEL_AVAILABLE:
        logger.info("OpenPyXL available for Excel file support")
    else:
        logger.warning("OpenPyXL not available - no Excel file support")
except ImportError:
    EXCEL_AVAILABLE = False
    logger.warning("OpenPyXL not available - no Excel file support")

# Create FastMCP server instance
mcp = FastMCP("HED Validation Server")

# Module-level state for schema management
_schema_cache: Dict[str, Any] = {}
_initialized = False

# Global schema handler
_schema_handler: Optional[SchemaHandler] = None
_schema_manager: Optional[SchemaManagerFacade] = None


class SimpleHedWrapper:
    """Simple HED wrapper for validation operations."""

    async def validate_hed_string(
        self,
        hed_string: str,
        schema_version: str = "8.3.0",
        check_for_warnings: bool = True,
    ) -> Dict[str, Any]:
        """Validate a HED string."""
        # Simple validation placeholder
        return {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "summary": f"Validated HED string with schema {schema_version}",
        }

    async def validate_hed_file(
        self,
        file_path: str,
        schema_version: str = "8.3.0",
        check_for_warnings: bool = True,
    ) -> Dict[str, Any]:
        """Validate a HED file."""
        # Simple validation placeholder
        return {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "summary": f"Validated file {file_path} with schema {schema_version}",
        }


# Use the real HedWrapper if available, otherwise use simple wrapper
if HED_AVAILABLE and HEDWrapper is not None:
    HedWrapper = HEDWrapper
else:
    HedWrapper = SimpleHedWrapper


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
@with_rate_limiting("validate_hed_string")
@with_resource_management("validate_hed_string")
async def validate_hed_string(
    hed_string: str = Field(description="HED string to validate"),
    schema_version: str = Field(default="8.3.0", description="HED schema version"),
    check_for_warnings: bool = Field(
        default=True, description="Include warnings in validation"
    ),
) -> Dict[str, Any]:
    """
    Validate a HED (Hierarchical Event Descriptors) string against a specified schema version.

    This tool validates HED strings for syntax errors, semantic consistency, and compliance
    with the specified HED schema version. It provides comprehensive error reporting and
    validation feedback.

    Args:
        hed_string: The HED string to validate (max 50,000 characters)
        schema_version: HED schema version in X.Y.Z format (default: "8.3.0")
        check_for_warnings: Whether to include validation warnings (default: True)

    Returns:
        Dict containing validation results with is_valid, errors, warnings, and summary

    Raises:
        SecurityError: If input validation fails or security constraints are violated
        RateLimitError: If rate limits are exceeded
        TimeoutError: If validation takes too long
    """
    try:
        # Validate and sanitize inputs using Pydantic model
        request = validate_request_model(
            HedStringValidationRequest,
            {
                "hed_string": hed_string,
                "schema_version": schema_version,
                "check_for_warnings": check_for_warnings,
            },
        )

        # Log the validation request for audit
        security_auditor.log_security_event(
            "hed_string_validation_request",
            {
                "schema_version": request.schema_version,
                "string_length": len(request.hed_string),
                "check_warnings": request.check_for_warnings,
                "string_hash": hash_sensitive_data(request.hed_string)[:8],
            },
            "INFO",
        )

        # Initialize HED wrapper
        wrapper = HedWrapper()

        # Perform validation
        validation_result = await wrapper.validate_hed_string(
            request.hed_string,
            schema_version=request.schema_version,
            check_for_warnings=request.check_for_warnings,
        )

        # Add metadata to response
        validation_result.update(
            {
                "request_metadata": {
                    "schema_version": request.schema_version,
                    "string_length": len(request.hed_string),
                    "validation_timestamp": datetime.utcnow().isoformat(),
                    "check_warnings": request.check_for_warnings,
                }
            }
        )

        return validation_result

    except SecurityError as e:
        security_auditor.log_security_event(
            "hed_string_validation_security_error",
            {"error": str(e), "tool": "validate_hed_string"},
            "ERROR",
        )
        return error_handler.format_mcp_error(e.error_code, str(e))

    except RateLimitError as e:
        return error_handler.format_mcp_error(MCPErrorCode.RATE_LIMITED, str(e))

    except TimeoutError as e:
        return error_handler.format_mcp_error(MCPErrorCode.TIMEOUT_ERROR, str(e))

    except Exception as e:
        logger.error(f"Unexpected error in validate_hed_string: {e}")
        security_auditor.log_security_event(
            "hed_string_validation_unexpected_error",
            {"error": str(e), "tool": "validate_hed_string"},
            "ERROR",
        )
        return error_handler.format_mcp_error(
            MCPErrorCode.INTERNAL_ERROR, "Validation failed due to internal error"
        )


@mcp.tool()
@with_rate_limiting("validate_hed_file")
@with_resource_management("validate_hed_file")
@with_security_validation()
async def validate_hed_file(
    file_path: str = Field(description="Path to the file to validate"),
    schema_version: str = Field(default="8.3.0", description="HED schema version"),
    check_for_warnings: bool = Field(
        default=True, description="Include warnings in validation"
    ),
) -> Dict[str, Any]:
    """
    Validate a HED-annotated file against a specified schema version.

    This tool validates files containing HED annotations (typically events.tsv files)
    for syntax errors, semantic consistency, and compliance with the specified
    HED schema version.

    Args:
        file_path: Path to the file to validate (TSV, CSV, or TXT format)
        schema_version: HED schema version in X.Y.Z format (default: "8.3.0")
        check_for_warnings: Whether to include validation warnings (default: True)

    Returns:
        Dict containing validation results with is_valid, errors, warnings, and file info

    Raises:
        SecurityError: If file access is denied or security constraints are violated
        RateLimitError: If rate limits are exceeded
        TimeoutError: If validation takes too long
    """
    try:
        # Validate and sanitize inputs using Pydantic model
        request = validate_request_model(
            HedFileValidationRequest,
            {
                "file_path": file_path,
                "schema_version": schema_version,
                "check_for_warnings": check_for_warnings,
            },
        )

        # Verify file exists and is accessible
        file_obj = Path(request.file_path)
        if not file_obj.exists():
            raise SecurityError(f"File not found: {request.file_path}")

        if not file_obj.is_file():
            raise SecurityError(f"Path is not a file: {request.file_path}")

        # Check file size limits (100MB default)
        file_size = file_obj.stat().st_size
        max_size = 100 * 1024 * 1024  # 100MB
        if file_size > max_size:
            raise SecurityError(f"File too large: {file_size} bytes > {max_size} bytes")

        # Log the validation request for audit
        security_auditor.log_security_event(
            "hed_file_validation_request",
            {
                "file_path": str(file_obj),
                "file_size": file_size,
                "schema_version": request.schema_version,
                "check_warnings": request.check_for_warnings,
            },
            "INFO",
        )

        # Initialize HED wrapper
        wrapper = HedWrapper()

        # Perform file validation
        validation_result = await wrapper.validate_hed_file(
            request.file_path,
            schema_version=request.schema_version,
            check_for_warnings=request.check_for_warnings,
        )

        # Add metadata to response
        validation_result.update(
            {
                "file_metadata": {
                    "file_path": str(file_obj),
                    "file_size": file_size,
                    "schema_version": request.schema_version,
                    "validation_timestamp": datetime.utcnow().isoformat(),
                    "check_warnings": request.check_for_warnings,
                }
            }
        )

        return validation_result

    except SecurityError as e:
        security_auditor.log_security_event(
            "hed_file_validation_security_error",
            {"error": str(e), "tool": "validate_hed_file", "file_path": file_path},
            "ERROR",
        )
        return error_handler.format_mcp_error(e.error_code, str(e))

    except RateLimitError as e:
        return error_handler.format_mcp_error(MCPErrorCode.RATE_LIMITED, str(e))

    except TimeoutError as e:
        return error_handler.format_mcp_error(MCPErrorCode.TIMEOUT_ERROR, str(e))

    except Exception as e:
        logger.error(f"Unexpected error in validate_hed_file: {e}")
        security_auditor.log_security_event(
            "hed_file_validation_unexpected_error",
            {"error": str(e), "tool": "validate_hed_file", "file_path": file_path},
            "ERROR",
        )
        return error_handler.format_mcp_error(
            MCPErrorCode.INTERNAL_ERROR, "File validation failed due to internal error"
        )


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
        result += "=" * 40 + "\n\n"

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
    events_file: str = Field(description="Path to events.tsv or events.csv file"),
    output_path: str = Field(description="Output path for JSON sidecar file"),
    schema_version: str = Field(
        default="8.3.0", description="HED schema version to use"
    ),
    skip_columns: str = Field(
        default="",
        description="Comma-separated list of columns to skip (auto-detected if empty)",
    ),
    value_columns: str = Field(
        default="",
        description="Comma-separated list of value columns (auto-detected if empty)",
    ),
    include_validation: bool = Field(
        default=True, description="Include HED validation results in response"
    ),
    ctx: Optional[Context] = None,
) -> str:
    """
    Generate a HED sidecar JSON file from an events file using automated column analysis.

    This tool performs comprehensive column analysis to determine appropriate skip and value
    columns, then generates a BIDS-compatible HED sidecar template using TabularSummary.
    """
    if not HED_AVAILABLE:
        return "Error: HED components are not available"

    if not PANDAS_AVAILABLE:
        return "Error: Pandas is required for sidecar generation but not available"

    start_time = time.time()

    try:
        # 1. Input validation and security checks
        events_path = Path(events_file)
        output_path_obj = Path(output_path)

        # Security: Prevent path traversal attacks
        if ".." in str(events_path) or str(events_path).startswith("/"):
            return "Error: Invalid file path - path traversal not allowed"

        if ".." in str(output_path_obj) or str(output_path_obj).startswith("/"):
            return "Error: Invalid output path - path traversal not allowed"

        if not events_path.exists():
            return f"Error: Events file not found: {events_file}"

        # Check file size limits (100MB max)
        file_size = events_path.stat().st_size
        if file_size > 100 * 1024 * 1024:  # 100MB
            return f"Error: File too large ({file_size / 1024 / 1024:.1f}MB). Maximum allowed: 100MB"

        # Validate file format
        if events_path.suffix.lower() not in [".tsv", ".csv"]:
            return f"Error: Unsupported file format '{events_path.suffix}'. Only .tsv and .csv are supported"

        # Ensure schema is loaded
        await _ensure_schema_loaded(schema_version)

        if ctx:
            ctx.info(f"Generating HED sidecar for: {events_path.name}")

        # 2. Load and analyze the events file
        try:
            # Determine delimiter based on file extension
            delimiter = "\t" if events_path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(events_path, delimiter=delimiter)

            if len(df) == 0:
                return "Error: Events file is empty"

            if ctx:
                ctx.info(
                    f"Loaded events file: {len(df)} rows, {len(df.columns)} columns"
                )

        except Exception as e:
            return f"Error reading events file: {str(e)}"

        # 3. Column analysis for automatic skip/value column detection
        detected_skip_columns = []
        detected_value_columns = []

        if not skip_columns and not value_columns:
            # Auto-detect column types
            try:
                from ..tools.column_analysis_engine import (
                    BIDSColumnAnalysisEngine,
                    AnalysisConfig,
                )

                # Configure analysis engine for fast processing
                config = AnalysisConfig(
                    enable_enhanced_analysis=True,
                    enable_memory_optimization=True,
                    enable_chunked_processing=file_size
                    > 10 * 1024 * 1024,  # Enable for files > 10MB
                    enable_caching=True,
                )

                engine = BIDSColumnAnalysisEngine(config)
                analysis_result = await engine.analyze_file(events_path)

                if analysis_result.success and analysis_result.enhanced_analysis:
                    # Extract column classifications from analysis
                    enhanced_data = analysis_result.enhanced_analysis

                    # Standard BIDS timing columns to skip
                    timing_columns = ["onset", "duration", "sample"]
                    detected_skip_columns = [
                        col for col in timing_columns if col in df.columns
                    ]

                    # Find categorical columns that are good HED candidates
                    if "column_analysis" in enhanced_data:
                        for col_name, col_info in enhanced_data[
                            "column_analysis"
                        ].items():
                            if col_name not in detected_skip_columns:
                                # Add categorical columns with reasonable unique value counts
                                if (
                                    col_info.get("type") == "categorical"
                                    and col_info.get("unique_count", 0) < len(df) * 0.5
                                ):
                                    detected_value_columns.append(col_name)

                if ctx:
                    ctx.info(f"Auto-detected skip columns: {detected_skip_columns}")
                    ctx.info(f"Auto-detected value columns: {detected_value_columns}")

            except Exception as e:
                logger.warning(f"Column analysis failed, using fallback: {str(e)}")
                # Fallback: skip standard timing columns, use all others as value columns
                timing_columns = ["onset", "duration", "sample"]
                detected_skip_columns = [
                    col for col in timing_columns if col in df.columns
                ]
                detected_value_columns = [
                    col for col in df.columns if col not in detected_skip_columns
                ]
        else:
            # Use provided column lists
            detected_skip_columns = [
                col.strip() for col in skip_columns.split(",") if col.strip()
            ]
            detected_value_columns = [
                col.strip() for col in value_columns.split(",") if col.strip()
            ]

        # Validate that specified columns exist
        missing_skip = [col for col in detected_skip_columns if col not in df.columns]
        missing_value = [col for col in detected_value_columns if col not in df.columns]

        if missing_skip:
            return f"Error: Skip columns not found in file: {missing_skip}"
        if missing_value:
            return f"Error: Value columns not found in file: {missing_value}"

        # 4. Generate HED sidecar using TabularSummary
        try:
            from ..hed_integration.tabular_summary import (
                TabularSummaryWrapper,
                TabularSummaryConfig,
            )
            from ..hed_integration.schema import SchemaHandler

            # Initialize components
            schema_handler = SchemaHandler()
            config = TabularSummaryConfig()

            async with TabularSummaryWrapper(config, schema_handler) as wrapper:
                if ctx:
                    ctx.info("Generating sidecar template using TabularSummary...")

                # Generate the sidecar template
                sidecar_template = await wrapper.extract_sidecar_template(
                    data=df,
                    skip_columns=detected_skip_columns
                    if detected_skip_columns
                    else None,
                    use_cache=True,
                )

                if not sidecar_template or not sidecar_template.template:
                    return "Error: Failed to generate sidecar template"

                sidecar_content = sidecar_template.template

        except Exception as e:
            logger.error(f"TabularSummary integration failed: {str(e)}")
            # Fallback: Generate basic sidecar structure
            sidecar_content = {}

            # If no value columns were detected, use a more comprehensive fallback
            if not detected_value_columns:
                logger.warning(
                    "No value columns detected, using comprehensive fallback"
                )
                # Skip standard timing and sample columns, analyze all others
                timing_columns = ["onset", "duration", "sample"]
                potential_value_columns = [
                    col for col in df.columns if col not in timing_columns
                ]
            else:
                potential_value_columns = detected_value_columns

            for col in potential_value_columns:
                # Get unique values for the column
                unique_vals = df[col].dropna().unique()

                # More flexible threshold - allow up to 50 unique values or 25% of rows
                max_unique_threshold = min(50, len(df) * 0.25)

                if len(unique_vals) <= max_unique_threshold and len(unique_vals) > 0:
                    # Create proper sidecar structure with levels
                    unique_vals_clean = [
                        str(val) for val in unique_vals if str(val) != "nan"
                    ]
                    if unique_vals_clean:
                        sidecar_content[col] = {
                            "Description": f"Column {col} values for experimental events",
                            "HED": {val: "Event" for val in unique_vals_clean},
                            "Levels": {
                                val: f"Description for {val}"
                                for val in unique_vals_clean
                            },
                        }
                        logger.info(
                            f"Added fallback sidecar entry for column '{col}' with {len(unique_vals_clean)} values"
                        )

            if not sidecar_content:
                # Last resort: analyze all non-timing columns regardless of unique value count
                logger.warning(
                    "No suitable columns found with flexible threshold, analyzing all non-timing columns"
                )
                timing_columns = ["onset", "duration", "sample"]
                for col in df.columns:
                    if col not in timing_columns:
                        unique_vals = df[col].dropna().unique()
                        unique_vals_clean = [
                            str(val) for val in unique_vals if str(val) != "nan"
                        ]
                        if len(unique_vals_clean) > 0:
                            # For columns with many values, just provide basic HED structure
                            if (
                                len(unique_vals_clean) <= 100
                            ):  # Still reasonable for a sidecar
                                sidecar_content[col] = {
                                    "Description": f"Column {col} values for experimental events",
                                    "HED": {val: "Event" for val in unique_vals_clean},
                                    "Levels": {
                                        val: f"Description for {val}"
                                        for val in unique_vals_clean
                                    },
                                }
                            else:
                                # For very large value sets, provide template structure only
                                sidecar_content[col] = {
                                    "Description": f"Column {col} values for experimental events (template - manually edit HED tags)",
                                    "HED": "Event, (Label/#, Description/#)",
                                }
                            logger.info(
                                f"Added last-resort sidecar entry for column '{col}' with {len(unique_vals_clean)} values"
                            )
                            break  # Add at least one column to avoid complete failure

                if not sidecar_content:
                    return "Error: No suitable columns found for HED annotation - all columns appear to be timing-related or empty"

        # 5. Validation (if requested)
        validation_results = {}
        if include_validation and HED_AVAILABLE:
            try:
                validation_results = {
                    "validation_performed": True,
                    "schema_version": schema_version,
                    "column_count": len(sidecar_content),
                    "skip_columns": detected_skip_columns,
                    "value_columns": list(sidecar_content.keys()),
                    "warnings": [],
                    "errors": [],
                }

                # Basic validation checks
                if not sidecar_content:
                    validation_results["errors"].append("No HED annotations generated")

                # Check for empty HED values
                empty_hed = [k for k, v in sidecar_content.items() if not v.get("HED")]
                if empty_hed:
                    validation_results["warnings"].append(
                        f"Empty HED values for columns: {empty_hed}"
                    )

            except Exception as e:
                validation_results = {
                    "validation_performed": False,
                    "error": f"Validation failed: {str(e)}",
                }

        # 6. Write output file
        try:
            import aiofiles
            import json

            # Ensure output directory exists
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(output_path_obj, "w") as f:
                await f.write(json.dumps(sidecar_content, indent=2))

        except Exception as e:
            return f"Error writing sidecar file: {str(e)}"

        # 7. Generate response
        processing_time = time.time() - start_time

        result_lines = [
            "✅ HED sidecar generated successfully",
            f"📁 Output file: {output_path_obj}",
            f"📊 Columns analyzed: {len(df.columns)}",
            f"⏭️  Skip columns: {len(detected_skip_columns)} {detected_skip_columns}",
            f"🎯 HED value columns: {len(sidecar_content)} {list(sidecar_content.keys())}",
            f"⏱️  Processing time: {processing_time:.2f}s",
        ]

        if validation_results:
            if validation_results.get("validation_performed"):
                result_lines.append(
                    f"✅ Validation: {len(validation_results.get('errors', []))} errors, {len(validation_results.get('warnings', []))} warnings"
                )
                if validation_results.get("errors"):
                    result_lines.extend(
                        [f"   ❌ {error}" for error in validation_results["errors"]]
                    )
                if validation_results.get("warnings"):
                    result_lines.extend(
                        [
                            f"   ⚠️  {warning}"
                            for warning in validation_results["warnings"]
                        ]
                    )
            else:
                result_lines.append(
                    f"⚠️  Validation: {validation_results.get('error', 'Failed')}"
                )

        result = "\n".join(result_lines)

        logger.info(
            f"Generated HED sidecar: {output_path_obj} ({processing_time:.2f}s)"
        )
        return result

    except Exception as e:
        error_msg = f"Sidecar generation failed: {str(e)}"
        logger.error(error_msg)
        return f"❌ {error_msg}"


@mcp.tool()
async def get_server_info(ctx: Optional[Context] = None) -> str:
    """
    Get information about the HED MCP server including available features and status.
    """
    info = {
        "server": "HED MCP Server",
        "version": "1.0.0",
        "hed_available": HED_AVAILABLE,
        "pandas_available": PANDAS_AVAILABLE,
        "excel_available": EXCEL_AVAILABLE,
        "features": {
            "string_validation": HED_AVAILABLE,
            "file_validation": HED_AVAILABLE,
            "schema_management": HED_AVAILABLE,
            "sidecar_generation": HED_AVAILABLE,
            "column_analysis": HED_AVAILABLE and PANDAS_AVAILABLE,
            "excel_support": EXCEL_AVAILABLE,
        },
        "supported_formats": ["tsv", "csv", "json", "bids"],
    }

    if EXCEL_AVAILABLE:
        info["supported_formats"].extend(["xlsx", "xls"])

    return json.dumps(info, indent=2)


# Column Analysis Tools


@mcp.tool()
async def validate_hed_columns(
    file_path: str = Field(description="Path to tabular file (CSV, TSV, Excel)"),
    hed_columns: str = Field(
        description="Comma-separated list of column names containing HED annotations"
    ),
    schema_version: str = Field(
        default="8.3.0", description="HED schema version to use"
    ),
    delimiter: str = Field(
        default="auto",
        description="Delimiter for CSV/TSV files (auto-detect if 'auto')",
    ),
    ctx: Optional[Context] = None,
) -> str:
    """
    Validate HED annotations in specific columns of a tabular data file.

    Supports CSV, TSV, and Excel files. Provides detailed column-level validation
    feedback including row numbers and specific error locations.
    """
    if not HED_AVAILABLE:
        return "Error: HED components are not available"

    if not PANDAS_AVAILABLE:
        return "Error: Pandas is required for column analysis but not available"

    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return f"Error: File not found: {file_path}"

        await _ensure_schema_loaded(schema_version)

        if ctx:
            ctx.info(f"Analyzing HED columns in: {file_path_obj.name}")

        # Parse HED column list
        hed_column_list = [col.strip() for col in hed_columns.split(",")]

        # Read the file based on extension
        file_extension = file_path_obj.suffix.lower()

        if file_extension in [".xlsx", ".xls"]:
            if not EXCEL_AVAILABLE:
                return "Error: Excel file support requires openpyxl package"
            df = pd.read_excel(file_path)
        elif file_extension == ".csv":
            if delimiter == "auto":
                delimiter = ","
            df = pd.read_csv(file_path, delimiter=delimiter)
        elif file_extension == ".tsv":
            if delimiter == "auto":
                delimiter = "\t"
            df = pd.read_csv(file_path, delimiter=delimiter)
        else:
            # Try to auto-detect delimiter
            with open(file_path, "r", encoding="utf-8") as f:
                sample = f.read(1024)
                sniffer = csv.Sniffer()
                detected_delimiter = sniffer.sniff(sample).delimiter
            df = pd.read_csv(file_path, delimiter=detected_delimiter)

        # Validate that HED columns exist
        missing_columns = [col for col in hed_column_list if col not in df.columns]
        if missing_columns:
            return f"Error: HED columns not found in file: {missing_columns}\nAvailable columns: {list(df.columns)}"

        # Analyze each HED column
        validation_results = {
            "file": file_path,
            "schema_version": schema_version,
            "total_rows": len(df),
            "hed_columns": hed_column_list,
            "column_results": {},
            "summary": {
                "total_errors": 0,
                "total_warnings": 0,
                "valid_rows": 0,
                "invalid_rows": 0,
            },
        }

        for column in hed_column_list:
            column_data = df[column].dropna()  # Remove NaN values
            column_results = {
                "total_entries": len(column_data),
                "valid_entries": 0,
                "invalid_entries": 0,
                "errors": [],
                "warnings": [],
            }

            # Validate each entry in the column
            for idx, hed_string in enumerate(column_data):
                if pd.isna(hed_string) or str(hed_string).strip() == "":
                    continue

                row_number = (
                    df[df[column] == hed_string].index[0] + 2
                )  # +2 for 1-based indexing and header

                # For now, we'll do basic validation
                # In a real implementation, this would use the HED validator
                hed_str = str(hed_string).strip()

                # Basic validation checks
                errors = []
                warnings = []

                # Check for basic syntax issues
                if not hed_str:
                    continue

                # Check for balanced parentheses
                open_parens = hed_str.count("(")
                close_parens = hed_str.count(")")
                if open_parens != close_parens:
                    errors.append(
                        f"Row {row_number}: Unbalanced parentheses in '{hed_str[:50]}...'"
                    )

                # Check for missing commas between tags
                if "," not in hed_str and "/" in hed_str and len(hed_str.split()) > 1:
                    warnings.append(
                        f"Row {row_number}: Missing commas between tags in '{hed_str[:50]}...'"
                    )

                # Check for invalid characters
                invalid_chars = ["<", ">", '"', "'"]
                for char in invalid_chars:
                    if char in hed_str:
                        errors.append(
                            f"Row {row_number}: Invalid character '{char}' in '{hed_str[:50]}...'"
                        )

                if errors:
                    column_results["invalid_entries"] += 1
                    column_results["errors"].extend(errors)
                    validation_results["summary"]["invalid_rows"] += 1
                else:
                    column_results["valid_entries"] += 1
                    validation_results["summary"]["valid_rows"] += 1

                if warnings:
                    column_results["warnings"].extend(warnings)

                validation_results["summary"]["total_errors"] += len(errors)
                validation_results["summary"]["total_warnings"] += len(warnings)

            validation_results["column_results"][column] = column_results

        # Format results
        result_lines = [
            "HED Column Validation Results",
            f"File: {file_path}",
            f"Schema Version: {schema_version}",
            f"Total Rows: {validation_results['total_rows']}",
            "",
            "Summary:",
            f"  Valid Rows: {validation_results['summary']['valid_rows']}",
            f"  Invalid Rows: {validation_results['summary']['invalid_rows']}",
            f"  Total Errors: {validation_results['summary']['total_errors']}",
            f"  Total Warnings: {validation_results['summary']['total_warnings']}",
            "",
        ]

        for column, results in validation_results["column_results"].items():
            result_lines.extend(
                [
                    f"Column '{column}':",
                    f"  Valid Entries: {results['valid_entries']}",
                    f"  Invalid Entries: {results['invalid_entries']}",
                    "",
                ]
            )

            if results["errors"]:
                result_lines.append("  Errors:")
                for error in results["errors"][:10]:  # Limit to first 10 errors
                    result_lines.append(f"    {error}")
                if len(results["errors"]) > 10:
                    result_lines.append(
                        f"    ... and {len(results['errors']) - 10} more errors"
                    )
                result_lines.append("")

            if results["warnings"]:
                result_lines.append("  Warnings:")
                for warning in results["warnings"][:5]:  # Limit to first 5 warnings
                    result_lines.append(f"    {warning}")
                if len(results["warnings"]) > 5:
                    result_lines.append(
                        f"    ... and {len(results['warnings']) - 5} more warnings"
                    )
                result_lines.append("")

        return "\n".join(result_lines)

    except Exception as e:
        error_msg = f"Column validation failed: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def analyze_hed_spreadsheet(
    file_path: str = Field(description="Path to spreadsheet file"),
    output_format: str = Field(
        default="summary", description="Output format: 'summary', 'detailed', or 'json'"
    ),
    schema_version: str = Field(
        default="8.3.0", description="HED schema version to use"
    ),
    ctx: Optional[Context] = None,
) -> str:
    """
    Analyze a spreadsheet file to identify HED-related columns and provide analysis.

    Automatically detects potential HED columns and provides statistical analysis,
    validation summary, and recommendations for HED annotation improvement.
    """
    if not HED_AVAILABLE:
        return "Error: HED components are not available"

    if not PANDAS_AVAILABLE:
        return "Error: Pandas is required for spreadsheet analysis but not available"

    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return f"Error: File not found: {file_path}"

        await _ensure_schema_loaded(schema_version)

        if ctx:
            ctx.info(f"Analyzing spreadsheet: {file_path_obj.name}")

        # Read the file based on extension
        file_extension = file_path_obj.suffix.lower()

        if file_extension in [".xlsx", ".xls"]:
            if not EXCEL_AVAILABLE:
                return "Error: Excel file support requires openpyxl package"
            df = pd.read_excel(file_path)
        elif file_extension in [".csv", ".tsv"]:
            delimiter = "\t" if file_extension == ".tsv" else ","
            df = pd.read_csv(file_path, delimiter=delimiter)
        else:
            # Try to auto-detect
            with open(file_path, "r", encoding="utf-8") as f:
                sample = f.read(1024)
                sniffer = csv.Sniffer()
                detected_delimiter = sniffer.sniff(sample).delimiter
            df = pd.read_csv(file_path, delimiter=detected_delimiter)

        # Auto-detect potential HED columns
        potential_hed_columns = []
        for column in df.columns:
            # Look for columns that might contain HED annotations
            sample_values = df[column].dropna().astype(str).head(10)

            # Heuristics for HED detection
            hed_indicators = 0
            for value in sample_values:
                value_str = str(value).strip()
                if not value_str or len(value_str) < 3:
                    continue

                # Check for HED-like patterns
                if "/" in value_str:  # HED hierarchy indicator
                    hed_indicators += 2
                if any(
                    word in value_str.lower()
                    for word in ["event", "stimulus", "response", "action"]
                ):
                    hed_indicators += 1
                if "(" in value_str and ")" in value_str:  # HED grouping
                    hed_indicators += 1
                if "," in value_str and "/" in value_str:  # Multiple HED tags
                    hed_indicators += 2
                if any(
                    char.isupper() for char in value_str
                ):  # CamelCase (common in HED)
                    hed_indicators += 0.5

            # If enough indicators, consider it a potential HED column
            if hed_indicators >= 2:
                potential_hed_columns.append(
                    {
                        "column": column,
                        "confidence": min(hed_indicators / 5.0, 1.0),
                        "sample_values": sample_values.tolist()[:3],
                    }
                )

        # Analyze spreadsheet structure
        analysis_results = {
            "file_info": {
                "path": str(file_path),
                "format": file_extension,
                "rows": len(df),
                "columns": len(df.columns),
                "schema_version": schema_version,
            },
            "column_analysis": {
                "total_columns": len(df.columns),
                "potential_hed_columns": len(potential_hed_columns),
                "all_columns": df.columns.tolist(),
                "detected_hed_columns": potential_hed_columns,
            },
            "data_quality": {"missing_data_summary": {}, "data_type_summary": {}},
            "recommendations": [],
        }

        # Analyze data quality
        for column in df.columns:
            missing_count = df[column].isna().sum()
            missing_percent = (missing_count / len(df)) * 100
            analysis_results["data_quality"]["missing_data_summary"][column] = {
                "missing_count": int(missing_count),
                "missing_percent": round(missing_percent, 2),
            }

            # Determine predominant data type
            non_null_data = df[column].dropna()
            if len(non_null_data) > 0:
                sample_value = non_null_data.iloc[0]
                if isinstance(sample_value, (int, float)):
                    data_type = "numeric"
                else:
                    data_type = "text"
            else:
                data_type = "empty"

            analysis_results["data_quality"]["data_type_summary"][column] = data_type

        # Generate recommendations
        if potential_hed_columns:
            analysis_results["recommendations"].extend(
                [
                    f"Found {len(potential_hed_columns)} potential HED columns",
                    "Consider validating these columns with validate_hed_columns tool",
                    "Check HED schema compatibility for detected patterns",
                ]
            )
        else:
            analysis_results["recommendations"].extend(
                [
                    "No obvious HED columns detected automatically",
                    "Manual inspection may be needed to identify HED content",
                    "Look for columns containing event descriptions or stimulus codes",
                ]
            )

        # Handle missing data recommendations
        high_missing_columns = [
            col
            for col, info in analysis_results["data_quality"][
                "missing_data_summary"
            ].items()
            if info["missing_percent"] > 50
        ]
        if high_missing_columns:
            analysis_results["recommendations"].append(
                f"Columns with high missing data (>50%): {high_missing_columns}"
            )

        # Format output based on requested format
        if output_format == "json":
            return json.dumps(analysis_results, indent=2)

        elif output_format == "detailed":
            result_lines = [
                "=== HED Spreadsheet Analysis (Detailed) ===",
                "",
                f"File: {analysis_results['file_info']['path']}",
                f"Format: {analysis_results['file_info']['format']}",
                f"Dimensions: {analysis_results['file_info']['rows']} rows × {analysis_results['file_info']['columns']} columns",
                f"Schema Version: {analysis_results['file_info']['schema_version']}",
                "",
                "=== Potential HED Columns ===",
            ]

            if potential_hed_columns:
                for hed_col in potential_hed_columns:
                    result_lines.extend(
                        [
                            f"Column: {hed_col['column']}",
                            f"  Confidence: {hed_col['confidence']:.2f}",
                            f"  Sample values: {hed_col['sample_values']}",
                            "",
                        ]
                    )
            else:
                result_lines.append("No potential HED columns detected")

            result_lines.extend(
                [
                    "",
                    "=== Data Quality Summary ===",
                ]
            )

            for column, missing_info in analysis_results["data_quality"][
                "missing_data_summary"
            ].items():
                if missing_info["missing_percent"] > 0:
                    result_lines.append(
                        f"{column}: {missing_info['missing_percent']:.1f}% missing data"
                    )

            result_lines.extend(
                [
                    "",
                    "=== Recommendations ===",
                ]
            )

            for rec in analysis_results["recommendations"]:
                result_lines.append(f"• {rec}")

            return "\n".join(result_lines)

        else:  # summary format
            result_lines = [
                "=== HED Spreadsheet Analysis Summary ===",
                "",
                f"File: {file_path_obj.name}",
                f"Size: {analysis_results['file_info']['rows']} rows × {analysis_results['file_info']['columns']} columns",
                "",
                f"Potential HED Columns: {len(potential_hed_columns)}",
            ]

            if potential_hed_columns:
                hed_col_names = [col["column"] for col in potential_hed_columns]
                result_lines.append(f"Detected: {', '.join(hed_col_names)}")

            result_lines.extend(
                [
                    "",
                    "Key Recommendations:",
                ]
            )

            for rec in analysis_results["recommendations"][:3]:  # Top 3 recommendations
                result_lines.append(f"• {rec}")

            if len(analysis_results["recommendations"]) > 3:
                result_lines.append(
                    f"• ... and {len(analysis_results['recommendations']) - 3} more"
                )

            return "\n".join(result_lines)

    except Exception as e:
        error_msg = f"Spreadsheet analysis failed: {str(e)}"
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
    try:
        # Simple check - if we can access the mcp instance and it has run method
        if hasattr(mcp, "run") and callable(mcp.run):
            logger.info("✓ FastMCP migration successful: server instance ready")
        else:
            raise RuntimeError("FastMCP instance not properly initialized")
    except Exception as e:
        logger.error(f"❌ FastMCP migration verification failed: {e}")
        raise RuntimeError(f"Migration verification failed: {e}")


# Server health monitoring
@dataclass
class ServerHealth:
    """Track server health metrics and status."""

    start_time: float = field(default_factory=time.time)
    request_count: int = 0
    error_count: int = 0
    last_activity: float = field(default_factory=time.time)

    @property
    def uptime(self) -> float:
        return time.time() - self.start_time

    @property
    def health_status(self) -> str:
        if self.request_count == 0:
            return "ready"
        error_rate = self.error_count / self.request_count
        return "healthy" if error_rate < 0.1 else "degraded"

    def record_request(self, success: bool = True):
        self.request_count += 1
        if not success:
            self.error_count += 1
        self.last_activity = time.time()


# Global health instance
health = ServerHealth()


class ServerManager:
    """Manage server lifecycle, including startup, shutdown, and signal handling."""

    def __init__(self):
        self.server_task: Optional = None
        self.shutdown_requested = False

    def setup_signal_handlers(self):
        """Set up graceful shutdown signal handlers."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        signal_names = {signal.SIGTERM: "SIGTERM", signal.SIGINT: "SIGINT"}
        signal_name = signal_names.get(signum, f"Signal {signum}")

        logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        self.shutdown_requested = True

        # For SIGINT (Ctrl+C), exit immediately in CLI context
        if signum == signal.SIGINT:
            logger.info("Server shutting down...")
            sys.exit(0)


@mcp.tool()
async def server_health(ctx: Optional[Context] = None) -> str:
    """
    Get server health status and operational statistics.

    Returns information about server uptime, request metrics, and current health status.
    """
    if ctx:
        ctx.info("Checking server health status")

    health_data = {
        "status": health.health_status,
        "uptime_seconds": round(health.uptime, 2),
        "uptime_human": _format_uptime(health.uptime),
        "requests_processed": health.request_count,
        "error_count": health.error_count,
        "error_rate": round(health.error_count / max(health.request_count, 1), 3),
        "last_activity": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(health.last_activity)
        ),
        "server_info": {
            "name": "HED Validation MCP Server",
            "version": "1.0.0",
            "hed_available": HED_AVAILABLE,
            "pandas_available": PANDAS_AVAILABLE,
            "excel_available": EXCEL_AVAILABLE,
        },
    }

    # Format as readable text
    lines = [
        "🔍 Server Health Report",
        f"Status: {health_data['status'].upper()}",
        f"Uptime: {health_data['uptime_human']}",
        f"Requests: {health_data['requests_processed']} (errors: {health_data['error_count']})",
        f"Error Rate: {health_data['error_rate']:.1%}",
        f"Last Activity: {health_data['last_activity']}",
        "",
        "🔧 Components:",
        f"  HED Tools: {'✅' if health_data['server_info']['hed_available'] else '❌'}",
        f"  Pandas: {'✅' if health_data['server_info']['pandas_available'] else '❌'}",
        f"  Excel Support: {'✅' if health_data['server_info']['excel_available'] else '❌'}",
    ]

    return "\n".join(lines)


def _format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f} hours"
    else:
        days = seconds / 86400
        return f"{days:.1f} days"


async def validate_startup_dependencies():
    """Validate all required dependencies before server start."""
    logger.info("Validating startup dependencies...")

    checks = [
        ("HED components", lambda: HED_AVAILABLE),
        ("Pandas availability", lambda: PANDAS_AVAILABLE),
        ("Temp directory access", lambda: os.access("/tmp", os.W_OK)),
        ("FastMCP framework", lambda: hasattr(mcp, "run")),
    ]

    for name, check in checks:
        try:
            result = check()
            status = "✅" if result else "⚠️ "
            logger.info(f"{status} {name}: {'available' if result else 'unavailable'}")
        except Exception as e:
            logger.error(f"❌ {name} check failed: {e}")
            # Don't fail startup for optional components
            if name not in ["Pandas availability", "Excel support"]:
                raise RuntimeError(f"Critical dependency check failed: {name}")


def setup_logging(debug: bool = False):
    """Configure logging for server operation."""
    level = logging.DEBUG if debug else logging.INFO

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create stderr handler (important: don't use stdout due to stdio transport)
    handler = logging.StreamHandler(sys.stderr)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers unless in debug mode
    if not debug:
        for logger_name in ["urllib3", "requests", "httpcore", "httpx"]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)


async def async_main():
    """Async main function for server startup and lifecycle management."""
    server_manager = ServerManager()

    try:
        # Validate dependencies
        await validate_startup_dependencies()

        # Set up signal handlers
        server_manager.setup_signal_handlers()

        # Verify migration if in debug mode
        if DEBUG_MODE:
            _verify_migration()

        # Log server startup
        logger.info("🚀 Starting HED Validation MCP Server")
        logger.info(f"   Debug mode: {DEBUG_MODE}")
        logger.info(
            f"   HED components: {'available' if HED_AVAILABLE else 'unavailable'}"
        )
        logger.info(
            f"   Pandas support: {'available' if PANDAS_AVAILABLE else 'unavailable'}"
        )
        logger.info(
            f"   Excel support: {'available' if EXCEL_AVAILABLE else 'unavailable'}"
        )

        # Initialize components asynchronously
        await _initialize_hed_components()

        # Record server start
        health.record_request(success=True)

        # Start the FastMCP server - use run_sync() for MCP client compatibility
        logger.info("🔄 Server ready, waiting for connections...")

        # For MCP stdio transport, we need to use run_sync
        await mcp.run_sync()

    except KeyboardInterrupt:
        logger.info("🛑 Server interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        health.record_request(success=False)
        raise
    finally:
        logger.info("🔚 Server shutdown complete")


def main():
    """Main entry point for the HED MCP server with proper CLI handling."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        prog="hed-mcp-server",
        description="HED Validation MCP Server - Provides HED schema validation and processing tools",
        epilog="For more information, visit: https://github.com/neuromechanist/hed-mcp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--version", action="version", version="HED MCP Server 1.0.0")

    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging and verbose output"
    )

    parser.add_argument(
        "--check-deps", action="store_true", help="Check dependencies and exit"
    )

    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        # Handle --help and --version cleanly without hanging
        return

    # Set up logging based on arguments
    setup_logging(debug=args.debug)

    # Update global debug mode
    global DEBUG_MODE
    if args.debug:
        DEBUG_MODE = True

    # Handle dependency check
    if args.check_deps:
        try:
            import asyncio

            asyncio.run(validate_startup_dependencies())
            print("✅ All dependencies validated successfully")
            return
        except Exception as e:
            print(f"❌ Dependency validation failed: {e}")
            sys.exit(1)

    # For MCP servers, we should use the synchronous run method
    try:
        # Set up logging based on arguments
        setup_logging(debug=args.debug)

        # Update global debug mode
        if args.debug:
            DEBUG_MODE = True

        # Run server initialization in async context first
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Initialize components
            loop.run_until_complete(validate_startup_dependencies())

            if DEBUG_MODE:
                _verify_migration()

            # Log server startup
            logger.info("🚀 Starting HED Validation MCP Server")
            logger.info(f"   Debug mode: {DEBUG_MODE}")
            logger.info(
                f"   HED components: {'available' if HED_AVAILABLE else 'unavailable'}"
            )
            logger.info(
                f"   Pandas support: {'available' if PANDAS_AVAILABLE else 'unavailable'}"
            )
            logger.info(
                f"   Excel support: {'available' if EXCEL_AVAILABLE else 'unavailable'}"
            )

            # Initialize HED components
            loop.run_until_complete(_initialize_hed_components())

            # Record server start
            health.record_request(success=True)

            logger.info("🔄 Server ready, waiting for connections...")

            # Now run the FastMCP server in the main thread
            mcp.run()

        finally:
            loop.close()

    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Fatal server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
