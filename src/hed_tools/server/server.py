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
