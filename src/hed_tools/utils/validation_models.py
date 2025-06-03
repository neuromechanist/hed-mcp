"""
Pydantic validation models for HED MCP Server tools.

This module provides comprehensive input validation models with security
validators for all server tool parameters, ensuring type safety and
preventing injection attacks.
"""

import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, validator

from .security import sanitize_input


class HedVersionValidator:
    """Common validator for HED schema versions."""

    VALID_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")

    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate HED schema version format."""
        if not cls.VALID_VERSION_PATTERN.match(v):
            raise ValueError("Schema version must be in format 'X.Y.Z' (e.g., '8.3.0')")
        return v


class SecurePathValidator:
    """Common validator for file paths with security checks."""

    @classmethod
    def validate_file_path(cls, v: str, field_name: str = "path") -> str:
        """Validate file paths with security checks."""
        if not v:
            return v

        # Basic security checks
        if ".." in v:
            raise ValueError(f"Path traversal attempt detected in {field_name}")

        # Check for absolute paths (potential security risk)
        if v.startswith("/") and not v.startswith("/tmp"):
            raise ValueError(
                f"Absolute paths not allowed for {field_name} (except /tmp)"
            )

        # Check for dangerous characters
        dangerous_chars = ["|", ";", "&", "$", "`", "<", ">"]
        if any(char in v for char in dangerous_chars):
            raise ValueError(f"Dangerous characters detected in {field_name}")

        return v

    @classmethod
    def validate_output_path(cls, v: Optional[str]) -> Optional[str]:
        """Validate output paths with additional write permission considerations."""
        if not v:
            return v

        # Apply basic path validation
        v = cls.validate_file_path(v, "output_path")

        # Additional checks for output paths
        if v.endswith("/"):
            raise ValueError("Output path cannot be a directory")

        return v


class SecureFileTypeValidator:
    """Common validator for file extensions and types."""

    ALLOWED_EXTENSIONS = {
        "events": [".tsv", ".csv", ".txt"],
        "sidecar": [".json"],
        "schema": [".xml", ".json"],
        "spreadsheet": [".xlsx", ".xls", ".csv", ".tsv"],
    }

    @classmethod
    def validate_file_extension(cls, v: str, file_type: str) -> str:
        """Validate file extensions against allowed types."""
        if not v:
            return v

        path = Path(v)
        extension = path.suffix.lower()

        allowed = cls.ALLOWED_EXTENSIONS.get(file_type, [])
        if extension and extension not in allowed:
            raise ValueError(
                f"Invalid file extension '{extension}' for {file_type}. "
                f"Allowed: {', '.join(allowed)}"
            )

        return v


class HedStringValidationRequest(BaseModel):
    """Validation model for HED string validation requests."""

    hed_string: str = Field(
        ..., min_length=1, max_length=50000, description="HED string to validate"
    )
    schema_version: str = Field(
        default="8.3.0", description="HED schema version to use for validation"
    )
    check_for_warnings: bool = Field(
        default=True, description="Whether to check for validation warnings"
    )

    @validator("hed_string")
    def sanitize_hed_string(cls, v):
        """Sanitize HED string input."""
        return sanitize_input(v, max_length=50000)

    @validator("schema_version")
    def validate_schema_version(cls, v):
        """Validate schema version format."""
        return HedVersionValidator.validate_version(v)

    class Config:
        extra = "forbid"  # Prevent additional fields


class HedFileValidationRequest(BaseModel):
    """Validation model for HED file validation requests."""

    file_path: str = Field(
        ..., max_length=500, description="Path to file for validation"
    )
    schema_version: str = Field(
        default="8.3.0", description="HED schema version to use for validation"
    )
    check_for_warnings: bool = Field(
        default=True, description="Whether to check for validation warnings"
    )

    @validator("file_path")
    def validate_file_path(cls, v):
        """Validate and sanitize file path."""
        v = SecurePathValidator.validate_file_path(v, "file_path")
        return SecureFileTypeValidator.validate_file_extension(v, "events")

    @validator("schema_version")
    def validate_schema_version(cls, v):
        """Validate schema version format."""
        return HedVersionValidator.validate_version(v)

    class Config:
        extra = "forbid"


class HedSidecarGenerationRequest(BaseModel):
    """Validation model for HED sidecar generation requests."""

    events_file: str = Field(
        ..., max_length=500, description="Path to events.tsv or events.csv file"
    )
    output_path: Optional[str] = Field(
        None, max_length=500, description="Output path for JSON sidecar file"
    )
    schema_version: str = Field(
        default="8.3.0", description="HED schema version to use"
    )
    skip_columns: str = Field(
        default="",
        max_length=1000,
        description="Comma-separated list of columns to skip",
    )
    value_columns: str = Field(
        default="", max_length=1000, description="Comma-separated list of value columns"
    )
    include_validation: bool = Field(
        default=True, description="Include validation in the process"
    )

    @validator("events_file")
    def validate_events_file(cls, v):
        """Validate events file path and type."""
        v = SecurePathValidator.validate_file_path(v, "events_file")
        return SecureFileTypeValidator.validate_file_extension(v, "events")

    @validator("output_path")
    def validate_output_path(cls, v):
        """Validate output path."""
        if v:
            v = SecurePathValidator.validate_output_path(v)
            v = SecureFileTypeValidator.validate_file_extension(v, "sidecar")
        return v

    @validator("schema_version")
    def validate_schema_version(cls, v):
        """Validate schema version format."""
        return HedVersionValidator.validate_version(v)

    @validator("skip_columns", "value_columns")
    def validate_column_lists(cls, v):
        """Validate and sanitize column lists."""
        if not v:
            return v

        # Sanitize and validate column names
        columns = [col.strip() for col in v.split(",") if col.strip()]

        for col in columns:
            # Check for valid column name format
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", col):
                raise ValueError(f"Invalid column name format: '{col}'")

            # Check length
            if len(col) > 100:
                raise ValueError(f"Column name too long: '{col}'")

        return ",".join(columns)

    class Config:
        extra = "forbid"


class SchemaSearchRequest(BaseModel):
    """Validation model for schema search requests."""

    search_term: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Term to search for in HED schema",
    )
    schema_version: str = Field(
        default="8.3.0", description="HED schema version to search"
    )
    search_type: str = Field(
        default="tag",
        pattern="^(tag|attribute|unit|unitclass|valueclasses)$",
        description="Type of schema element to search",
    )
    case_sensitive: bool = Field(
        default=False, description="Whether search should be case sensitive"
    )

    @validator("search_term")
    def sanitize_search_term(cls, v):
        """Sanitize search term."""
        return sanitize_input(v, max_length=200)

    @validator("schema_version")
    def validate_schema_version(cls, v):
        """Validate schema version format."""
        return HedVersionValidator.validate_version(v)

    class Config:
        extra = "forbid"


class ServerInfoRequest(BaseModel):
    """Validation model for server info requests."""

    include_health: bool = Field(
        default=True, description="Include health check information"
    )
    include_tools: bool = Field(
        default=True, description="Include available tools information"
    )
    include_schemas: bool = Field(
        default=False, description="Include available schemas information"
    )

    class Config:
        extra = "forbid"


class SpreadsheetValidationRequest(BaseModel):
    """Validation model for spreadsheet validation requests."""

    spreadsheet_path: str = Field(
        ..., max_length=500, description="Path to spreadsheet file"
    )
    schema_version: str = Field(
        default="8.3.0", description="HED schema version to use"
    )
    worksheet_name: Optional[str] = Field(
        None, max_length=100, description="Name of worksheet to validate (Excel only)"
    )
    tag_columns: str = Field(
        default="", max_length=1000, description="Comma-separated list of tag columns"
    )
    has_column_names: bool = Field(
        default=True, description="Whether spreadsheet has column names in first row"
    )

    @validator("spreadsheet_path")
    def validate_spreadsheet_path(cls, v):
        """Validate spreadsheet file path and type."""
        v = SecurePathValidator.validate_file_path(v, "spreadsheet_path")
        return SecureFileTypeValidator.validate_file_extension(v, "spreadsheet")

    @validator("schema_version")
    def validate_schema_version(cls, v):
        """Validate schema version format."""
        return HedVersionValidator.validate_version(v)

    @validator("worksheet_name")
    def validate_worksheet_name(cls, v):
        """Validate worksheet name."""
        if v:
            v = sanitize_input(v, max_length=100)
        return v

    @validator("tag_columns")
    def validate_tag_columns(cls, v):
        """Validate tag columns list."""
        if not v:
            return v

        # Parse and validate column identifiers (names or indices)
        columns = [col.strip() for col in v.split(",") if col.strip()]

        for col in columns:
            # Check if it's a column index (number) or name
            if col.isdigit():
                col_num = int(col)
                if col_num < 1 or col_num > 1000:  # Reasonable spreadsheet column limit
                    raise ValueError(f"Column index out of range: {col_num}")
            else:
                # Validate column name format
                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_\s]*$", col):
                    raise ValueError(f"Invalid column name format: '{col}'")

                if len(col) > 100:
                    raise ValueError(f"Column name too long: '{col}'")

        return ",".join(columns)

    class Config:
        extra = "forbid"


class BatchValidationRequest(BaseModel):
    """Validation model for batch validation requests."""

    input_files: List[str] = Field(
        ..., min_items=1, max_items=50, description="List of file paths to validate"
    )
    schema_version: str = Field(
        default="8.3.0", description="HED schema version to use"
    )
    output_dir: Optional[str] = Field(
        None, max_length=500, description="Directory for validation report output"
    )
    fail_fast: bool = Field(default=False, description="Stop validation on first error")

    @validator("input_files")
    def validate_input_files(cls, v):
        """Validate list of input files."""
        validated_files = []
        for file_path in v:
            validated_path = SecurePathValidator.validate_file_path(
                file_path, "input_file"
            )
            validated_files.append(validated_path)

        # Check for duplicates
        if len(set(validated_files)) != len(validated_files):
            raise ValueError("Duplicate file paths in input list")

        return validated_files

    @validator("schema_version")
    def validate_schema_version(cls, v):
        """Validate schema version format."""
        return HedVersionValidator.validate_version(v)

    @validator("output_dir")
    def validate_output_dir(cls, v):
        """Validate output directory path."""
        if v:
            v = SecurePathValidator.validate_file_path(v, "output_dir")
        return v

    class Config:
        extra = "forbid"


# Request validation helper functions
def validate_request_model(
    model_class: type, request_data: Dict[str, Any]
) -> BaseModel:
    """
    Validate request data against a Pydantic model with comprehensive error handling.

    Args:
        model_class: Pydantic model class to validate against
        request_data: Raw request data to validate

    Returns:
        Validated model instance

    Raises:
        ValueError: If validation fails
    """
    try:
        return model_class(**request_data)
    except Exception as e:
        # Sanitize error message to prevent information leakage
        error_msg = str(e)
        if "password" in error_msg.lower() or "token" in error_msg.lower():
            error_msg = "Validation failed: sensitive data in request"
        elif len(error_msg) > 200:
            error_msg = error_msg[:200] + "..."

        raise ValueError(f"Request validation failed: {error_msg}")


def get_validation_model(tool_name: str) -> Optional[type]:
    """
    Get the appropriate validation model for a tool.

    Args:
        tool_name: Name of the tool

    Returns:
        Pydantic model class or None if no specific model exists
    """
    model_mapping = {
        "validate_hed_string": HedStringValidationRequest,
        "validate_hed_file": HedFileValidationRequest,
        "generate_hed_sidecar": HedSidecarGenerationRequest,
        "search_hed_schema": SchemaSearchRequest,
        "get_server_info": ServerInfoRequest,
        "validate_spreadsheet": SpreadsheetValidationRequest,
        "batch_validate": BatchValidationRequest,
    }

    return model_mapping.get(tool_name)
