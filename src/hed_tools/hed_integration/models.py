"""Data models using Pydantic for robust validation and configuration.

This module defines data models for configuration options, input/output data structures,
and API interfaces used throughout the HED integration system.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import pandas as pd


class SchemaConfig(BaseModel):
    """Configuration for HED schema handling."""
    
    version: str = Field(default="8.3.0", description="HED schema version to use")
    custom_path: Optional[Path] = Field(default=None, description="Path to custom schema file")
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".hed_cache", 
        description="Directory for caching schema files"
    )
    auto_update: bool = Field(default=True, description="Automatically update schema cache")
    fallback_versions: List[str] = Field(
        default_factory=lambda: ["8.3.0", "8.2.0", "8.1.0"],
        description="Fallback versions if primary fails"
    )
    
    @field_validator('version')
    @classmethod
    def validate_version(cls, v):
        """Validate schema version format."""
        if not v or not isinstance(v, str):
            raise ValueError("Schema version must be a non-empty string")
        return v
    
    @field_validator('cache_dir')
    @classmethod
    def validate_cache_dir(cls, v):
        """Ensure cache directory exists or can be created."""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v


class ValidationConfig(BaseModel):
    """Configuration for HED validation operations."""
    
    check_for_warnings: bool = Field(default=True, description="Include warnings in validation")
    check_syntax: bool = Field(default=True, description="Check HED syntax")
    check_required_tags: bool = Field(default=True, description="Check for required tags")
    max_errors: int = Field(default=100, description="Maximum errors to report")
    timeout_seconds: int = Field(default=60, description="Validation timeout")
    
    @field_validator('max_errors')
    @classmethod
    def validate_max_errors(cls, v):
        """Ensure max_errors is positive."""
        if v <= 0:
            raise ValueError("max_errors must be positive")
        return v


class TabularSummaryConfig(BaseModel):
    """Configuration for TabularSummary operations."""
    
    skip_columns: List[str] = Field(
        default_factory=lambda: ["onset", "duration", "sample"],
        description="Columns to skip in analysis"
    )
    value_columns: Optional[List[str]] = Field(
        default=None, 
        description="Specific columns to treat as value columns"
    )
    name: str = Field(default="", description="Name for the summary")
    include_description: bool = Field(default=True, description="Include descriptions in output")
    max_unique_values: int = Field(default=50, description="Maximum unique values to include")
    
    @field_validator('max_unique_values')
    @classmethod
    def validate_max_unique_values(cls, v):
        """Ensure max_unique_values is reasonable."""
        if v <= 0 or v > 1000:
            raise ValueError("max_unique_values must be between 1 and 1000")
        return v


class HEDWrapperConfig(BaseModel):
    """Master configuration for HED wrapper operations."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    hed_schema: SchemaConfig = Field(default_factory=SchemaConfig, alias='schema')
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    tabular_summary: TabularSummaryConfig = Field(default_factory=TabularSummaryConfig)
    async_timeout: float = Field(default=30.0, description="Default timeout for async operations")
    debug_mode: bool = Field(default=False, description="Enable debug logging")


class ColumnInfo(BaseModel):
    """Information about a specific column in event data."""
    
    name: str = Field(description="Column name")
    data_type: str = Field(description="Detected data type")
    unique_count: int = Field(description="Number of unique values")
    null_count: int = Field(description="Number of null values")
    sample_values: List[Any] = Field(description="Sample values from the column")
    is_categorical: bool = Field(description="Whether column is categorical")
    is_numeric: bool = Field(description="Whether column is numeric")
    is_temporal: bool = Field(description="Whether column is temporal")
    suggested_hed_category: Optional[str] = Field(
        default=None, 
        description="Suggested HED category"
    )
    
    @field_validator('unique_count', 'null_count')
    @classmethod
    def validate_counts(cls, v):
        """Ensure counts are non-negative."""
        if v < 0:
            raise ValueError("Counts must be non-negative")
        return v


class EventsData(BaseModel):
    """Model for BIDS events data."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    file_path: Optional[Path] = Field(default=None, description="Path to events file")
    dataframe: Optional[pd.DataFrame] = Field(default=None, description="Events DataFrame")
    columns: List[ColumnInfo] = Field(default_factory=list, description="Column information")
    row_count: int = Field(description="Number of rows in the data")
    required_columns_present: bool = Field(description="Whether onset/duration are present")
    
    @model_validator(mode='after')
    def validate_data_source(self):
        """Ensure either file_path or dataframe is provided."""
        if self.file_path is None and self.dataframe is None:
            raise ValueError("Either file_path or dataframe must be provided")
        return self
    
    @field_validator('row_count')
    @classmethod
    def validate_row_count(cls, v):
        """Ensure row count is non-negative."""
        if v < 0:
            raise ValueError("Row count must be non-negative")
        return v


class ValidationResult(BaseModel):
    """Result of HED validation operation."""
    
    valid: bool = Field(description="Whether validation passed")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Validation errors")
    warnings: List[Dict[str, Any]] = Field(default_factory=list, description="Validation warnings")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Validation statistics")
    processing_time: float = Field(description="Time taken for validation in seconds")
    schema_version: str = Field(description="HED schema version used")
    
    @field_validator('processing_time')
    @classmethod
    def validate_processing_time(cls, v):
        """Ensure processing time is non-negative."""
        if v < 0:
            raise ValueError("Processing time must be non-negative")
        return v


class SidecarTemplate(BaseModel):
    """Model for HED sidecar template structure."""
    
    template: Dict[str, Any] = Field(description="The sidecar template dictionary")
    generated_columns: List[str] = Field(description="Columns included in template")
    schema_version: str = Field(description="HED schema version used")
    generation_time: float = Field(description="Time taken to generate template")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('generation_time')
    @classmethod
    def validate_generation_time(cls, v):
        """Ensure generation time is non-negative."""
        if v < 0:
            raise ValueError("Generation time must be non-negative")
        return v
    
    @field_validator('template')
    @classmethod
    def validate_template_structure(cls, v):
        """Basic validation of template structure."""
        if not isinstance(v, dict):
            raise ValueError("Template must be a dictionary")
        return v


class SchemaInfo(BaseModel):
    """Information about a HED schema."""
    
    version: str = Field(description="Schema version")
    loaded: bool = Field(description="Whether schema is loaded")
    path: Optional[Path] = Field(default=None, description="Path to schema file")
    tag_count: int = Field(default=0, description="Number of tags in schema")
    description: str = Field(default="", description="Schema description")
    library_schemas: List[str] = Field(
        default_factory=list, 
        description="Associated library schemas"
    )
    
    @field_validator('tag_count')
    @classmethod
    def validate_tag_count(cls, v):
        """Ensure tag count is non-negative."""
        if v < 0:
            raise ValueError("Tag count must be non-negative")
        return v


class OperationResult(BaseModel):
    """Generic result model for wrapper operations."""
    
    success: bool = Field(description="Whether operation succeeded")
    data: Optional[Any] = Field(default=None, description="Operation result data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    processing_time: float = Field(description="Time taken for operation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('processing_time')
    @classmethod
    def validate_processing_time(cls, v):
        """Ensure processing time is non-negative."""
        if v < 0:
            raise ValueError("Processing time must be non-negative")
        return v
    
    @model_validator(mode='after')
    def validate_result_consistency(self):
        """Ensure success/error consistency."""
        if self.success and self.error:
            raise ValueError("Cannot have both success=True and an error message")
        if not self.success and not self.error:
            raise ValueError("Failed operations must include an error message")
        return self


class BatchProcessingConfig(BaseModel):
    """Configuration for batch processing operations."""
    
    max_workers: int = Field(default=4, description="Maximum parallel workers")
    chunk_size: int = Field(default=1000, description="Chunk size for large datasets")
    progress_reporting: bool = Field(default=True, description="Enable progress reporting")
    continue_on_error: bool = Field(default=True, description="Continue processing if one item fails")
    timeout_per_item: float = Field(default=60.0, description="Timeout per item in batch")
    
    @field_validator('max_workers')
    @classmethod
    def validate_max_workers(cls, v):
        """Ensure reasonable number of workers."""
        if v <= 0 or v > 32:
            raise ValueError("max_workers must be between 1 and 32")
        return v
    
    @field_validator('chunk_size')
    @classmethod
    def validate_chunk_size(cls, v):
        """Ensure reasonable chunk size."""
        if v <= 0 or v > 100000:
            raise ValueError("chunk_size must be between 1 and 100000")
        return v 