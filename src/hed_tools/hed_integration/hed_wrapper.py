"""Main interface wrapper for HED library operations.

This module provides a simplified interface to the HED (Hierarchical Event Descriptor)
Python tools library, focusing on schema loading, validation, and TabularSummary operations.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
import pandas as pd

try:
    from hed.tools.analysis.tabular_summary import TabularSummary
    from hed.models import HedString, HedTag
    from hed.schema import HedSchema
    from hed.validator import HedValidator
    from hed.errors.exceptions import HedFileError
except ImportError:
    # Graceful fallback if hed is not available yet
    TabularSummary = None
    HedString = None
    HedTag = None
    HedSchema = None
    HedValidator = None
    HedFileError = Exception

from .models import (
    HEDWrapperConfig, ValidationResult, SidecarTemplate, OperationResult,
    EventsData, ColumnInfo, SchemaInfo
)
from .schema import SchemaHandler, SchemaManagerFacade

logger = logging.getLogger(__name__)


class HEDWrapper:
    """Wrapper class for HED library functionality.
    
    This class provides a simplified interface to common HED operations including:
    - Schema loading and management
    - Event validation and processing
    - TabularSummary operations for sidecar generation
    - HED string parsing and manipulation
    """
    
    def __init__(self, config: Optional[HEDWrapperConfig] = None):
        """Initialize the HED wrapper.
        
        Args:
            config: Configuration object for wrapper behavior
        """
        self.config = config or HEDWrapperConfig()
        self.schema_handler = SchemaHandler(self.config.schema)
        self._initialized = False
        
        if TabularSummary is None:
            logger.warning("HED library not available - wrapper will run in stub mode")
    
    async def initialize(self) -> OperationResult:
        """Initialize the wrapper by loading the default schema.
        
        Returns:
            OperationResult indicating initialization success
        """
        if self._initialized:
            return OperationResult(success=True, processing_time=0.0)
        
        start_time = time.time()
        result = await self.schema_handler.load_schema()
        
        if result.success:
            self._initialized = True
            logger.info("HED wrapper initialized successfully")
        else:
            logger.error(f"HED wrapper initialization failed: {result.error}")
        
        return OperationResult(
            success=result.success,
            data=result.data,
            error=result.error,
            processing_time=time.time() - start_time
        )
    
    async def load_schema(self, version: Optional[str] = None, 
                         custom_path: Optional[Path] = None) -> OperationResult:
        """Load HED schema asynchronously.
        
        Args:
            version: HED schema version to load (overrides config setting)
            custom_path: Custom schema file path (overrides config setting)
            
        Returns:
            OperationResult with schema loading details
        """
        return await self.schema_handler.load_schema(version, custom_path)
    
    def _analyze_dataframe_columns(self, df: pd.DataFrame) -> List[ColumnInfo]:
        """Analyze DataFrame columns to extract metadata.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of ColumnInfo objects with column metadata
        """
        columns_info = []
        
        for col in df.columns:
            col_data = df[col]
            
            # Basic statistics
            unique_count = col_data.nunique()
            null_count = col_data.isnull().sum()
            sample_values = col_data.dropna().unique()[:5].tolist()
            
            # Data type analysis
            data_type = str(col_data.dtype)
            is_numeric = pd.api.types.is_numeric_dtype(col_data)
            is_categorical = unique_count < len(df) * 0.1 and unique_count < 50
            is_temporal = col in ['onset', 'duration'] or 'time' in col.lower()
            
            # Suggest HED category based on column analysis
            suggested_hed = self._suggest_hed_category(col, col_data, is_categorical, is_numeric)
            
            columns_info.append(ColumnInfo(
                name=col,
                data_type=data_type,
                unique_count=unique_count,
                null_count=null_count,
                sample_values=sample_values,
                is_categorical=is_categorical,
                is_numeric=is_numeric,
                is_temporal=is_temporal,
                suggested_hed_category=suggested_hed
            ))
        
        return columns_info
    
    def _suggest_hed_category(self, column_name: str, data: pd.Series, 
                            is_categorical: bool, is_numeric: bool) -> Optional[str]:
        """Suggest HED category based on column characteristics.
        
        Args:
            column_name: Name of the column
            data: Column data
            is_categorical: Whether column is categorical
            is_numeric: Whether column is numeric
            
        Returns:
            Suggested HED category or None
        """
        col_lower = column_name.lower()
        
        # Standard BIDS temporal columns
        if col_lower in ['onset', 'duration']:
            return None  # These don't need HED annotations
        
        # Response-related columns
        if any(term in col_lower for term in ['response', 'button', 'key', 'reaction']):
            return "Agent-action/Response"
        
        # Stimulus-related columns
        if any(term in col_lower for term in ['stimulus', 'image', 'sound', 'visual', 'auditory']):
            return "Sensory-event"
        
        # Task-related columns
        if any(term in col_lower for term in ['task', 'condition', 'trial', 'block']):
            return "Experiment-control"
        
        # Event types
        if col_lower in ['event_type', 'type', 'category']:
            return "Event"
        
        # Numeric measures
        if is_numeric and not is_categorical:
            return "Data-feature"
        
        # Categorical data
        if is_categorical:
            return "Experiment-control"
        
        return "Event/Category"  # Default fallback
    
    async def validate_events(self, events_data: Union[pd.DataFrame, Path, Dict[str, Any]], 
                            sidecar: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate HED annotations in events data.
        
        Args:
            events_data: BIDS events DataFrame, file path, or dictionary
            sidecar: Optional HED sidecar with column definitions
            
        Returns:
            ValidationResult with errors, warnings, and statistics
        """
        start_time = time.time()
        
        if not self._initialized:
            await self.initialize()
        
        if TabularSummary is None or not self.schema_handler.is_schema_loaded():
            return ValidationResult(
                valid=False,
                errors=[{"message": "HED library or schema not available", "code": "LIBRARY_ERROR"}],
                warnings=[],
                statistics={},
                processing_time=time.time() - start_time,
                schema_version="unknown"
            )
        
        try:
            # Convert input to DataFrame if needed
            if isinstance(events_data, Path):
                df = pd.read_csv(events_data, sep='\t')
            elif isinstance(events_data, dict):
                df = pd.DataFrame(events_data)
            else:
                df = events_data.copy()
            
            logger.info(f"Validating HED annotations for {len(df)} events")
            
            # For now, implement basic validation
            # TODO: Integrate with hed.validator.HedValidator when implementing actual validation
            errors = []
            warnings = []
            
            # Check for required columns
            if 'onset' not in df.columns:
                errors.append({
                    "message": "Required 'onset' column missing",
                    "code": "MISSING_ONSET",
                    "severity": "error"
                })
            
            # Check for HED annotations in sidecar or events
            hed_columns = []
            if sidecar:
                for col, meta in sidecar.items():
                    if isinstance(meta, dict) and 'HED' in meta:
                        hed_columns.append(col)
            
            # Basic validation statistics
            validation_stats = {
                "total_events": len(df),
                "validated_columns": hed_columns,
                "hed_tags_found": len(hed_columns),
                "columns_analyzed": list(df.columns),
                "has_sidecar": sidecar is not None
            }
            
            is_valid = len(errors) == 0
            schema_info = self.schema_handler.get_schema_info()
            
            return ValidationResult(
                valid=is_valid,
                errors=errors,
                warnings=warnings,
                statistics=validation_stats,
                processing_time=time.time() - start_time,
                schema_version=schema_info.version
            )
            
        except Exception as e:
            logger.error(f"HED validation failed: {e}")
            return ValidationResult(
                valid=False,
                errors=[{"message": str(e), "code": "VALIDATION_ERROR", "severity": "error"}],
                warnings=[],
                statistics={},
                processing_time=time.time() - start_time,
                schema_version="unknown"
            )
    
    async def generate_sidecar_template(self, 
                                      events_data: Union[pd.DataFrame, Path],
                                      skip_columns: Optional[List[str]] = None,
                                      value_columns: Optional[List[str]] = None) -> SidecarTemplate:
        """Generate HED sidecar template using TabularSummary.
        
        Args:
            events_data: BIDS events DataFrame or file path
            skip_columns: Columns to skip in analysis
            value_columns: Specific columns to treat as value columns
            
        Returns:
            SidecarTemplate with generated HED template
        """
        start_time = time.time()
        
        if not self._initialized:
            await self.initialize()
        
        if TabularSummary is None:
            logger.error("HED TabularSummary not available")
            return SidecarTemplate(
                template={"error": "HED library not available"},
                generated_columns=[],
                schema_version="unknown",
                generation_time=time.time() - start_time
            )
        
        try:
            # Load data if path provided
            if isinstance(events_data, Path):
                df = pd.read_csv(events_data, sep='\t')
                logger.info(f"Loaded events data from {events_data}: {len(df)} rows")
            else:
                df = events_data.copy()
            
            # Use config defaults if not specified
            skip_cols = skip_columns or self.config.tabular_summary.skip_columns
            value_cols = value_columns or self.config.tabular_summary.value_columns
            
            logger.info(f"Generating HED sidecar template using TabularSummary")
            logger.info(f"Skip columns: {skip_cols}")
            logger.info(f"Value columns: {value_cols}")
            
            # Create TabularSummary
            # Based on research: TabularSummary(value_cols=None, skip_cols=None, name='')
            summary = TabularSummary(
                value_cols=value_cols,
                skip_cols=skip_cols,
                name=self.config.tabular_summary.name
            )
            
            # Process the data
            summary.update(df)
            
            # Extract sidecar template
            sidecar_template = summary.extract_sidecar_template()
            
            # Enhance template with HED structure
            enhanced_template = self._enhance_sidecar_template(sidecar_template, df)
            
            schema_info = self.schema_handler.get_schema_info()
            generated_columns = list(enhanced_template.keys())
            
            logger.info(f"Generated sidecar template for {len(generated_columns)} columns")
            
            return SidecarTemplate(
                template=enhanced_template,
                generated_columns=generated_columns,
                schema_version=schema_info.version,
                generation_time=time.time() - start_time,
                metadata={
                    "source_rows": len(df),
                    "source_columns": list(df.columns),
                    "skip_columns": skip_cols,
                    "value_columns": value_cols
                }
            )
            
        except Exception as e:
            logger.error(f"Sidecar generation failed: {e}")
            return SidecarTemplate(
                template={"error": str(e)},
                generated_columns=[],
                schema_version="unknown",
                generation_time=time.time() - start_time
            )
    
    def _enhance_sidecar_template(self, base_template: Dict[str, Any], 
                                df: pd.DataFrame) -> Dict[str, Any]:
        """Enhance the basic sidecar template with HED-specific information.
        
        Args:
            base_template: Base template from TabularSummary
            df: Source DataFrame for additional analysis
            
        Returns:
            Enhanced template with HED annotations
        """
        enhanced = base_template.copy()
        columns_info = self._analyze_dataframe_columns(df)
        
        for col_info in columns_info:
            col_name = col_info.name
            
            # Skip standard BIDS columns that already have proper definitions
            if col_name in ['onset', 'duration', 'sample']:
                continue
            
            # Enhance or create column entry
            if col_name not in enhanced:
                enhanced[col_name] = {}
            
            col_entry = enhanced[col_name]
            
            # Add description if missing
            if 'Description' not in col_entry:
                col_entry['Description'] = f"Description for {col_name}"
            
            # Add HED annotation structure based on column analysis
            if col_info.suggested_hed_category and 'HED' not in col_entry:
                if col_info.is_categorical:
                    # Create value-based HED mapping for categorical columns
                    hed_mapping = {}
                    for value in col_info.sample_values:
                        hed_mapping[str(value)] = f"{col_info.suggested_hed_category}/Label/{value}"
                    col_entry['HED'] = hed_mapping
                else:
                    # For non-categorical columns, provide a template
                    col_entry['HED'] = f"{col_info.suggested_hed_category}/# "
            
            # Add additional metadata
            col_entry['_metadata'] = {
                "data_type": col_info.data_type,
                "unique_values": col_info.unique_count,
                "is_categorical": col_info.is_categorical,
                "sample_values": col_info.sample_values[:3]  # Limit sample values
            }
        
        return enhanced
    
    def get_available_schemas(self) -> List[Dict[str, str]]:
        """Get list of available HED schema versions.
        
        Returns:
            List of schema information dictionaries
        """
        return self.schema_handler.get_available_schemas()
    
    def parse_hed_string(self, hed_string: str) -> Dict[str, Any]:
        """Parse a HED string and return structured information.
        
        Args:
            hed_string: HED annotation string to parse
            
        Returns:
            Parsed HED information including tags, structure, and validation status
        """
        if HedString is None:
            return {"error": "HED library not available"}
        
        try:
            logger.info(f"Parsing HED string: {hed_string[:50]}...")
            
            # Create HedString object
            hed_obj = HedString(hed_string)
            
            # Extract information
            result = {
                "original": hed_string,
                "valid": True,  # Basic assumption, would need validation
                "tags": [],
                "structure": {},
                "errors": []
            }
            
            # TODO: Implement actual parsing and validation
            # This would use the schema for validation
            
            return result
            
        except Exception as e:
            logger.error(f"HED string parsing failed: {e}")
            return {
                "original": hed_string,
                "valid": False,
                "tags": [],
                "structure": {},
                "errors": [str(e)]
            }
    
    def get_schema_info(self) -> SchemaInfo:
        """Get information about the currently loaded schema.
        
        Returns:
            Schema metadata and statistics
        """
        return self.schema_handler.get_schema_info()
    
    async def analyze_events_data(self, events_data: Union[pd.DataFrame, Path]) -> EventsData:
        """Analyze events data and return structured information.
        
        Args:
            events_data: Events DataFrame or file path
            
        Returns:
            EventsData object with analysis results
        """
        try:
            # Load data if path provided
            if isinstance(events_data, Path):
                df = pd.read_csv(events_data, sep='\t')
                file_path = events_data
            else:
                df = events_data.copy()
                file_path = None
            
            # Analyze columns
            columns_info = self._analyze_dataframe_columns(df)
            
            # Check for required BIDS columns
            required_columns_present = all(col in df.columns for col in ['onset'])
            
            return EventsData(
                file_path=file_path,
                dataframe=df,
                columns=columns_info,
                row_count=len(df),
                required_columns_present=required_columns_present
            )
            
        except Exception as e:
            logger.error(f"Events data analysis failed: {e}")
            raise
    
    def close(self):
        """Clean up resources."""
        if hasattr(self.schema_handler, 'close'):
            self.schema_handler.close()
        logger.info("HED wrapper closed")


def create_hed_wrapper(schema_version: str = "8.3.0", 
                      config: Optional[HEDWrapperConfig] = None) -> HEDWrapper:
    """Factory function to create and initialize a HED wrapper.
    
    Args:
        schema_version: HED schema version to load
        config: Optional configuration object
        
    Returns:
        Initialized HEDWrapper instance
    """
    if config is None:
        config = HEDWrapperConfig()
        config.schema.version = schema_version
    
    wrapper = HEDWrapper(config)
    return wrapper


# Simplified facade for common operations
class HEDIntegration:
    """Simplified facade for common HED integration operations."""
    
    def __init__(self, schema_version: str = "8.3.0"):
        """Initialize with default configuration.
        
        Args:
            schema_version: HED schema version to use
        """
        self.wrapper = create_hed_wrapper(schema_version)
        self._initialized = False
    
    async def generate_sidecar(self, events_file: Path) -> Dict[str, Any]:
        """Generate a HED sidecar for an events file.
        
        Args:
            events_file: Path to BIDS events file
            
        Returns:
            Generated sidecar dictionary
        """
        if not self._initialized:
            await self.wrapper.initialize()
            self._initialized = True
        
        result = await self.wrapper.generate_sidecar_template(events_file)
        return result.template
    
    async def validate_events_file(self, events_file: Path, 
                                 sidecar_file: Optional[Path] = None) -> Dict[str, Any]:
        """Validate a BIDS events file with optional sidecar.
        
        Args:
            events_file: Path to BIDS events file
            sidecar_file: Optional path to HED sidecar file
            
        Returns:
            Validation results dictionary
        """
        if not self._initialized:
            await self.wrapper.initialize()
            self._initialized = True
        
        sidecar = None
        if sidecar_file and sidecar_file.exists():
            import json
            with open(sidecar_file, 'r') as f:
                sidecar = json.load(f)
        
        result = await self.wrapper.validate_events(events_file, sidecar)
        
        return {
            "valid": result.valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "statistics": result.statistics
        } 