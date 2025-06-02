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
from .schema import SchemaHandler
from .tabular_summary import create_tabular_summary_wrapper

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
        self.schema_handler = SchemaHandler(self.config.hed_schema)
        self._initialized = False
        
        # Initialize TabularSummary wrapper
        self._tabular_summary_wrapper = None
        
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
            # Initialize TabularSummary wrapper with schema handler
            self._tabular_summary_wrapper = create_tabular_summary_wrapper(
                config=self.config.tabular_summary,
                schema_handler=self.schema_handler
            )
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
        result = await self.schema_handler.load_schema(version, custom_path)
        
        # Update TabularSummary wrapper if schema loaded successfully
        if result.success and self._tabular_summary_wrapper:
            self._tabular_summary_wrapper.schema_handler = self.schema_handler
        
        return result
    
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
        
        # Implementation placeholder - would integrate with HED validation
        return ValidationResult(
            valid=True,
            errors=[],
            warnings=[],
            statistics={},
            processing_time=time.time() - start_time,
            schema_version="unknown"
        )
    
    async def generate_sidecar_template(self, 
                                      events_data: Union[pd.DataFrame, Path],
                                      skip_columns: Optional[List[str]] = None,
                                      value_columns: Optional[List[str]] = None,
                                      use_cache: bool = True) -> SidecarTemplate:
        """Generate HED sidecar template using enhanced TabularSummary integration.
        
        Args:
            events_data: BIDS events DataFrame or file path
            skip_columns: Columns to skip in analysis
            value_columns: Specific columns to treat as value columns
            use_cache: Whether to use caching for performance
            
        Returns:
            SidecarTemplate with generated HED template
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._tabular_summary_wrapper:
            logger.error("TabularSummary wrapper not available")
            return SidecarTemplate(
                template={"error": "TabularSummary wrapper not initialized"},
                generated_columns=[],
                schema_version="unknown",
                generation_time=0.0
            )
        
        try:
            # Use the enhanced TabularSummaryWrapper
            template = await self._tabular_summary_wrapper.extract_sidecar_template(
                data=events_data,
                skip_columns=skip_columns,
                use_cache=use_cache
            )
            
            # Enhance template with HED-specific information if available
            if isinstance(events_data, pd.DataFrame):
                df = events_data
            else:
                df = await self._tabular_summary_wrapper.load_data(events_data)
            
            enhanced_template = self._enhance_sidecar_template(template.template, df)
            
            return SidecarTemplate(
                template=enhanced_template,
                generated_columns=template.generated_columns,
                schema_version=template.schema_version,
                generation_time=template.generation_time,
                metadata={
                    **template.metadata,
                    "enhanced": True,
                    "wrapper_version": "TabularSummaryWrapper"
                }
            )
            
        except Exception as e:
            logger.error(f"Enhanced sidecar generation failed: {e}")
            return SidecarTemplate(
                template={"error": str(e)},
                generated_columns=[],
                schema_version="unknown",
                generation_time=0.0
            )
    
    async def generate_summary(self,
                             events_data: Union[pd.DataFrame, Path],
                             skip_columns: Optional[List[str]] = None,
                             value_columns: Optional[List[str]] = None,
                             use_cache: bool = True) -> OperationResult:
        """Generate comprehensive tabular summary using enhanced wrapper.
        
        Args:
            events_data: BIDS events DataFrame or file path
            skip_columns: Columns to skip in analysis
            value_columns: Specific columns to treat as value columns
            use_cache: Whether to use caching for performance
            
        Returns:
            OperationResult with comprehensive summary data
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._tabular_summary_wrapper:
            return OperationResult(
                success=False,
                error="TabularSummary wrapper not initialized",
                processing_time=0.0
            )
        
        return await self._tabular_summary_wrapper.generate_summary(
            data=events_data,
            skip_columns=skip_columns,
            value_columns=value_columns,
            use_cache=use_cache
        )
    
    async def process_batch_files(self,
                                file_paths: List[Union[str, Path]], 
                                chunk_size: int = 10,
                                continue_on_error: bool = True) -> List[Dict[str, Any]]:
        """Process multiple event files in batches using enhanced wrapper.
        
        Args:
            file_paths: List of file paths to process
            chunk_size: Number of files to process simultaneously
            continue_on_error: Whether to continue if one file fails
            
        Returns:
            List of processing results for each file
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._tabular_summary_wrapper:
            return [{"error": "TabularSummary wrapper not initialized"}]
        
        results = []
        async for result in self._tabular_summary_wrapper.process_batch(
            file_paths, chunk_size, continue_on_error
        ):
            results.append(result)
        
        return results

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
            
            # Create HedString object (for future use)
            HedString(hed_string)
            
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
            SchemaInfo object with schema details
        """
        return self.schema_handler.get_schema_info()
    
    async def analyze_events_data(self, events_data: Union[pd.DataFrame, Path]) -> EventsData:
        """Analyze events data and return structured information.
        
        Args:
            events_data: Events DataFrame or file path
            
        Returns:
            EventsData object with analysis results
        """
        if isinstance(events_data, Path):
            df = pd.read_csv(events_data, sep='\t')
            file_path = events_data
        else:
            df = events_data.copy()
            file_path = None
        
        columns_info = self._analyze_dataframe_columns(df)
        required_columns_present = all(col in df.columns for col in ['onset', 'duration'])
        
        return EventsData(
            file_path=file_path,
            dataframe=df,
            columns=columns_info,
            row_count=len(df),
            required_columns_present=required_columns_present
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the wrapper and its components.
        
        Returns:
            Dictionary with performance statistics
        """
        metrics = {}
        
        if self._tabular_summary_wrapper:
            metrics.update(self._tabular_summary_wrapper.get_performance_metrics())
        
        metrics.update({
            "initialized": self._initialized,
            "schema_loaded": self.schema_handler.get_schema_info().loaded,
        })
        
        return metrics
    
    async def close(self):
        """Close the wrapper and clean up resources."""
        if self._tabular_summary_wrapper:
            await self._tabular_summary_wrapper.close()
        
        self.schema_handler = None
        self._initialized = False
        logger.info("HED wrapper closed")


def create_hed_wrapper(schema_version: str = "8.3.0", 
                      config: Optional[HEDWrapperConfig] = None) -> HEDWrapper:
    """Create a HED wrapper instance with specified configuration.
    
    Args:
        schema_version: HED schema version to use
        config: Optional configuration object
        
    Returns:
        Configured HEDWrapper instance
    """
    if config is None:
        from .models import SchemaConfig
        config = HEDWrapperConfig(hed_schema=SchemaConfig(version=schema_version))
    else:
        # Update schema version if provided
        config.hed_schema.version = schema_version
    
    return HEDWrapper(config=config)


class HEDIntegration:
    """Simplified interface for common HED operations."""
    
    def __init__(self, schema_version: str = "8.3.0"):
        """Initialize HED integration with specified schema version."""
        self.wrapper = create_hed_wrapper(schema_version)
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure wrapper is initialized."""
        if not self._initialized:
            await self.wrapper.initialize()
            self._initialized = True
    
    async def generate_sidecar(self, events_file: Path) -> Dict[str, Any]:
        """Generate HED sidecar for events file.
        
        Args:
            events_file: Path to BIDS events file
            
        Returns:
            Generated sidecar dictionary
        """
        await self._ensure_initialized()
        
        template = await self.wrapper.generate_sidecar_template(events_file)
        return {
            "template": template.template,
            "metadata": {
                "generated_columns": template.generated_columns,
                "schema_version": template.schema_version,
                "generation_time": template.generation_time
            }
        }
    
    async def validate_events_file(self, events_file: Path, 
                                 sidecar_file: Optional[Path] = None) -> Dict[str, Any]:
        """Validate events file with optional sidecar.
        
        Args:
            events_file: Path to BIDS events file
            sidecar_file: Optional path to sidecar file
            
        Returns:
            Validation results dictionary
        """
        await self._ensure_initialized()
        
        sidecar = None
        if sidecar_file:
            import json
            with open(sidecar_file, 'r') as f:
                sidecar = json.load(f)
        
        result = await self.wrapper.validate_events(events_file, sidecar)
        return {
            "valid": result.valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "statistics": result.statistics,
            "processing_time": result.processing_time,
            "schema_version": result.schema_version
        } 