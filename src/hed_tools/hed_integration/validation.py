"""HED annotation validation and sidecar generation utilities.

This module provides comprehensive tools for validating HED annotations and generating
sidecar files from events data. It includes support for batch processing, BIDS datasets,
and various validation patterns.
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Iterator
import pandas as pd

# Import HED validation components (with graceful fallback)
try:
    from hed.errors import get_printable_issue_string, HedFileError
    from hed.validator import HedValidator
    from hed.models.tabular_input import TabularInput
    HED_AVAILABLE = True
except ImportError:
    HED_AVAILABLE = False
    # Stub classes for when HED is not available
    class HedValidator:
        def __init__(self, schema): pass
        def validate_string(self, hed_string: str): return []
    
    class TabularInput:
        def __init__(self, *args, **kwargs): pass
        def validate_file(self, schema): return []
    
    def get_printable_issue_string(issues): return str(issues)
    
    class HedFileError(Exception): pass

from .models import (
    ValidationResult, SidecarTemplate, OperationResult,
    HEDWrapperConfig, EventsData
)
from .schema import SchemaHandler
from .tabular_summary import TabularSummaryWrapper, create_tabular_summary_wrapper

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class HEDValidator:
    """HED annotation validation utilities.
    
    Provides methods for validating HED strings, events data, and sidecar files
    against HED schemas with comprehensive error reporting.
    """
    
    def __init__(self, schema_handler: Optional[SchemaHandler] = None):
        """Initialize the validator.
        
        Args:
            schema_handler: Optional schema handler for loading HED schemas
        """
        self.schema_handler = schema_handler or SchemaHandler()
        self.logger = logging.getLogger(__name__)
        self._validator_cache = {}
    
    def _get_validator(self, schema_version: Optional[str] = None) -> HedValidator:
        """Get or create a cached HED validator for the specified schema.
        
        Args:
            schema_version: HED schema version to use
            
        Returns:
            HED validator instance
        """
        if not HED_AVAILABLE:
            self.logger.warning("HED library not available - validation will be limited")
            return HedValidator(None)
        
        cache_key = schema_version or "default"
        if cache_key not in self._validator_cache:
            schema = self.schema_handler.get_schema(schema_version)
            self._validator_cache[cache_key] = HedValidator(schema)
        
        return self._validator_cache[cache_key]
    
    async def validate_string(self, hed_string: str, 
                             schema_version: Optional[str] = None) -> ValidationResult:
        """Validate a single HED string.
        
        Args:
            hed_string: HED annotation string to validate
            schema_version: Optional schema version to validate against
            
        Returns:
            ValidationResult with validation status and any issues
        """
        try:
            validator = self._get_validator(schema_version)
            
            # Run validation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                issues = await loop.run_in_executor(
                    executor, validator.validate_string, hed_string
                )
            
            is_valid = len(issues) == 0
            formatted_issues = get_printable_issue_string(issues) if issues else None
            
            return ValidationResult(
                is_valid=is_valid,
                errors=issues,
                warnings=[],
                summary=f"Validation {'passed' if is_valid else 'failed'} for HED string",
                formatted_errors=formatted_issues
            )
            
        except Exception as e:
            self.logger.error(f"Error validating HED string: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[{"message": str(e), "code": "VALIDATION_ERROR"}],
                warnings=[],
                summary="Validation failed due to internal error"
            )
    
    async def validate_events_data(self, events_data: Union[pd.DataFrame, Path, str],
                                  sidecar: Optional[Dict[str, Any]] = None,
                                  schema_version: Optional[str] = None) -> ValidationResult:
        """Validate events data with optional sidecar.
        
        Args:
            events_data: Events DataFrame, file path, or file content
            sidecar: Optional sidecar dictionary for HED annotations
            schema_version: Optional schema version to validate against
            
        Returns:
            ValidationResult with validation status and issues
        """
        try:
            # Prepare tabular input
            if isinstance(events_data, pd.DataFrame):
                # For DataFrames, we need to save temporarily for TabularInput
                temp_path = Path("/tmp/temp_events.tsv")
                events_data.to_csv(temp_path, sep='\t', index=False)
                file_path = temp_path
            else:
                file_path = Path(events_data)
            
            if not HED_AVAILABLE:
                self.logger.warning("HED library not available - using basic validation")
                return ValidationResult(
                    is_valid=True,
                    errors=[],
                    warnings=[{"message": "HED library not available - validation skipped"}],
                    summary="Validation skipped due to missing HED library"
                )
            
            # Create tabular input with sidecar
            tabular_input = TabularInput(str(file_path), sidecar=sidecar)
            schema = self.schema_handler.get_schema(schema_version)
            
            # Run validation in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                issues = await loop.run_in_executor(
                    executor, tabular_input.validate_file, schema
                )
            
            # Categorize issues
            errors = [issue for issue in issues if issue.get('severity') == 'error']
            warnings = [issue for issue in issues if issue.get('severity') == 'warning']
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                summary=f"Found {len(errors)} errors and {len(warnings)} warnings",
                formatted_errors=get_printable_issue_string(issues) if issues else None
            )
            
        except Exception as e:
            self.logger.error(f"Error validating events data: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[{"message": str(e), "code": "VALIDATION_ERROR"}],
                warnings=[],
                summary="Validation failed due to internal error"
            )
    
    async def validate_sidecar_file(self, sidecar_path: Union[Path, str],
                                   schema_version: Optional[str] = None) -> ValidationResult:
        """Validate a sidecar file.
        
        Args:
            sidecar_path: Path to the sidecar JSON file
            schema_version: Optional schema version to validate against
            
        Returns:
            ValidationResult with validation status and issues
        """
        try:
            sidecar_path = Path(sidecar_path)
            
            # Load sidecar
            with open(sidecar_path, 'r') as f:
                sidecar = json.load(f)
            
            # Validate each HED string in the sidecar
            all_issues = []
            
            for column, column_def in sidecar.items():
                if 'HED' in column_def:
                    hed_info = column_def['HED']
                    if isinstance(hed_info, dict):
                        # Value-specific HED annotations
                        for value, hed_string in hed_info.items():
                            if isinstance(hed_string, str):
                                result = await self.validate_string(hed_string, schema_version)
                                for issue in result.errors:
                                    issue['column'] = column
                                    issue['value'] = value
                                all_issues.extend(result.errors)
                    elif isinstance(hed_info, str):
                        # Column-level HED annotation
                        result = await self.validate_string(hed_info, schema_version)
                        for issue in result.errors:
                            issue['column'] = column
                        all_issues.extend(result.errors)
            
            is_valid = len(all_issues) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                errors=all_issues,
                warnings=[],
                summary=f"Sidecar validation {'passed' if is_valid else 'failed'} with {len(all_issues)} issues"
            )
            
        except Exception as e:
            self.logger.error(f"Error validating sidecar file: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[{"message": str(e), "code": "SIDECAR_ERROR"}],
                warnings=[],
                summary="Sidecar validation failed due to internal error"
            )


class SidecarGenerator:
    """HED sidecar generation utilities.
    
    Provides methods for generating HED sidecar templates from events data
    using TabularSummary integration and various customization options.
    """
    
    def __init__(self, schema_handler: Optional[SchemaHandler] = None,
                 tabular_summary_wrapper: Optional[TabularSummaryWrapper] = None):
        """Initialize the sidecar generator.
        
        Args:
            schema_handler: Optional schema handler for HED schemas
            tabular_summary_wrapper: Optional TabularSummary wrapper
        """
        self.schema_handler = schema_handler or SchemaHandler()
        self.tabular_summary_wrapper = tabular_summary_wrapper
        self.logger = logging.getLogger(__name__)
    
    async def generate_sidecar_template(self, events_data: Union[pd.DataFrame, Path, str],
                                       skip_columns: Optional[List[str]] = None,
                                       value_columns: Optional[List[str]] = None,
                                       schema_version: Optional[str] = None) -> SidecarTemplate:
        """Generate a HED sidecar template from events data.
        
        Args:
            events_data: Events DataFrame or file path
            skip_columns: Columns to skip in the template
            value_columns: Columns to include as value columns
            schema_version: HED schema version to use
            
        Returns:
            SidecarTemplate with generated HED annotations
        """
        try:
            # Load events data if it's a file path
            if isinstance(events_data, (str, Path)):
                events_df = pd.read_csv(events_data, sep='\t')
            else:
                events_df = events_data.copy()
            
            # Create or use existing TabularSummary wrapper
            if self.tabular_summary_wrapper is None:
                wrapper = await create_tabular_summary_wrapper(
                    data=events_df,
                    skip_columns=skip_columns or ['onset', 'duration']
                )
            else:
                wrapper = self.tabular_summary_wrapper
            
            # Generate sidecar template using TabularSummary
            template_result = await wrapper.extract_sidecar_template()
            
            if not template_result.success:
                raise ValidationError(f"Failed to generate template: {template_result.error}")
            
            template = template_result.result
            
            # Enhance template with schema-specific suggestions
            enhanced_template = await self._enhance_template_with_schema(
                template, schema_version
            )
            
            return SidecarTemplate(
                template=enhanced_template,
                schema_version=schema_version or self.schema_handler.current_version,
                generated_columns=list(enhanced_template.keys()),
                metadata={
                    "generation_time": time.time(),
                    "events_file_shape": events_df.shape,
                    "skip_columns": skip_columns or [],
                    "value_columns": value_columns or []
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating sidecar template: {e}")
            raise ValidationError(f"Sidecar generation failed: {e}")
    
    async def _enhance_template_with_schema(self, template: Dict[str, Any],
                                          schema_version: Optional[str] = None) -> Dict[str, Any]:
        """Enhance template with schema-specific HED tag suggestions.
        
        Args:
            template: Base template from TabularSummary
            schema_version: HED schema version for enhancements
            
        Returns:
            Enhanced template with better HED annotations
        """
        enhanced = template.copy()
        
        try:
            schema = self.schema_handler.get_schema(schema_version)
            
            for column, column_def in enhanced.items():
                if 'HED' in column_def and isinstance(column_def['HED'], dict):
                    # Enhance HED annotations for each value
                    for value, hed_string in column_def['HED'].items():
                        if not hed_string or hed_string == '#':
                            # Suggest HED tags based on column name and value
                            suggested_hed = await self._suggest_hed_annotation(
                                column, value, schema
                            )
                            if suggested_hed:
                                column_def['HED'][value] = suggested_hed
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"Could not enhance template with schema: {e}")
            return template
    
    async def _suggest_hed_annotation(self, column_name: str, value: str,
                                    schema) -> Optional[str]:
        """Suggest HED annotation for a column value.
        
        Args:
            column_name: Name of the column
            value: Value in the column
            schema: HED schema for tag suggestions
            
        Returns:
            Suggested HED annotation string
        """
        # Basic pattern-based suggestions
        column_lower = column_name.lower()
        value_lower = str(value).lower()
        
        suggestions = []
        
        # Common BIDS column patterns
        if 'trial_type' in column_lower or 'condition' in column_lower:
            suggestions.append(f"(Condition/{value}, Label/{value})")
        elif 'response' in column_lower:
            if value_lower in ['left', 'right']:
                suggestions.append(f"(Agent-action, (Movement, {value.title()}))")
            else:
                suggestions.append(f"(Agent-action, (Response, Label/{value}))")
        elif 'stimulus' in column_lower or 'stim' in column_lower:
            suggestions.append(f"(Sensory-event, (Visual-presentation, Label/{value}))")
        elif 'accuracy' in column_lower:
            if value in [0, 1, '0', '1']:
                acc_label = "Correct" if str(value) == "1" else "Incorrect"
                suggestions.append(f"(Performance, {acc_label})")
        elif 'rt' in column_lower or 'reaction_time' in column_lower:
            suggestions.append(f"(Response-time, Label/{value})")
        
        # Return first suggestion or placeholder
        if suggestions:
            return suggestions[0]
        else:
            return f"(Label/{value})"
    
    async def save_sidecar(self, template: SidecarTemplate, output_path: Union[Path, str],
                          format: str = 'json') -> OperationResult:
        """Save sidecar template to file.
        
        Args:
            template: SidecarTemplate to save
            output_path: Path to save the file
            format: Output format ('json' or 'yaml')
            
        Returns:
            OperationResult with save status
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(template.template, f, indent=2, ensure_ascii=False)
            elif format.lower() == 'yaml':
                import yaml
                with open(output_path, 'w') as f:
                    yaml.dump(template.template, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return OperationResult(
                success=True,
                result={"saved_to": str(output_path), "format": format},
                message=f"Sidecar saved successfully to {output_path}"
            )
            
        except Exception as e:
            self.logger.error(f"Error saving sidecar: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                message="Failed to save sidecar"
            )


class BatchValidator:
    """Batch validation utilities for processing multiple files.
    
    Provides efficient batch processing of HED validation tasks with
    progress tracking and error handling.
    """
    
    def __init__(self, validator: Optional[HEDValidator] = None,
                 max_workers: int = 4):
        """Initialize batch validator.
        
        Args:
            validator: Optional HED validator instance
            max_workers: Maximum number of concurrent workers
        """
        self.validator = validator or HEDValidator()
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
    
    async def validate_directory(self, directory: Path, 
                                pattern: str = "*.tsv",
                                sidecar_pattern: str = "*.json") -> Iterator[Dict[str, Any]]:
        """Validate all events files in a directory.
        
        Args:
            directory: Directory containing events files
            pattern: File pattern for events files
            sidecar_pattern: File pattern for sidecar files
            
        Yields:
            Validation results for each file
        """
        events_files = list(directory.glob(pattern))
        
        # Process files in chunks to manage memory
        chunk_size = min(self.max_workers, 10)
        
        for i in range(0, len(events_files), chunk_size):
            chunk = events_files[i:i + chunk_size]
            
            # Create validation tasks
            tasks = []
            for events_file in chunk:
                # Look for corresponding sidecar
                sidecar_file = events_file.with_suffix('.json')
                sidecar = None
                
                if sidecar_file.exists():
                    try:
                        with open(sidecar_file, 'r') as f:
                            sidecar = json.load(f)
                    except Exception as e:
                        self.logger.warning(f"Could not load sidecar {sidecar_file}: {e}")
                
                task = self._validate_single_file(events_file, sidecar)
                tasks.append(task)
            
            # Execute chunk concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Yield results
            for events_file, result in zip(chunk, results):
                if isinstance(result, Exception):
                    yield {
                        "file": str(events_file),
                        "validation_result": ValidationResult(
                            is_valid=False,
                            errors=[{"message": str(result)}],
                            warnings=[],
                            summary="Validation failed due to error"
                        ),
                        "error": str(result)
                    }
                else:
                    yield {
                        "file": str(events_file),
                        "validation_result": result,
                        "error": None
                    }
    
    async def _validate_single_file(self, events_file: Path, 
                                   sidecar: Optional[Dict]) -> ValidationResult:
        """Validate a single events file.
        
        Args:
            events_file: Path to events file
            sidecar: Optional sidecar dictionary
            
        Returns:
            ValidationResult for the file
        """
        return await self.validator.validate_events_data(events_file, sidecar)


class BIDSValidator:
    """BIDS-specific validation utilities.
    
    Provides specialized validation for BIDS datasets with support for
    dataset-level validation and BIDS specification compliance.
    """
    
    def __init__(self, validator: Optional[HEDValidator] = None):
        """Initialize BIDS validator.
        
        Args:
            validator: Optional HED validator instance
        """
        self.validator = validator or HEDValidator()
        self.batch_validator = BatchValidator(self.validator)
        self.logger = logging.getLogger(__name__)
    
    async def validate_bids_dataset(self, dataset_path: Path,
                                   validate_hed: bool = True) -> Dict[str, Any]:
        """Validate an entire BIDS dataset.
        
        Args:
            dataset_path: Path to BIDS dataset root
            validate_hed: Whether to validate HED annotations
            
        Returns:
            Comprehensive validation report
        """
        try:
            # Find all events files in the dataset
            events_files = list(dataset_path.rglob("*_events.tsv"))
            
            if not events_files:
                return {
                    "valid": True,
                    "summary": "No events files found in dataset",
                    "files_validated": 0,
                    "errors": [],
                    "warnings": []
                }
            
            # Validate BIDS structure
            structure_issues = await self._validate_bids_structure(dataset_path)
            
            # Validate HED annotations if requested
            hed_results = []
            if validate_hed:
                async for result in self.batch_validator.validate_directory(
                    dataset_path, "*_events.tsv"
                ):
                    hed_results.append(result)
            
            # Compile results
            all_errors = structure_issues.get('errors', [])
            all_warnings = structure_issues.get('warnings', [])
            
            for result in hed_results:
                if result['validation_result'].errors:
                    all_errors.extend(result['validation_result'].errors)
                if result['validation_result'].warnings:
                    all_warnings.extend(result['validation_result'].warnings)
            
            is_valid = len(all_errors) == 0
            
            return {
                "valid": is_valid,
                "summary": f"Validated {len(events_files)} files with {len(all_errors)} errors",
                "files_validated": len(events_files),
                "errors": all_errors,
                "warnings": all_warnings,
                "file_results": hed_results
            }
            
        except Exception as e:
            self.logger.error(f"Error validating BIDS dataset: {e}")
            return {
                "valid": False,
                "summary": f"Dataset validation failed: {e}",
                "files_validated": 0,
                "errors": [{"message": str(e), "code": "DATASET_ERROR"}],
                "warnings": []
            }
    
    async def _validate_bids_structure(self, dataset_path: Path) -> Dict[str, Any]:
        """Validate BIDS dataset structure.
        
        Args:
            dataset_path: Path to BIDS dataset root
            
        Returns:
            Structure validation results
        """
        issues = {"errors": [], "warnings": []}
        
        # Check for dataset_description.json
        desc_file = dataset_path / "dataset_description.json"
        if not desc_file.exists():
            issues["errors"].append({
                "message": "Missing dataset_description.json",
                "code": "BIDS_STRUCTURE"
            })
        
        # Check for proper events file naming
        events_files = list(dataset_path.rglob("*_events.tsv"))
        for events_file in events_files:
            if not events_file.stem.endswith("_events"):
                issues["warnings"].append({
                    "message": f"Events file {events_file} may not follow BIDS naming",
                    "code": "BIDS_NAMING"
                })
        
        return issues


# Factory functions for easy instantiation
async def create_hed_validator(schema_version: Optional[str] = None) -> HEDValidator:
    """Create a HED validator with optional schema version.
    
    Args:
        schema_version: Optional HED schema version
        
    Returns:
        Configured HEDValidator instance
    """
    schema_handler = SchemaHandler()
    if schema_version:
        await schema_handler.load_schema(schema_version)
    
    return HEDValidator(schema_handler)


async def create_sidecar_generator(schema_version: Optional[str] = None,
                                  **tabular_summary_config) -> SidecarGenerator:
    """Create a sidecar generator with optional configuration.
    
    Args:
        schema_version: Optional HED schema version
        **tabular_summary_config: Configuration for TabularSummary wrapper
        
    Returns:
        Configured SidecarGenerator instance
    """
    schema_handler = SchemaHandler()
    if schema_version:
        await schema_handler.load_schema(schema_version)
    
    # Create TabularSummary wrapper if config provided
    tabular_wrapper = None
    if tabular_summary_config:
        tabular_wrapper = await create_tabular_summary_wrapper(**tabular_summary_config)
    
    return SidecarGenerator(schema_handler, tabular_wrapper)


async def create_batch_validator(schema_version: Optional[str] = None,
                                max_workers: int = 4) -> BatchValidator:
    """Create a batch validator with optional configuration.
    
    Args:
        schema_version: Optional HED schema version
        max_workers: Maximum number of concurrent workers
        
    Returns:
        Configured BatchValidator instance
    """
    validator = await create_hed_validator(schema_version)
    return BatchValidator(validator, max_workers)


async def create_bids_validator(schema_version: Optional[str] = None) -> BIDSValidator:
    """Create a BIDS validator with optional schema version.
    
    Args:
        schema_version: Optional HED schema version
        
    Returns:
        Configured BIDSValidator instance
    """
    validator = await create_hed_validator(schema_version)
    return BIDSValidator(validator) 