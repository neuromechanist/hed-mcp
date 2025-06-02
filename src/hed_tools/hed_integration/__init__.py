"""HED-specific integration components.

This module provides direct integration with the HED Python tools library,
including wrappers and utilities for HED schema handling and validation.
"""

from .hed_wrapper import HEDWrapper, create_hed_wrapper, HEDIntegration
from .schema import SchemaHandler, SchemaManagerFacade
from .tabular_summary import (
    TabularSummaryWrapper, create_tabular_summary_wrapper,
    DataValidator, MemoryManager, CacheManager, 
    ValidationError, DataFormat, PerformanceMetrics
)
from .models import (
    # Configuration models
    HEDWrapperConfig, SchemaConfig, ValidationConfig, TabularSummaryConfig,
    BatchProcessingConfig,
    
    # Data models
    EventsData, ColumnInfo, ValidationResult, SidecarTemplate, 
    SchemaInfo, OperationResult,
)

__all__ = [
    # Main wrapper classes
    'HEDWrapper',
    'HEDIntegration', 
    'create_hed_wrapper',
    
    # Schema handling
    'SchemaHandler',
    'SchemaManagerFacade',
    
    # TabularSummary integration
    'TabularSummaryWrapper',
    'create_tabular_summary_wrapper',
    'DataValidator',
    'MemoryManager',
    'CacheManager',
    'ValidationError',
    'DataFormat',
    'PerformanceMetrics',
    
    # Configuration models
    'HEDWrapperConfig',
    'SchemaConfig',
    'ValidationConfig', 
    'TabularSummaryConfig',
    'BatchProcessingConfig',
    
    # Data models
    'EventsData',
    'ColumnInfo',
    'ValidationResult',
    'SidecarTemplate',
    'SchemaInfo',
    'OperationResult',
] 