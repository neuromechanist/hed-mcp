"""Tools and utilities for HED integration.

This module contains various tools for analyzing and processing HED data.
"""

from .bids_parser import (
    BIDSEventParser,
    BIDSValidationError,
    parse_bids_events,
)
from .enhanced_column_analyzer import (
    EnhancedColumnAnalyzer,
    analyze_columns_enhanced,
)
from .llm_preprocessor import (
    LLMPreprocessor,
    SamplingConfig,
    ColumnSample,
    ColumnClassification,
    create_llm_preprocessor,
    process_for_llm_classification,
)
from .column_analysis_engine import (
    BIDSColumnAnalysisEngine,
    AnalysisConfig,
    FileAnalysisResult,
    BatchAnalysisResult,
    create_analysis_engine,
    analyze_bids_files,
    analyze_bids_directory,
)
from .cli import cli_main

__all__ = [
    # Parsing
    "BIDSEventParser",
    "BIDSValidationError",
    "parse_bids_events",
    # Enhanced Analysis
    "EnhancedColumnAnalyzer",
    "analyze_columns_enhanced",
    # LLM Preprocessor
    "LLMPreprocessor",
    "SamplingConfig",
    "ColumnSample",
    "ColumnClassification",
    "create_llm_preprocessor",
    "process_for_llm_classification",
    # Analysis Engine
    "BIDSColumnAnalysisEngine",
    "AnalysisConfig",
    "FileAnalysisResult",
    "BatchAnalysisResult",
    "create_analysis_engine",
    "analyze_bids_files",
    "analyze_bids_directory",
    # CLI
    "cli_main",
]
