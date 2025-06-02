"""HED Tools package - Advanced tools for working with HED (Hierarchical Event Descriptor) data."""

from .bids_parser import (
    BIDSEventParser,
    BIDSValidationError,
    parse_bids_events,
)
from .enhanced_column_analyzer import (
    EnhancedColumnAnalyzer,
    NumericColumnAnalyzer,
    CategoricalColumnAnalyzer,
    TemporalColumnAnalyzer,
    TextColumnAnalyzer,
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
from .performance_optimizer import (
    MemoryManager,
    ChunkedProcessor,
    LazyDataLoader,
    ParallelProcessor,
    PerformanceBenchmark,
    MemoryMetrics,
    PerformanceMetrics,
    ChunkProcessingConfig,
    ParallelProcessingConfig,
    create_optimized_config,
    optimize_for_large_datasets,
)
from .cli import cli_main

__all__ = [
    # Parsing
    "BIDSEventParser",
    "BIDSValidationError",
    "parse_bids_events",
    # Enhanced Analysis
    "EnhancedColumnAnalyzer",
    "NumericColumnAnalyzer",
    "CategoricalColumnAnalyzer",
    "TemporalColumnAnalyzer",
    "TextColumnAnalyzer",
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
    # Performance Optimizer
    "MemoryManager",
    "ChunkedProcessor",
    "LazyDataLoader",
    "ParallelProcessor",
    "PerformanceBenchmark",
    "MemoryMetrics",
    "PerformanceMetrics",
    "ChunkProcessingConfig",
    "ParallelProcessingConfig",
    "create_optimized_config",
    "optimize_for_large_datasets",
    # CLI
    "cli_main",
]
