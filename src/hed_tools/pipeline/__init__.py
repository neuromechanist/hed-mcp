"""HED Sidecar Generation Pipeline Package.

This package provides a modular pipeline architecture for generating HED sidecars
from event data files, with support for performance optimization, caching, and
parallel processing.

Architecture:
- Core pipeline orchestration and stage management
- Configurable pipeline stages for modular processing
- Performance optimization for <10 second requirement
- Integration with existing HED tools and TabularSummary
"""

from .core import (
    PipelineContext,
    StageStatus,
    SidecarPipeline,
    PipelineException,
)

from .config import PipelineConfig, StageConfig, ConfigurationError

from .performance import (
    PipelinePerformanceManager,
    PerformanceProfiler,
    AdvancedCache,
    ParallelProcessor,
    MemoryOptimizer,
    PerformanceTestFramework,
    create_optimized_pipeline_manager,
    performance_decorator,
)

# Import all pipeline stages
from .stages import (
    PipelineStage,
    DataIngestionStage,
    ColumnClassificationStage,
    HEDMappingStage,
    SidecarGenerationStage,
    ValidationStage,
)

__all__ = [
    # Core pipeline components
    "PipelineStage",
    "PipelineContext",
    "StageStatus",
    "SidecarPipeline",
    "PipelineException",
    # Configuration
    "PipelineConfig",
    "StageConfig",
    "ConfigurationError",
    # Performance optimization
    "PipelinePerformanceManager",
    "PerformanceProfiler",
    "AdvancedCache",
    "ParallelProcessor",
    "MemoryOptimizer",
    "PerformanceTestFramework",
    "create_optimized_pipeline_manager",
    "performance_decorator",
    # Pipeline stages
    "DataIngestionStage",
    "ColumnClassificationStage",
    "HEDMappingStage",
    "SidecarGenerationStage",
    "ValidationStage",
]


def create_sidecar_pipeline(config: PipelineConfig = None) -> SidecarPipeline:
    """Create a fully configured sidecar generation pipeline.

    Args:
        config: Optional pipeline configuration

    Returns:
        Configured SidecarPipeline instance
    """
    if config is None:
        config = PipelineConfig()

    pipeline = SidecarPipeline(config)

    return pipeline


def create_default_pipeline() -> SidecarPipeline:
    """Create a pipeline with default configuration and all standard stages.

    Returns:
        Ready-to-use SidecarPipeline instance
    """
    config = PipelineConfig()
    pipeline = create_sidecar_pipeline(config)

    # Add default stage instances in execution order
    pipeline.add_stage(
        DataIngestionStage(config.stage_configs.get("data_ingestion", {}))
    )
    pipeline.add_stage(
        ColumnClassificationStage(config.stage_configs.get("column_classification", {}))
    )
    pipeline.add_stage(HEDMappingStage(config.stage_configs.get("hed_mapping", {})))
    pipeline.add_stage(
        SidecarGenerationStage(config.stage_configs.get("sidecar_generation", {}))
    )

    if config.validation_enabled:
        pipeline.add_stage(ValidationStage(config.stage_configs.get("validation", {})))

    return pipeline
