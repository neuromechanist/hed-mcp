"""HED Sidecar Generation Pipeline.

This package provides a modular pipeline architecture for generating HED sidecars
from BIDS event files. The pipeline consists of configurable stages that handle
data ingestion, column classification, HED mapping, and validation.

Architecture:
- Core pipeline orchestration and stage management
- Configurable pipeline stages for modular processing
- Performance optimization for <10 second requirement
- Integration with existing HED tools and TabularSummary
"""

from .core import (
    PipelineStage,
    SidecarPipeline,
    PipelineContext,
    StageStatus,
    PipelineException,
)
from .config import PipelineConfig, StageConfig
from .stages import (
    DataIngestionStage,
    ColumnClassificationStage,
    HEDMappingStage,
    SidecarGenerationStage,
    ValidationStage,
)

__all__ = [
    # Core pipeline components
    "PipelineStage",
    "SidecarPipeline",
    "PipelineContext",
    "StageStatus",
    "PipelineException",
    # Configuration
    "PipelineConfig",
    "StageConfig",
    # Stages
    "DataIngestionStage",
    "ColumnClassificationStage",
    "HEDMappingStage",
    "SidecarGenerationStage",
    "ValidationStage",
    # Factory functions
    "create_sidecar_pipeline",
    "create_default_pipeline",
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

    # Register default stage implementations
    pipeline.register_stage_type(DataIngestionStage)
    pipeline.register_stage_type(ColumnClassificationStage)
    pipeline.register_stage_type(HEDMappingStage)
    pipeline.register_stage_type(SidecarGenerationStage)
    pipeline.register_stage_type(ValidationStage)

    return pipeline


def create_default_pipeline() -> SidecarPipeline:
    """Create a pipeline with default configuration and all standard stages.

    Returns:
        Ready-to-use SidecarPipeline instance
    """
    config = PipelineConfig()
    pipeline = create_sidecar_pipeline(config)

    # Add default stage instances in execution order
    pipeline.add_stage("data_ingestion", config.stage_configs.get("data_ingestion", {}))
    pipeline.add_stage(
        "column_classification", config.stage_configs.get("column_classification", {})
    )
    pipeline.add_stage("hed_mapping", config.stage_configs.get("hed_mapping", {}))
    pipeline.add_stage(
        "sidecar_generation", config.stage_configs.get("sidecar_generation", {})
    )

    if config.validation_enabled:
        pipeline.add_stage("validation", config.stage_configs.get("validation", {}))

    return pipeline
