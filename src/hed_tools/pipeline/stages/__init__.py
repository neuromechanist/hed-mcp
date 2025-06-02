"""Pipeline stages package for HED sidecar generation.

This package contains all pipeline stages including:
- Data ingestion and validation
- Column classification and analysis
- HED mapping and tagging
- Sidecar generation and formatting
- Validation and quality assurance

Each stage follows the base PipelineStage interface for consistency.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Type
import logging
import time

logger = logging.getLogger(__name__)

# Stage registry for dynamic stage loading
_stage_registry: Dict[str, Type["PipelineStage"]] = {}


def register_stage(name: str, stage_class: Type["PipelineStage"]) -> None:
    """Register a pipeline stage class with a name."""
    _stage_registry[name] = stage_class
    logger.debug(f"Registered pipeline stage: {name}")


def get_stage_class(name: str) -> Type["PipelineStage"]:
    """Get a registered stage class by name."""
    if name not in _stage_registry:
        raise ValueError(
            f"Unknown stage: {name}. Available stages: {list(_stage_registry.keys())}"
        )
    return _stage_registry[name]


def get_available_stages() -> List[str]:
    """Get list of available stage names."""
    return list(_stage_registry.keys())


@dataclass
class StageInput:
    """Input data container for pipeline stages."""

    data: Any
    metadata: Dict[str, Any]
    context: Dict[str, Any]

    def get_data(self) -> Any:
        """Get the main data payload."""
        return self.data


@dataclass
class StageOutput:
    """Output data container from pipeline stages."""

    data: Any
    metadata: Dict[str, Any]
    context: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    stage_metrics: Dict[str, Any]

    def has_errors(self) -> bool:
        """Check if output contains errors."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if output contains warnings."""
        return len(self.warnings) > 0

    def get_data(self) -> Any:
        """Get the main data payload."""
        return self.data


def create_stage_output(
    data: Any = None,
    metadata: Dict[str, Any] = None,
    context: Dict[str, Any] = None,
    warnings: List[str] = None,
    errors: List[str] = None,
    stage_metrics: Dict[str, Any] = None,
) -> StageOutput:
    """Helper function to create StageOutput with defaults."""
    return StageOutput(
        data=data,
        metadata=metadata or {},
        context=context or {},
        warnings=warnings or [],
        errors=errors or [],
        stage_metrics=stage_metrics or {},
    )


class PipelineStage(ABC):
    """Abstract base class for all pipeline stages.

    All pipeline stages must implement this interface to ensure
    consistent behavior and integration with the pipeline framework.
    """

    def __init__(self, name: str, config: Any = None):
        self.name = name
        # Handle both dict and StageConfig objects
        if config is None:
            self.config = {}
        elif hasattr(config, "get"):  # Dict-like object
            self.config = config
        elif hasattr(config, "parameters"):  # StageConfig object
            self.config = config
        else:
            self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._initialized = False
        self._performance_metrics = {}

    async def initialize(self) -> None:
        """Initialize the pipeline stage."""
        if self._initialized:
            return

        self.logger.info(f"Initializing pipeline stage: {self.name}")
        await self._initialize_implementation()
        self._initialized = True

    async def execute(self, stage_input: StageInput) -> StageOutput:
        """Execute the pipeline stage with input validation."""
        if not self._initialized:
            await self.initialize()

        # Validate input
        await self._validate_input(stage_input)

        # Execute stage-specific logic
        self.logger.debug(f"Executing stage: {self.name}")
        start_time = time.perf_counter()

        try:
            result = await self._execute_implementation(stage_input)
            execution_time = time.perf_counter() - start_time

            # Add performance metrics
            result.stage_metrics.update(
                {
                    "execution_time_seconds": execution_time,
                    "stage_name": self.name,
                    "success": not result.has_errors(),
                }
            )

            if result.has_errors():
                self.logger.error(
                    f"Stage {self.name} completed with errors: {result.errors}"
                )
            elif result.has_warnings():
                self.logger.warning(
                    f"Stage {self.name} completed with warnings: {result.warnings}"
                )
            else:
                self.logger.info(
                    f"Stage {self.name} completed successfully in {execution_time:.3f}s"
                )

            return result

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            self.logger.error(f"Stage {self.name} failed: {e}", exc_info=True)

            return create_stage_output(
                data=None,
                metadata=stage_input.metadata,
                context=stage_input.context,
                errors=[f"Stage execution failed: {str(e)}"],
                stage_metrics={
                    "execution_time_seconds": execution_time,
                    "stage_name": self.name,
                    "success": False,
                    "error": str(e),
                },
            )

    async def cleanup(self) -> None:
        """Clean up stage resources."""
        if self._initialized:
            self.logger.info(f"Cleaning up pipeline stage: {self.name}")
            await self._cleanup_implementation()
            self._initialized = False

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with optional default."""
        if hasattr(self.config, "get"):
            # Dict-like object or StageConfig with get method
            return self.config.get(key, default)
        elif hasattr(self.config, "parameters"):
            # StageConfig object - check parameters
            return self.config.parameters.get(key, default)
        else:
            # Fallback for other types
            return getattr(self.config, key, default)

    @abstractmethod
    async def _initialize_implementation(self) -> None:
        """Stage-specific initialization logic."""
        pass

    @abstractmethod
    async def _execute_implementation(self, stage_input: StageInput) -> StageOutput:
        """Stage-specific execution logic."""
        pass

    @abstractmethod
    async def _cleanup_implementation(self) -> None:
        """Stage-specific cleanup logic."""
        pass

    async def _validate_input(self, stage_input: StageInput) -> None:
        """Validate stage input. Override in subclasses for specific validation."""
        if not isinstance(stage_input, StageInput):
            raise ValueError(
                f"Invalid input type: {type(stage_input)}. Expected StageInput."
            )

        if stage_input.metadata is None:
            stage_input.metadata = {}

        if stage_input.context is None:
            stage_input.context = {}


__all__ = [
    "PipelineStage",
    "StageInput",
    "StageOutput",
    "create_stage_output",
    "register_stage",
    "get_stage_class",
    "get_available_stages",
    "DataIngestionStage",
    "ColumnClassificationStage",
    "HEDMappingStage",
    "SidecarGenerationStage",
    "ValidationStage",
]

# Import stage implementations at the end to avoid circular imports
from .data_ingestion import DataIngestionStage  # noqa: E402
from .column_classification import ColumnClassificationStage  # noqa: E402
from .hed_mapping import HEDMappingStage  # noqa: E402
from .sidecar_generation import SidecarGenerationStage  # noqa: E402
from .validation import ValidationStage  # noqa: E402
