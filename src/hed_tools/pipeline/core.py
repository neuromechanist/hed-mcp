"""Core pipeline architecture for HED sidecar generation.

This module implements the foundational components for the modular pipeline:
- PipelineStage abstract base class for all processing stages
- SidecarPipeline orchestrator for stage execution and management
- PipelineContext for shared state across stages
- Error handling and performance monitoring
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class StageStatus(Enum):
    """Status values for pipeline stages."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineException(Exception):
    """Custom exception for pipeline-related errors."""

    def __init__(
        self,
        message: str,
        stage_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.stage_name = stage_name
        self.context = context or {}


@dataclass
class PipelineContext:
    """Shared state container across pipeline stages.

    This class maintains data and metadata that flows between pipeline stages,
    providing a centralized way to share information and track progress.
    """

    # Input data from MCP request
    input_data: Dict[str, Any] = field(default_factory=dict)

    # Processed data at each stage
    processed_data: Dict[str, Any] = field(default_factory=dict)

    # Metadata about the processing
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Results from each stage
    stage_results: Dict[str, Any] = field(default_factory=dict)

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Performance tracking
    stage_timings: Dict[str, float] = field(default_factory=dict)
    total_start_time: Optional[float] = None

    def add_error(self, error: str, stage_name: Optional[str] = None) -> None:
        """Add an error to the context."""
        if stage_name:
            error_msg = f"[{stage_name}] {error}"
        else:
            error_msg = error
        self.errors.append(error_msg)
        logger.error(error_msg)

    def add_warning(self, warning: str, stage_name: Optional[str] = None) -> None:
        """Add a warning to the context."""
        if stage_name:
            warning_msg = f"[{stage_name}] {warning}"
        else:
            warning_msg = warning
        self.warnings.append(warning_msg)
        logger.warning(warning_msg)

    def set_stage_result(self, stage_name: str, result: Any) -> None:
        """Set the result for a specific stage."""
        self.stage_results[stage_name] = result

    def get_stage_result(self, stage_name: str) -> Optional[Any]:
        """Get the result for a specific stage."""
        return self.stage_results.get(stage_name)

    def start_timing(self):
        """Start overall pipeline timing."""
        self.total_start_time = time.time()

    def record_stage_timing(self, stage_name: str, duration: float):
        """Record timing for a specific stage."""
        self.stage_timings[stage_name] = duration

    def get_total_duration(self) -> Optional[float]:
        """Get total pipeline duration if timing was started."""
        if self.total_start_time is None:
            return None
        return time.time() - self.total_start_time

    def has_errors(self) -> bool:
        """Check if any errors were recorded."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if any warnings were recorded."""
        return len(self.warnings) > 0


class PipelineStage(ABC):
    """Abstract base class for pipeline stages.

    All pipeline stages must inherit from this class and implement the execute method.
    Stages should be stateless and communicate only through the PipelineContext.
    """

    def __init__(self, name: str, config: Dict[str, Any] = None):
        """Initialize the pipeline stage.

        Args:
            name: Unique name for this stage
            config: Stage-specific configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.status = StageStatus.PENDING
        self.logger = logging.getLogger(f"pipeline.{name}")
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    @abstractmethod
    async def execute(self, context: PipelineContext) -> bool:
        """Execute the stage logic.

        Args:
            context: Pipeline context containing shared state

        Returns:
            True if stage completed successfully, False if it failed

        Raises:
            PipelineException: For stage-specific errors
        """
        pass

    async def validate_prerequisites(self, context: PipelineContext) -> bool:
        """Validate that prerequisites for this stage are met.

        Override this method to add stage-specific validation logic.

        Args:
            context: Pipeline context to validate

        Returns:
            True if prerequisites are met, False otherwise
        """
        return True

    async def cleanup(self, context: PipelineContext) -> None:
        """Cleanup resources after stage execution.

        Override this method to add stage-specific cleanup logic.

        Args:
            context: Pipeline context
        """
        pass

    def get_duration(self) -> Optional[float]:
        """Get the duration of the last execution."""
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    def _start_execution(self):
        """Mark the start of stage execution."""
        self.status = StageStatus.RUNNING
        self.start_time = time.time()
        self.logger.info(f"Starting stage: {self.name}")

    def _end_execution(self, success: bool):
        """Mark the end of stage execution."""
        self.end_time = time.time()
        self.status = StageStatus.COMPLETED if success else StageStatus.FAILED
        duration = self.get_duration()

        if success:
            self.logger.info(f"Completed stage: {self.name} ({duration:.2f}s)")
        else:
            self.logger.error(f"Failed stage: {self.name} ({duration:.2f}s)")


class SidecarPipeline:
    """Main pipeline orchestrator for HED sidecar generation.

    This class manages the execution of pipeline stages, handles errors,
    and provides monitoring and performance tracking capabilities.
    """

    def __init__(self, config):
        """Initialize the pipeline.

        Args:
            config: PipelineConfig instance
        """
        self.config = config
        self.stages: List[PipelineStage] = []
        self.stage_registry: Dict[str, Type[PipelineStage]] = {}
        self.logger = logging.getLogger("pipeline.core")
        self._execution_id: Optional[str] = None

    def register_stage_type(self, stage_class: Type[PipelineStage]):
        """Register a stage class for dynamic instantiation.

        Args:
            stage_class: PipelineStage subclass to register
        """
        stage_name = stage_class.__name__
        self.stage_registry[stage_name] = stage_class
        self.logger.debug(f"Registered stage type: {stage_name}")

    def add_stage(
        self, stage_name: str, stage_config: Dict[str, Any] = None
    ) -> PipelineStage:
        """Add a stage instance to the pipeline.

        Args:
            stage_name: Name of the stage class to instantiate
            stage_config: Configuration for the stage instance

        Returns:
            The created stage instance

        Raises:
            ValueError: If stage_name is not registered
        """
        # Convert snake_case to CamelCase for class lookup
        class_name = self._snake_to_camel(stage_name)

        if class_name not in self.stage_registry:
            available_stages = list(self.stage_registry.keys())
            raise ValueError(
                f"Stage '{class_name}' not registered. Available: {available_stages}"
            )

        stage_class = self.stage_registry[class_name]
        stage_instance = stage_class(stage_name, stage_config or {})
        self.stages.append(stage_instance)

        self.logger.info(f"Added stage: {stage_name} ({class_name})")
        return stage_instance

    def _snake_to_camel(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.split("_")
        return "".join(word.capitalize() for word in components) + "Stage"

    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Execute the complete pipeline.

        Args:
            context: Pipeline context with input data

        Returns:
            Dictionary containing execution results and metadata
        """
        self._execution_id = f"exec_{int(time.time())}"
        context.start_timing()

        self.logger.info(
            f"Starting pipeline execution {self._execution_id} with {len(self.stages)} stages"
        )

        results = {
            "success": True,
            "execution_id": self._execution_id,
            "stage_results": {},
            "errors": [],
            "warnings": [],
            "performance": {},
        }

        try:
            # Execute stages in order
            for stage in self.stages:
                stage_success = await self._execute_stage_with_monitoring(
                    stage, context
                )

                # Record stage results
                results["stage_results"][stage.name] = {
                    "success": stage_success,
                    "status": stage.status.value,
                    "duration": stage.get_duration(),
                    "result": context.get_stage_result(stage.name),
                }

                # Stop pipeline on stage failure unless configured to continue
                if not stage_success:
                    if not self.config.continue_on_error:
                        results["success"] = False
                        break
                    else:
                        stage.status = StageStatus.SKIPPED
                        context.add_warning(
                            f"Stage {stage.name} failed but continuing", stage.name
                        )

            # Final result compilation
            results["errors"] = context.errors
            results["warnings"] = context.warnings
            results["performance"] = {
                "total_duration": context.get_total_duration(),
                "stage_timings": context.stage_timings,
                "bottleneck_stage": self._identify_bottleneck(context.stage_timings),
            }

            success_msg = (
                "Pipeline completed successfully"
                if results["success"]
                else "Pipeline completed with errors"
            )
            self.logger.info(
                f"{success_msg} in {results['performance']['total_duration']:.2f}s"
            )

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            results["success"] = False
            results["errors"].append(f"Pipeline execution error: {str(e)}")

        return results

    async def _execute_stage_with_monitoring(
        self, stage: PipelineStage, context: PipelineContext
    ) -> bool:
        """Execute a single stage with comprehensive monitoring.

        Args:
            stage: Stage to execute
            context: Pipeline context

        Returns:
            True if stage succeeded, False otherwise
        """
        try:
            # Validate prerequisites
            if not await stage.validate_prerequisites(context):
                context.add_error("Prerequisites not met for stage", stage.name)
                return False

            stage._start_execution()

            # Execute with timeout if configured
            if self.config.timeout_seconds > 0:
                success = await asyncio.wait_for(
                    stage.execute(context), timeout=self.config.timeout_seconds
                )
            else:
                success = await stage.execute(context)

            stage._end_execution(success)

            # Record timing
            if stage.get_duration() is not None:
                context.record_stage_timing(stage.name, stage.get_duration())

            return success

        except asyncio.TimeoutError:
            stage._end_execution(False)
            error_msg = f"Stage timed out after {self.config.timeout_seconds}s"
            context.add_error(error_msg, stage.name)
            return False

        except Exception as e:
            stage._end_execution(False)
            error_msg = f"Stage execution failed: {str(e)}"
            context.add_error(error_msg, stage.name)
            self.logger.error(f"Stage {stage.name} failed: {e}", exc_info=True)
            return False

        finally:
            # Always attempt cleanup
            try:
                await stage.cleanup(context)
            except Exception as cleanup_error:
                self.logger.warning(
                    f"Stage {stage.name} cleanup failed: {cleanup_error}"
                )

    def _identify_bottleneck(self, stage_timings: Dict[str, float]) -> Optional[str]:
        """Identify the slowest stage in the pipeline.

        Args:
            stage_timings: Dictionary of stage names to durations

        Returns:
            Name of the slowest stage, or None if no timings available
        """
        if not stage_timings:
            return None

        return max(stage_timings.items(), key=lambda x: x[1])[0]

    def get_stage_count(self) -> int:
        """Get the number of stages in the pipeline."""
        return len(self.stages)

    def get_stage_names(self) -> List[str]:
        """Get the names of all stages in execution order."""
        return [stage.name for stage in self.stages]
