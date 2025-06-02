"""Core pipeline architecture for HED sidecar generation.

This module implements the foundational components for the modular pipeline:
- SidecarPipeline orchestrator for stage execution and management
- PipelineContext for shared state across stages
- Error handling and performance monitoring
"""

import logging
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .config import PipelineConfig

if TYPE_CHECKING:
    from .stages import PipelineStage

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


class SidecarPipeline:
    """Simplified pipeline for executing HED sidecar generation stages.

    This pipeline executes stages in sequence, handling data flow between
    stages using the StageInput/StageOutput interface.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stages: List["PipelineStage"] = []
        self.logger = logging.getLogger("pipeline.core")

    def add_stage(self, stage: "PipelineStage") -> None:
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        self.logger.debug(f"Added stage: {stage.name}")

    def add_stages(self, stages: List["PipelineStage"]) -> None:
        """Add multiple stages to the pipeline."""
        for stage in stages:
            self.add_stage(stage)

    async def execute(
        self, input_data: Any, metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute the pipeline with the given input data.

        Args:
            input_data: The data to process
            metadata: Optional metadata about the input

        Returns:
            Dict containing the final result and execution information
        """
        pipeline_start_time = time.time()
        self.logger.info(f"Starting pipeline execution with {len(self.stages)} stages")

        if metadata is None:
            metadata = {}

        # Initialize pipeline state
        current_data = input_data
        execution_info = {
            "stages_executed": [],
            "total_stages": len(self.stages),
            "success": True,
            "errors": [],
            "warnings": [],
            "stage_results": {},
        }

        try:
            # Execute stages sequentially
            for i, stage in enumerate(self.stages):
                stage_start_time = time.time()

                self.logger.info(
                    f"Executing stage {i+1}/{len(self.stages)}: {stage.name}"
                )

                # Create stage input
                from .stages import StageInput

                stage_input = StageInput(
                    data=current_data, metadata=metadata.copy(), context={}
                )

                # Execute stage
                try:
                    result = await stage.execute(stage_input)
                    stage_duration = time.time() - stage_start_time

                    if result.has_errors():
                        execution_info["errors"].extend(result.errors)
                        execution_info["success"] = False
                        self.logger.error(
                            f"Stage {stage.name} failed with errors: {result.errors}"
                        )
                        break

                    if result.has_warnings():
                        execution_info["warnings"].extend(result.warnings)
                        self.logger.warning(
                            f"Stage {stage.name} completed with warnings: {result.warnings}"
                        )

                    # Update data for next stage
                    current_data = result.data
                    metadata.update(result.metadata)
                    execution_info["stage_results"][stage.name] = result.data

                    # Record stage execution
                    stage_info = {
                        "stage_name": stage.name,
                        "execution_time": stage_duration,
                        "success": True,
                        "warnings": len(result.warnings),
                        "stage_index": i,
                    }
                    execution_info["stages_executed"].append(stage_info)

                    self.logger.info(
                        f"Stage {stage.name} completed successfully in {stage_duration:.2f}s"
                    )

                except Exception as e:
                    stage_duration = time.time() - stage_start_time
                    error_msg = f"Stage {stage.name} execution failed: {str(e)}"
                    execution_info["errors"].append(error_msg)
                    execution_info["success"] = False

                    stage_info = {
                        "stage_name": stage.name,
                        "execution_time": stage_duration,
                        "success": False,
                        "error": str(e),
                        "stage_index": i,
                    }
                    execution_info["stages_executed"].append(stage_info)

                    self.logger.error(f"Stage {stage.name} failed: {e}", exc_info=True)
                    break

        except Exception as e:
            execution_info["success"] = False
            execution_info["errors"].append(f"Pipeline execution failed: {str(e)}")
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)

        # Finalize execution info
        total_duration = time.time() - pipeline_start_time
        execution_info["total_execution_time"] = total_duration

        # Log final result
        if execution_info["success"]:
            self.logger.info(
                f"Pipeline completed successfully in {total_duration:.2f}s "
                f"({len(execution_info['stages_executed'])}/{len(self.stages)} stages)"
            )
        else:
            self.logger.error(
                f"Pipeline failed after {total_duration:.2f}s with {len(execution_info['errors'])} errors"
            )

        return {
            "data": current_data,
            "metadata": metadata,
            "execution_info": execution_info,
            "success": execution_info["success"],
        }
