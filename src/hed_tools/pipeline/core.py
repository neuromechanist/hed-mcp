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
from typing import Any, Dict, List, Optional

from .config import PipelineConfig
from .performance import create_optimized_pipeline_manager

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
    """Enhanced pipeline with performance optimization capabilities.

    This pipeline includes:
    - Performance monitoring and profiling
    - Intelligent caching
    - Parallel processing for independent stages
    - Memory optimization
    - Comprehensive error handling
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stages: List[PipelineStage] = []
        self.logger = logging.getLogger("pipeline.core")

        # Initialize performance manager
        performance_config = {
            "enable_memory_tracking": True,
            "cache_size": 1000,
            "cache_ttl": 3600,
            "target_total_time": 10.0,
            "target_stage_time": 3.0,
            "memory_limit_mb": 500,
        }

        self.performance_manager = create_optimized_pipeline_manager(performance_config)

        # Stage execution tracking
        self.execution_history: List[Dict[str, Any]] = []

    def add_stage(self, stage: PipelineStage) -> None:
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        self.logger.debug(f"Added stage: {stage.name}")

    def add_stages(self, stages: List[PipelineStage]) -> None:
        """Add multiple stages to the pipeline."""
        for stage in stages:
            self.add_stage(stage)

    async def execute(self, context: PipelineContext) -> bool:
        """Execute the pipeline with performance optimization."""
        pipeline_start_time = time.time()
        self.logger.info(f"Starting pipeline execution with {len(self.stages)} stages")

        # Initialize execution tracking
        execution_summary = {
            "pipeline_start_time": pipeline_start_time,
            "stages_executed": [],
            "total_stages": len(self.stages),
            "success": True,
            "errors": [],
            "performance_metrics": {},
        }

        try:
            # Check for parallelizable stages
            parallelizable_stages = self._identify_parallelizable_stages()

            if parallelizable_stages and len(parallelizable_stages) > 1:
                await self._execute_with_parallel_optimization(
                    context, execution_summary
                )
            else:
                await self._execute_sequential_optimized(context, execution_summary)

        except Exception as e:
            execution_summary["success"] = False
            execution_summary["errors"].append(str(e))
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            return False
        finally:
            # Finalize execution summary
            execution_summary["total_execution_time"] = (
                time.time() - pipeline_start_time
            )
            execution_summary["performance_metrics"] = (
                self.performance_manager.get_performance_report()
            )
            self.execution_history.append(execution_summary)

            # Log performance summary
            self._log_execution_summary(execution_summary)

        return execution_summary["success"]

    async def _execute_sequential_optimized(
        self, context: PipelineContext, execution_summary: Dict[str, Any]
    ) -> None:
        """Execute stages sequentially with optimization."""

        for i, stage in enumerate(self.stages):
            stage_start_time = time.time()

            try:
                # Check prerequisites
                if not await stage.validate_prerequisites(context):
                    error_msg = f"Stage {stage.name} prerequisites not met"
                    context.add_error(error_msg, "pipeline")
                    execution_summary["errors"].append(error_msg)
                    execution_summary["success"] = False
                    break

                # Execute stage with performance optimization
                success = await self.performance_manager.execute_optimized_stage(
                    stage_name=stage.name,
                    stage_func=self._execute_stage_wrapper,
                    stage_args=(stage, context),
                    stage_kwargs={},
                    enable_caching=self._is_stage_cacheable(stage),
                )

                if not success:
                    execution_summary["success"] = False
                    execution_summary["errors"].extend(context.errors)
                    break

                # Track successful execution
                stage_metrics = {
                    "stage_name": stage.name,
                    "execution_time": time.time() - stage_start_time,
                    "success": True,
                    "stage_index": i,
                }
                execution_summary["stages_executed"].append(stage_metrics)

            except Exception as e:
                stage_metrics = {
                    "stage_name": stage.name,
                    "execution_time": time.time() - stage_start_time,
                    "success": False,
                    "error": str(e),
                    "stage_index": i,
                }
                execution_summary["stages_executed"].append(stage_metrics)
                execution_summary["errors"].append(str(e))
                execution_summary["success"] = False
                break

    async def _execute_with_parallel_optimization(
        self, context: PipelineContext, execution_summary: Dict[str, Any]
    ) -> None:
        """Execute pipeline with parallel processing for independent stages."""

        # Group stages by dependencies
        stage_groups = self._group_stages_by_dependencies()

        for group_index, stage_group in enumerate(stage_groups):
            if len(stage_group) == 1:
                # Single stage - execute normally
                stage = stage_group[0]
                success = await self._execute_single_stage_optimized(stage, context)
                if not success:
                    execution_summary["success"] = False
                    break
            else:
                # Multiple independent stages - execute in parallel
                success = await self._execute_parallel_stage_group(stage_group, context)
                if not success:
                    execution_summary["success"] = False
                    break

    async def _execute_single_stage_optimized(
        self, stage: PipelineStage, context: PipelineContext
    ) -> bool:
        """Execute a single stage with optimization."""

        # Check prerequisites
        if not await stage.validate_prerequisites(context):
            context.add_error(f"Stage {stage.name} prerequisites not met", "pipeline")
            return False

        # Execute with performance monitoring
        return await self.performance_manager.execute_optimized_stage(
            stage_name=stage.name,
            stage_func=self._execute_stage_wrapper,
            stage_args=(stage, context),
            stage_kwargs={},
            enable_caching=self._is_stage_cacheable(stage),
        )

    async def _execute_parallel_stage_group(
        self, stage_group: List[PipelineStage], context: PipelineContext
    ) -> bool:
        """Execute a group of independent stages in parallel."""

        self.logger.info(f"Executing {len(stage_group)} stages in parallel")

        # Prepare stage functions and contexts
        stage_functions = []
        stage_contexts = []

        for stage in stage_group:
            # Create separate context for each parallel stage
            stage_context = PipelineContext(
                input_data=context.input_data.copy(),
                processed_data=context.processed_data.copy(),
                metadata=context.metadata.copy(),
                stage_results=context.stage_results.copy(),
            )

            stage_functions.append(
                lambda s=stage, sc=stage_context: self._execute_stage_sync(s, sc)
            )
            stage_contexts.append(stage_context)

        # Execute in parallel
        try:
            results = await self.performance_manager.parallel_processor.execute_parallel_stages(
                stage_functions=stage_functions,
                stage_contexts=stage_contexts,
                timeout=30.0,
            )

            # Merge results back to main context
            success = True
            for i, result in enumerate(results):
                if result is None or (isinstance(result, bool) and not result):
                    success = False
                    self.logger.error(f"Parallel stage {stage_group[i].name} failed")
                else:
                    # Merge successful stage results
                    stage_context = stage_contexts[i]
                    context.stage_results.update(stage_context.stage_results)
                    context.processed_data.update(stage_context.processed_data)
                    context.errors.extend(stage_context.errors)

            return success

        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}")
            return False

    def _execute_stage_sync(
        self, stage: PipelineStage, context: PipelineContext
    ) -> bool:
        """Synchronous wrapper for stage execution (for parallel processing)."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(stage.execute(context))
        finally:
            loop.close()

    async def _execute_stage_wrapper(
        self, stage: PipelineStage, context: PipelineContext
    ) -> bool:
        """Wrapper for stage execution with additional monitoring."""
        try:
            return await stage.execute(context)
        except Exception as e:
            context.add_error(
                f"Stage {stage.name} execution failed: {str(e)}", stage.name
            )
            self.logger.error(f"Stage {stage.name} failed: {e}", exc_info=True)
            return False

    def _identify_parallelizable_stages(self) -> List[PipelineStage]:
        """Identify stages that can be executed in parallel."""
        # For now, identify stages with no dependencies or independent stages
        # This is a simplified implementation - could be enhanced with dependency analysis

        independent_stages = []
        for stage in self.stages:
            # Check if stage has specific parallel processing support
            if hasattr(stage, "supports_parallel") and stage.supports_parallel:
                independent_stages.append(stage)

        return independent_stages

    def _group_stages_by_dependencies(self) -> List[List[PipelineStage]]:
        """Group stages by their dependencies for parallel execution."""
        # Simplified grouping - stages with no dependencies can run in parallel
        # More sophisticated dependency analysis could be implemented

        groups = []
        current_group = []

        for stage in self.stages:
            # Check if stage depends on results from previous stages
            if self._stage_has_dependencies(stage, current_group):
                # Start new group
                if current_group:
                    groups.append(current_group)
                current_group = [stage]
            else:
                # Add to current group
                current_group.append(stage)

        if current_group:
            groups.append(current_group)

        return groups

    def _stage_has_dependencies(
        self, stage: PipelineStage, previous_stages: List[PipelineStage]
    ) -> bool:
        """Check if a stage has dependencies on previous stages."""
        # Simple implementation - could be enhanced with formal dependency tracking
        return hasattr(stage, "dependencies") and stage.dependencies

    def _is_stage_cacheable(self, stage: PipelineStage) -> bool:
        """Determine if a stage's results can be cached."""
        # Stages that are deterministic and don't depend on external state
        cacheable_stage_types = [
            "ColumnClassificationStage",
            "HEDMappingStage",
            "SidecarGenerationStage",
        ]

        return type(stage).__name__ in cacheable_stage_types

    def _log_execution_summary(self, execution_summary: Dict[str, Any]) -> None:
        """Log pipeline execution summary."""
        total_time = execution_summary["total_execution_time"]
        success = execution_summary["success"]
        stages_count = len(execution_summary["stages_executed"])

        if success:
            self.logger.info(
                f"Pipeline completed successfully: {total_time:.2f}s, "
                f"{stages_count}/{execution_summary['total_stages']} stages executed"
            )

            # Check performance target
            if total_time > 10.0:
                self.logger.warning(
                    f"Pipeline execution time ({total_time:.2f}s) exceeds 10s target"
                )
        else:
            self.logger.error(
                f"Pipeline failed after {total_time:.2f}s, "
                f"errors: {len(execution_summary['errors'])}"
            )

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return self.performance_manager.get_performance_report()

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get pipeline execution history."""
        return self.execution_history

    def clear_cache(self) -> None:
        """Clear performance caches."""
        self.performance_manager.cache.clear()
        self.logger.info("Pipeline caches cleared")

    async def cleanup(self, context: PipelineContext) -> None:
        """Clean up pipeline resources."""
        for stage in self.stages:
            try:
                await stage.cleanup(context)
            except Exception as e:
                self.logger.warning(f"Stage {stage.name} cleanup failed: {e}")

        # Generate final performance report
        final_report = self.get_performance_report()
        performance_score = final_report.get("performance_summary", {}).get(
            "performance_score", 0
        )
        self.logger.info(
            f"Pipeline cleanup completed. Final performance score: {performance_score:.1f}%"
        )
