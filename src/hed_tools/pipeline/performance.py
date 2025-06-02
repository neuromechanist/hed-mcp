"""Performance optimization components for HED sidecar generation pipeline.

This module provides performance monitoring, caching, parallel processing,
and memory optimization to meet the <10 second execution time requirement.
"""

import asyncio
import gc
import hashlib
import json
import logging
import psutil
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple
import weakref

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    # Timing metrics
    total_execution_time: float = 0.0
    stage_execution_times: Dict[str, float] = field(default_factory=dict)
    cache_hit_rate: float = 0.0

    # Memory metrics
    peak_memory_usage_mb: float = 0.0
    memory_efficiency: float = 0.0

    # Throughput metrics
    operations_per_second: float = 0.0
    data_processed_mb: float = 0.0

    # Quality metrics
    performance_score: float = 0.0  # 0-100 score
    bottleneck_stages: List[str] = field(default_factory=list)

    def calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        # Performance targets
        target_time = 10.0  # seconds
        target_memory = 500  # MB

        # Time score (50% weight)
        time_score = max(0, 100 - (self.total_execution_time / target_time * 100))

        # Memory score (30% weight)
        memory_score = max(0, 100 - (self.peak_memory_usage_mb / target_memory * 100))

        # Cache efficiency score (20% weight)
        cache_score = self.cache_hit_rate * 100

        # Weighted average
        self.performance_score = (
            time_score * 0.5 + memory_score * 0.3 + cache_score * 0.2
        )

        return self.performance_score


class AdvancedCache:
    """Advanced caching system with TTL, size limits, and LRU eviction."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = Lock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _generate_key(self, stage_name: str, *args, **kwargs) -> str:
        """Generate cache key from function arguments."""
        key_data = {
            "stage": stage_name,
            "args": str(args),
            "kwargs": str(sorted(kwargs.items())),
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            value, timestamp = self._cache[key]

            # Check TTL
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                del self._access_times[key]
                self._stats["misses"] += 1
                return None

            # Update access time for LRU
            self._access_times[key] = time.time()
            self._stats["hits"] += 1
            return value

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        with self._lock:
            current_time = time.time()

            # Evict if necessary
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()

            self._cache[key] = (value, current_time)
            self._access_times[key] = current_time

    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return

        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        del self._cache[lru_key]
        del self._access_times[lru_key]
        self._stats["evictions"] += 1

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                self._stats["hits"] / total_requests if total_requests > 0 else 0.0
            )

            return {
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "max_size": self.max_size,
                "stats": self._stats.copy(),
            }


class MemoryOptimizer:
    """Memory optimization utilities."""

    def __init__(self, memory_limit_mb: int = 500):
        self.memory_limit_mb = memory_limit_mb
        self.process = psutil.Process()
        self._weak_refs: List[weakref.ref] = []

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits."""
        return self.get_memory_usage_mb() <= self.memory_limit_mb

    def optimize_memory(self) -> float:
        """Perform memory optimization and return freed memory in MB."""
        memory_before = self.get_memory_usage_mb()

        # Clean up weak references
        self._cleanup_weak_refs()

        # Force garbage collection
        collected = gc.collect()

        # Additional cleanup for large objects
        self._cleanup_large_objects()

        memory_after = self.get_memory_usage_mb()
        freed_mb = memory_before - memory_after

        logger.debug(
            f"Memory optimization: freed {freed_mb:.2f}MB, collected {collected} objects"
        )

        return freed_mb

    def _cleanup_weak_refs(self) -> None:
        """Clean up dead weak references."""
        self._weak_refs = [ref for ref in self._weak_refs if ref() is not None]

    def _cleanup_large_objects(self) -> None:
        """Clean up large objects that might be lingering."""
        # This is a placeholder for more sophisticated cleanup
        # Could include clearing specific caches, compacting data structures, etc.
        pass

    def register_for_cleanup(self, obj: Any) -> None:
        """Register object for automatic cleanup."""
        self._weak_refs.append(weakref.ref(obj))


class ParallelProcessor:
    """Parallel processing manager for independent pipeline stages."""

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(4, (psutil.cpu_count() or 1))
        self.thread_executor = None
        self.process_executor = None

    def __enter__(self):
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)
        if self.process_executor:
            self.process_executor.shutdown(wait=True)

    async def execute_parallel_stages(
        self,
        stage_functions: List[Callable],
        stage_contexts: List[Any],
        timeout: float = 30.0,
    ) -> List[Any]:
        """Execute multiple stage functions in parallel."""
        if not stage_functions:
            return []

        loop = asyncio.get_event_loop()

        # Create tasks for parallel execution
        tasks = []
        for func, context in zip(stage_functions, stage_contexts):
            task = loop.run_in_executor(self.thread_executor, func)
            tasks.append(task)

        try:
            # Wait for all tasks with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
            )

            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Parallel stage {i} failed: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)

            return processed_results

        except asyncio.TimeoutError:
            logger.error(f"Parallel execution timed out after {timeout}s")
            # Cancel remaining tasks
            for task in tasks:
                task.cancel()
            return [None] * len(stage_functions)

    async def execute_cpu_intensive_task(
        self, func: Callable, *args, timeout: float = 30.0, **kwargs
    ) -> Any:
        """Execute CPU-intensive task in process pool."""
        if not self.process_executor:
            self.process_executor = ProcessPoolExecutor(max_workers=2)

        loop = asyncio.get_event_loop()

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(self.process_executor, func, *args, **kwargs),
                timeout=timeout,
            )
            return result

        except asyncio.TimeoutError:
            logger.error(f"CPU-intensive task timed out after {timeout}s")
            return None


class PerformanceProfiler:
    """Performance profiler for detailed pipeline analysis."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.stage_timings: Dict[str, Dict[str, float]] = {}
        self.memory_snapshots: List[Tuple[float, float]] = []
        self.bottlenecks: List[Dict[str, Any]] = []

    def start_profiling(self) -> None:
        """Start profiling session."""
        self.start_time = time.perf_counter()
        self._take_memory_snapshot()

    def end_profiling(self) -> None:
        """End profiling session."""
        self.end_time = time.perf_counter()
        self._take_memory_snapshot()

    def start_stage(self, stage_name: str) -> None:
        """Start timing a pipeline stage."""
        if stage_name not in self.stage_timings:
            self.stage_timings[stage_name] = {}

        self.stage_timings[stage_name]["start"] = time.perf_counter()
        self._take_memory_snapshot()

    def end_stage(self, stage_name: str) -> None:
        """End timing a pipeline stage."""
        if stage_name not in self.stage_timings:
            return

        end_time = time.perf_counter()
        start_time = self.stage_timings[stage_name].get("start", end_time)

        self.stage_timings[stage_name]["end"] = end_time
        self.stage_timings[stage_name]["duration"] = end_time - start_time

        self._take_memory_snapshot()
        self._check_for_bottleneck(stage_name)

    def _take_memory_snapshot(self) -> None:
        """Take memory usage snapshot."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            timestamp = time.perf_counter()
            self.memory_snapshots.append((timestamp, memory_mb))
        except Exception as e:
            logger.warning(f"Failed to take memory snapshot: {e}")

    def _check_for_bottleneck(self, stage_name: str) -> None:
        """Check if stage is a performance bottleneck."""
        stage_data = self.stage_timings[stage_name]
        duration = stage_data.get("duration", 0)

        # Consider a stage a bottleneck if it takes more than 3 seconds
        if duration > 3.0:
            self.bottlenecks.append(
                {
                    "stage": stage_name,
                    "duration": duration,
                    "severity": "high" if duration > 5.0 else "medium",
                }
            )

    def get_metrics(self) -> PerformanceMetrics:
        """Get comprehensive performance metrics."""
        metrics = PerformanceMetrics()

        if self.start_time and self.end_time:
            metrics.total_execution_time = self.end_time - self.start_time

        # Stage execution times
        for stage_name, timing_data in self.stage_timings.items():
            duration = timing_data.get("duration", 0)
            metrics.stage_execution_times[stage_name] = duration

        # Memory metrics
        if self.memory_snapshots:
            memory_values = [snapshot[1] for snapshot in self.memory_snapshots]
            metrics.peak_memory_usage_mb = max(memory_values)

            # Calculate memory efficiency (lower variation is better)
            if len(memory_values) > 1:
                mean_memory = sum(memory_values) / len(memory_values)
                variance = sum((x - mean_memory) ** 2 for x in memory_values) / len(
                    memory_values
                )
                metrics.memory_efficiency = max(0, 100 - (variance / mean_memory * 100))

        # Bottlenecks
        metrics.bottleneck_stages = [b["stage"] for b in self.bottlenecks]

        # Calculate overall performance score
        metrics.calculate_performance_score()

        return metrics

    def get_detailed_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        metrics = self.get_metrics()

        return {
            "summary": {
                "total_time": metrics.total_execution_time,
                "performance_score": metrics.performance_score,
                "peak_memory_mb": metrics.peak_memory_usage_mb,
                "bottleneck_count": len(metrics.bottleneck_stages),
            },
            "stage_breakdown": {
                stage: {
                    "duration": duration,
                    "percentage": (
                        duration / metrics.total_execution_time * 100
                        if metrics.total_execution_time > 0
                        else 0
                    ),
                }
                for stage, duration in metrics.stage_execution_times.items()
            },
            "bottlenecks": self.bottlenecks,
            "memory_profile": {
                "peak_usage_mb": metrics.peak_memory_usage_mb,
                "efficiency_score": metrics.memory_efficiency,
                "snapshots": len(self.memory_snapshots),
            },
            "recommendations": self._generate_recommendations(metrics),
        }

    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []

        if metrics.total_execution_time > 10.0:
            recommendations.append(
                "Consider enabling parallel processing to reduce execution time"
            )

        if metrics.peak_memory_usage_mb > 500:
            recommendations.append(
                "Memory usage is high - consider enabling memory optimization"
            )

        if metrics.cache_hit_rate < 0.5:
            recommendations.append("Low cache hit rate - review caching strategy")

        for bottleneck in metrics.bottleneck_stages:
            recommendations.append(
                f"Stage '{bottleneck}' is a bottleneck - consider optimization"
            )

        if metrics.memory_efficiency < 70:
            recommendations.append(
                "High memory variance detected - review memory allocation patterns"
            )

        return recommendations


class PipelinePerformanceManager:
    """Main performance management system for the pipeline."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = AdvancedCache(
            max_size=config.get("cache_size", 1000),
            ttl_seconds=config.get("cache_ttl", 3600),
        )
        self.memory_optimizer = MemoryOptimizer(
            memory_limit_mb=config.get("memory_limit_mb", 500)
        )
        self.parallel_processor = ParallelProcessor(
            max_workers=config.get("max_workers", 4)
        )
        self.profiler = PerformanceProfiler()

        # Performance targets
        self.target_total_time = config.get("target_total_time", 10.0)
        self.target_stage_time = config.get("target_stage_time", 3.0)

        # Monitoring flags
        self.enable_memory_tracking = config.get("enable_memory_tracking", True)
        self.enable_profiling = config.get("enable_profiling", True)

    async def execute_optimized_stage(
        self,
        stage_name: str,
        stage_func: Callable,
        stage_args: Tuple = (),
        stage_kwargs: Dict = None,
        enable_caching: bool = True,
    ) -> Any:
        """Execute a stage with full optimization."""
        stage_kwargs = stage_kwargs or {}

        # Generate cache key
        cache_key = None
        if enable_caching:
            cache_key = self.cache._generate_key(
                stage_name, *stage_args, **stage_kwargs
            )
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for stage {stage_name}")
                return cached_result

        # Start profiling
        if self.enable_profiling:
            self.profiler.start_stage(stage_name)

        # Memory check before execution
        if self.enable_memory_tracking:
            if not self.memory_optimizer.check_memory_limit():
                freed_mb = self.memory_optimizer.optimize_memory()
                logger.info(
                    f"Memory optimization freed {freed_mb:.2f}MB before {stage_name}"
                )

        try:
            # Execute stage
            result = await stage_func(*stage_args, **stage_kwargs)

            # Cache successful results
            if enable_caching and cache_key and result is not None:
                self.cache.set(cache_key, result)
                logger.debug(f"Cached result for stage {stage_name}")

            return result

        finally:
            # End profiling
            if self.enable_profiling:
                self.profiler.end_stage(stage_name)

            # Memory cleanup after execution
            if self.enable_memory_tracking:
                current_memory = self.memory_optimizer.get_memory_usage_mb()
                if current_memory > self.memory_optimizer.memory_limit_mb * 0.8:
                    self.memory_optimizer.optimize_memory()

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            "cache_stats": self.cache.get_stats(),
            "memory_usage_mb": self.memory_optimizer.get_memory_usage_mb(),
            "memory_limit_mb": self.memory_optimizer.memory_limit_mb,
        }

        if self.enable_profiling:
            profiling_report = self.profiler.get_detailed_report()
            report.update(profiling_report)

        return report

    def start_session(self) -> None:
        """Start performance monitoring session."""
        if self.enable_profiling:
            self.profiler.start_profiling()

    def end_session(self) -> None:
        """End performance monitoring session."""
        if self.enable_profiling:
            self.profiler.end_profiling()

    async def cleanup(self) -> None:
        """Clean up performance manager resources."""
        self.cache.clear()
        self.memory_optimizer.optimize_memory()
        await self.parallel_processor.__aexit__(None, None, None)


def performance_decorator(stage_name: str, enable_caching: bool = True):
    """Decorator for automatic performance optimization."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # This would need access to a performance manager instance
            # For now, just add basic timing
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                execution_time = time.perf_counter() - start_time
                logger.debug(f"Stage {stage_name} executed in {execution_time:.3f}s")

        return wrapper

    return decorator


class PerformanceTestFramework:
    """Framework for performance testing and benchmarking."""

    def __init__(self):
        self.test_results: List[Dict[str, Any]] = []

    async def run_performance_test(
        self,
        test_name: str,
        pipeline_func: Callable,
        test_data: Any,
        iterations: int = 3,
    ) -> Dict[str, Any]:
        """Run performance test with multiple iterations."""
        results = []

        for i in range(iterations):
            profiler = PerformanceProfiler()
            profiler.start_profiling()

            start_time = time.perf_counter()
            try:
                await pipeline_func(test_data)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
            finally:
                execution_time = time.perf_counter() - start_time
                profiler.end_profiling()

            iteration_result = {
                "iteration": i + 1,
                "execution_time": execution_time,
                "success": success,
                "error": error,
                "metrics": profiler.get_metrics() if success else None,
            }

            results.append(iteration_result)

        # Calculate aggregate statistics
        successful_runs = [r for r in results if r["success"]]

        if successful_runs:
            execution_times = [r["execution_time"] for r in successful_runs]
            avg_time = sum(execution_times) / len(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)

            avg_performance_score = sum(
                r["metrics"].performance_score for r in successful_runs
            ) / len(successful_runs)
        else:
            avg_time = min_time = max_time = avg_performance_score = 0

        test_result = {
            "test_name": test_name,
            "iterations": iterations,
            "successful_runs": len(successful_runs),
            "success_rate": len(successful_runs) / iterations,
            "timing": {
                "average": avg_time,
                "minimum": min_time,
                "maximum": max_time,
                "meets_target": avg_time <= 10.0,
            },
            "average_performance_score": avg_performance_score,
            "individual_results": results,
        }

        self.test_results.append(test_result)
        return test_result

    def get_benchmark_report(self) -> Dict[str, Any]:
        """Get comprehensive benchmark report."""
        if not self.test_results:
            return {"error": "No test results available"}

        # Overall statistics
        all_times = []
        all_scores = []

        for test in self.test_results:
            if test["successful_runs"] > 0:
                all_times.append(test["timing"]["average"])
                all_scores.append(test["average_performance_score"])

        return {
            "summary": {
                "total_tests": len(self.test_results),
                "overall_avg_time": sum(all_times) / len(all_times) if all_times else 0,
                "overall_avg_score": sum(all_scores) / len(all_scores)
                if all_scores
                else 0,
                "tests_meeting_target": sum(
                    1 for test in self.test_results if test["timing"]["meets_target"]
                ),
            },
            "test_results": self.test_results,
        }


def create_optimized_pipeline_manager(
    config: Dict[str, Any],
) -> PipelinePerformanceManager:
    """Factory function to create optimized performance manager."""
    default_config = {
        "cache_size": 1000,
        "cache_ttl": 3600,
        "memory_limit_mb": 500,
        "max_workers": 4,
        "target_total_time": 10.0,
        "target_stage_time": 3.0,
        "enable_memory_tracking": True,
        "enable_profiling": True,
    }

    # Merge with provided config
    merged_config = {**default_config, **config}

    return PipelinePerformanceManager(merged_config)
