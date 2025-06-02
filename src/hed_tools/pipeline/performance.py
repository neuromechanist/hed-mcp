"""Performance optimization module for the HED sidecar generation pipeline.

This module provides comprehensive performance optimization capabilities:
- Profiling and bottleneck detection
- Parallel processing for pipeline stages
- Intelligent caching mechanisms
- Memory optimization strategies
- Performance monitoring and metrics
- Batch processing optimizations
"""

import asyncio
import time
import threading
import gc
import logging
import hashlib
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from functools import wraps
import cProfile
import tracemalloc
from collections import defaultdict, OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance measurement data."""

    stage_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: int
    memory_after: int
    memory_peak: int
    cpu_percent: float
    success: bool
    errors: List[str] = field(default_factory=list)

    @property
    def memory_delta(self) -> int:
        """Memory change during execution."""
        return self.memory_after - self.memory_before


@dataclass
class CacheStats:
    """Cache performance statistics."""

    hits: int = 0
    misses: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class PerformanceProfiler:
    """Advanced profiler for pipeline performance analysis."""

    def __init__(self, enable_memory_tracking: bool = True):
        self.enable_memory_tracking = enable_memory_tracking
        self.stage_metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.profiler_data: Dict[str, cProfile.Profile] = {}
        self.memory_snapshots: List[Tuple[float, int]] = []

        if self.enable_memory_tracking:
            tracemalloc.start()

    @contextmanager
    def profile_stage(self, stage_name: str):
        """Context manager for profiling pipeline stages."""
        # Initialize measurements
        start_time = time.perf_counter()
        process = psutil.Process()
        memory_before = process.memory_info().rss
        cpu_before = process.cpu_percent()

        # Start code profiling
        profiler = cProfile.Profile()
        profiler.enable()

        # Memory tracking
        if self.enable_memory_tracking:
            snapshot_before = tracemalloc.take_snapshot()

        success = True
        errors = []

        try:
            yield
        except Exception as e:
            success = False
            errors.append(str(e))
            raise
        finally:
            # Stop profiling
            profiler.disable()
            self.profiler_data[stage_name] = profiler

            # Final measurements
            end_time = time.perf_counter()
            memory_after = process.memory_info().rss
            cpu_after = process.cpu_percent()

            # Memory peak detection
            memory_peak = memory_after
            if self.enable_memory_tracking:
                snapshot_after = tracemalloc.take_snapshot()
                top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
                if top_stats:
                    memory_peak = max(stat.size for stat in top_stats[:10])

            # Store metrics
            metrics = PerformanceMetrics(
                stage_name=stage_name,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=memory_peak,
                cpu_percent=(cpu_before + cpu_after) / 2,
                success=success,
                errors=errors,
            )

            self.stage_metrics[stage_name].append(metrics)

            # Log performance summary
            self._log_stage_performance(metrics)

    def _log_stage_performance(self, metrics: PerformanceMetrics):
        """Log stage performance summary."""
        memory_mb = metrics.memory_delta / (1024 * 1024)

        if metrics.success:
            logger.info(
                f"Stage {metrics.stage_name}: {metrics.duration:.2f}s, "
                f"memory: {memory_mb:+.1f}MB, CPU: {metrics.cpu_percent:.1f}%"
            )
        else:
            logger.warning(
                f"Stage {metrics.stage_name} FAILED: {metrics.duration:.2f}s, "
                f"errors: {len(metrics.errors)}"
            )

    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """Identify performance bottlenecks."""
        analysis = {
            "slowest_stages": [],
            "memory_intensive_stages": [],
            "cpu_intensive_stages": [],
            "failure_prone_stages": [],
            "total_pipeline_time": 0.0,
            "recommendations": [],
        }

        # Analyze each stage
        for stage_name, metrics_list in self.stage_metrics.items():
            if not metrics_list:
                continue

            avg_duration = sum(m.duration for m in metrics_list) / len(metrics_list)
            avg_memory = sum(m.memory_delta for m in metrics_list) / len(metrics_list)
            avg_cpu = sum(m.cpu_percent for m in metrics_list) / len(metrics_list)
            failure_rate = sum(1 for m in metrics_list if not m.success) / len(
                metrics_list
            )

            analysis["slowest_stages"].append((stage_name, avg_duration))
            analysis["memory_intensive_stages"].append((stage_name, avg_memory))
            analysis["cpu_intensive_stages"].append((stage_name, avg_cpu))

            if failure_rate > 0:
                analysis["failure_prone_stages"].append((stage_name, failure_rate))

        # Sort by performance impact
        analysis["slowest_stages"].sort(key=lambda x: x[1], reverse=True)
        analysis["memory_intensive_stages"].sort(key=lambda x: x[1], reverse=True)
        analysis["cpu_intensive_stages"].sort(key=lambda x: x[1], reverse=True)

        analysis["total_pipeline_time"] = sum(
            metrics[1] for metrics in analysis["slowest_stages"]
        )

        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)

        return analysis

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        if analysis["total_pipeline_time"] > 10.0:
            recommendations.append(
                f"Pipeline exceeds 10s target ({analysis['total_pipeline_time']:.1f}s). "
                "Consider parallel processing or caching."
            )

        if analysis["slowest_stages"]:
            slowest = analysis["slowest_stages"][0]
            if slowest[1] > 3.0:
                recommendations.append(
                    f"Stage '{slowest[0]}' is bottleneck ({slowest[1]:.1f}s). "
                    "Consider optimization or parallelization."
                )

        if analysis["memory_intensive_stages"]:
            memory_heavy = analysis["memory_intensive_stages"][0]
            if memory_heavy[1] > 100 * 1024 * 1024:  # 100MB
                recommendations.append(
                    f"Stage '{memory_heavy[0]}' uses excessive memory. "
                    "Consider batch processing or streaming."
                )

        return recommendations


class AdvancedCache:
    """High-performance caching system with TTL and LRU eviction."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        self.stats = CacheStats(max_size=max_size)

    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache."""
        with self._lock:
            current_time = time.time()

            # Check if key exists and is not expired
            if key in self._cache:
                if current_time - self._timestamps[key] < self.ttl_seconds:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self.stats.hits += 1
                    return self._cache[key]
                else:
                    # Expired, remove
                    del self._cache[key]
                    del self._timestamps[key]

            self.stats.misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        """Store item in cache."""
        with self._lock:
            current_time = time.time()

            # Remove if already exists
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]

            # Evict oldest items if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]

            # Add new item
            self._cache[key] = value
            self._timestamps[key] = current_time
            self.stats.size = len(self._cache)

    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self.stats.size = 0

    def get_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        return self.stats


class ParallelProcessor:
    """Parallel processing manager for pipeline stages."""

    def __init__(self, max_workers: int = None, use_processes: bool = False):
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self.use_processes = use_processes
        self.executor_class = (
            ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        )

    async def execute_parallel_stages(
        self,
        stage_functions: List[Callable],
        stage_contexts: List[Any],
        timeout: float = 30.0,
    ) -> List[Any]:
        """Execute multiple independent stages in parallel."""

        loop = asyncio.get_event_loop()

        with self.executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_stage = {
                loop.run_in_executor(executor, func, context): (func, context)
                for func, context in zip(stage_functions, stage_contexts)
            }

            # Collect results with timeout
            results = []
            try:
                completed_futures = await asyncio.wait_for(
                    asyncio.gather(*future_to_stage.keys(), return_exceptions=True),
                    timeout=timeout,
                )

                for future_result in completed_futures:
                    if isinstance(future_result, Exception):
                        logger.error(f"Parallel stage failed: {future_result}")
                        results.append(None)
                    else:
                        results.append(future_result)

            except asyncio.TimeoutError:
                logger.error(f"Parallel execution timed out after {timeout}s")
                # Cancel remaining futures
                for future in future_to_stage:
                    future.cancel()
                raise

        return results

    async def process_batch_parallel(
        self,
        items: List[Any],
        processor_func: Callable,
        batch_size: int = 10,
        max_concurrent_batches: int = 3,
    ) -> List[Any]:
        """Process items in parallel batches."""

        # Split into batches
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

        # Semaphore to limit concurrent batches
        semaphore = asyncio.Semaphore(max_concurrent_batches)

        async def process_batch(batch):
            async with semaphore:
                loop = asyncio.get_event_loop()
                with self.executor_class(max_workers=len(batch)) as executor:
                    tasks = [
                        loop.run_in_executor(executor, processor_func, item)
                        for item in batch
                    ]
                    return await asyncio.gather(*tasks, return_exceptions=True)

        # Process all batches
        batch_tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks)

        # Flatten results
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)

        return results


class MemoryOptimizer:
    """Memory optimization utilities for large dataset processing."""

    def __init__(self, memory_limit_mb: int = 500):
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.process = psutil.Process()

    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return self.process.memory_info().rss

    def is_memory_pressure(self) -> bool:
        """Check if memory usage is approaching limit."""
        return self.get_memory_usage() > self.memory_limit_bytes * 0.8

    @contextmanager
    def memory_management(self):
        """Context manager for automatic memory management."""
        initial_memory = self.get_memory_usage()

        try:
            yield
        finally:
            # Force garbage collection
            gc.collect()

            final_memory = self.get_memory_usage()
            memory_delta = final_memory - initial_memory

            if memory_delta > 50 * 1024 * 1024:  # 50MB increase
                logger.warning(
                    f"High memory usage detected: +{memory_delta / (1024*1024):.1f}MB"
                )

    def optimize_dataframe_memory(self, df) -> Any:
        """Optimize pandas DataFrame memory usage."""
        try:
            import pandas as pd

            if not isinstance(df, pd.DataFrame):
                return df

            # Optimize numeric columns
            for col in df.select_dtypes(include=["int"]).columns:
                col_min = df[col].min()
                col_max = df[col].max()

                if col_min >= -128 and col_max <= 127:
                    df[col] = df[col].astype("int8")
                elif col_min >= -32768 and col_max <= 32767:
                    df[col] = df[col].astype("int16")
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df[col] = df[col].astype("int32")

            # Optimize float columns
            for col in df.select_dtypes(include=["float"]).columns:
                df[col] = pd.to_numeric(df[col], downcast="float")

            # Convert object columns to category if beneficial
            for col in df.select_dtypes(include=["object"]).columns:
                if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype("category")

            return df

        except ImportError:
            logger.warning("Pandas not available for DataFrame optimization")
            return df


class PipelinePerformanceManager:
    """Comprehensive performance management for the pipeline."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Initialize components
        self.profiler = PerformanceProfiler(
            enable_memory_tracking=self.config.get("enable_memory_tracking", True)
        )

        self.cache = AdvancedCache(
            max_size=self.config.get("cache_size", 1000),
            ttl_seconds=self.config.get("cache_ttl", 3600),
        )

        self.parallel_processor = ParallelProcessor(
            max_workers=self.config.get("max_workers"),
            use_processes=self.config.get("use_processes", False),
        )

        self.memory_optimizer = MemoryOptimizer(
            memory_limit_mb=self.config.get("memory_limit_mb", 500)
        )

        # Performance targets
        self.target_total_time = self.config.get("target_total_time", 10.0)
        self.target_stage_time = self.config.get("target_stage_time", 3.0)

    @contextmanager
    def performance_context(self, stage_name: str):
        """Combined performance management context."""
        with self.profiler.profile_stage(stage_name):
            with self.memory_optimizer.memory_management():
                yield

    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if available."""
        return self.cache.get(cache_key)

    def cache_result(self, cache_key: str, result: Any) -> None:
        """Cache computation result."""
        self.cache.put(cache_key, result)

    def generate_cache_key(self, *args, **kwargs) -> str:
        """Generate deterministic cache key from arguments."""
        key_data = {"args": args, "kwargs": sorted(kwargs.items())}
        return hashlib.md5(str(key_data).encode()).hexdigest()

    async def execute_optimized_stage(
        self,
        stage_name: str,
        stage_func: Callable,
        stage_args: Tuple,
        stage_kwargs: Dict[str, Any],
        enable_caching: bool = True,
    ) -> Any:
        """Execute pipeline stage with all optimizations."""

        # Generate cache key if caching enabled
        cache_key = None
        if enable_caching:
            cache_key = self.generate_cache_key(stage_name, *stage_args, **stage_kwargs)
            cached_result = self.get_cached_result(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for stage {stage_name}")
                return cached_result

        # Execute with performance monitoring
        with self.performance_context(stage_name):
            result = await stage_func(*stage_args, **stage_kwargs)

            # Cache successful results
            if enable_caching and cache_key and result is not None:
                self.cache_result(cache_key, result)
                logger.debug(f"Cached result for stage {stage_name}")

            return result

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        bottleneck_analysis = self.profiler.get_bottleneck_analysis()
        cache_stats = self.cache.get_stats()
        memory_usage = self.memory_optimizer.get_memory_usage()

        # Calculate compliance with performance targets
        total_time = bottleneck_analysis["total_pipeline_time"]
        meets_target = total_time <= self.target_total_time

        report = {
            "performance_summary": {
                "total_execution_time": total_time,
                "target_time": self.target_total_time,
                "meets_target": meets_target,
                "performance_score": min(
                    100, (self.target_total_time / max(total_time, 0.1)) * 100
                ),
            },
            "bottleneck_analysis": bottleneck_analysis,
            "cache_performance": {
                "hit_rate": cache_stats.hit_rate,
                "cache_size": cache_stats.size,
                "total_requests": cache_stats.hits + cache_stats.misses,
            },
            "memory_usage": {
                "current_mb": memory_usage / (1024 * 1024),
                "limit_mb": self.memory_optimizer.memory_limit_bytes / (1024 * 1024),
                "utilization_percent": (
                    memory_usage / self.memory_optimizer.memory_limit_bytes
                )
                * 100,
            },
            "optimization_recommendations": bottleneck_analysis["recommendations"],
        }

        return report


def performance_decorator(manager: PipelinePerformanceManager, stage_name: str):
    """Decorator for automatic performance optimization of pipeline stages."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await manager.execute_optimized_stage(
                stage_name=stage_name,
                stage_func=func,
                stage_args=args,
                stage_kwargs=kwargs,
                enable_caching=kwargs.pop("_enable_caching", True),
            )

        return wrapper

    return decorator


class PerformanceTestFramework:
    """Framework for testing and benchmarking pipeline performance."""

    def __init__(self, pipeline_manager: PipelinePerformanceManager):
        self.manager = pipeline_manager
        self.test_results: List[Dict[str, Any]] = []

    async def benchmark_pipeline(
        self, test_cases: List[Dict[str, Any]], iterations: int = 3
    ) -> Dict[str, Any]:
        """Benchmark pipeline with various test cases."""

        benchmark_results = {
            "test_cases": [],
            "overall_performance": {},
            "recommendations": [],
        }

        for test_case in test_cases:
            case_name = test_case["name"]
            case_data = test_case["data"]

            logger.info(f"Running benchmark: {case_name}")

            case_times = []
            for iteration in range(iterations):
                start_time = time.perf_counter()

                try:
                    # Execute test case (would be actual pipeline execution)
                    await self._execute_test_case(case_data)
                    execution_time = time.perf_counter() - start_time
                    case_times.append(execution_time)

                except Exception as e:
                    logger.error(f"Test case {case_name} failed: {e}")
                    case_times.append(float("inf"))

            # Calculate statistics
            avg_time = sum(t for t in case_times if t != float("inf")) / len(
                [t for t in case_times if t != float("inf")]
            )
            min_time = min(t for t in case_times if t != float("inf"))
            max_time = max(t for t in case_times if t != float("inf"))

            benchmark_results["test_cases"].append(
                {
                    "name": case_name,
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "meets_target": avg_time <= self.manager.target_total_time,
                    "iterations": iterations,
                }
            )

        # Generate overall assessment
        all_times = [case["avg_time"] for case in benchmark_results["test_cases"]]
        benchmark_results["overall_performance"] = {
            "average_execution_time": sum(all_times) / len(all_times),
            "worst_case_time": max(all_times),
            "best_case_time": min(all_times),
            "target_compliance_rate": sum(
                1 for case in benchmark_results["test_cases"] if case["meets_target"]
            )
            / len(benchmark_results["test_cases"]),
        }

        return benchmark_results

    async def _execute_test_case(self, test_data: Dict[str, Any]) -> None:
        """Execute a single test case (placeholder for actual pipeline execution)."""
        # Simulate pipeline execution
        await asyncio.sleep(0.1)  # Placeholder
        pass


# Performance optimization utilities
def create_optimized_pipeline_manager(
    config: Dict[str, Any] = None,
) -> PipelinePerformanceManager:
    """Factory function to create optimally configured performance manager."""
    default_config = {
        "enable_memory_tracking": True,
        "cache_size": 1000,
        "cache_ttl": 3600,
        "max_workers": None,  # Auto-detect
        "use_processes": False,
        "memory_limit_mb": 500,
        "target_total_time": 10.0,
        "target_stage_time": 3.0,
    }

    if config:
        default_config.update(config)

    return PipelinePerformanceManager(default_config)
