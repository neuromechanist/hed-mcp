"""Performance optimization utilities for BIDS Column Analysis Engine.

This module provides advanced performance optimization features including:
- Memory management and monitoring
- Chunked processing for large datasets
- Lazy loading strategies
- Parallel processing optimization
- Performance benchmarking and profiling
"""

import gc
import logging
import os
import time
import weakref
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple
from contextlib import contextmanager
import warnings

import pandas as pd

# Make psutil optional for graceful degradation
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False
    warnings.warn(
        "psutil not available - memory monitoring will be limited", UserWarning
    )


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""

    total_memory_gb: float
    available_memory_gb: float
    used_memory_gb: float
    memory_percent: float
    process_memory_mb: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""

    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: MemoryMetrics
    memory_after: MemoryMetrics
    items_processed: int = 0
    success: bool = True
    error_message: Optional[str] = None

    @property
    def throughput(self) -> float:
        """Items processed per second."""
        return self.items_processed / self.duration if self.duration > 0 else 0.0

    @property
    def memory_delta(self) -> float:
        """Memory usage change in MB."""
        return (
            self.memory_after.process_memory_mb - self.memory_before.process_memory_mb
        )


@dataclass
class ChunkProcessingConfig:
    """Configuration for chunked processing."""

    chunk_size: int = 10000  # Rows per chunk
    max_memory_usage_gb: float = 2.0  # Maximum memory usage before chunking
    overlap_rows: int = 0  # Overlap between chunks for temporal continuity
    preserve_order: bool = True  # Maintain original row order


@dataclass
class ParallelProcessingConfig:
    """Configuration for parallel processing."""

    max_workers: Optional[int] = None  # Auto-detect if None
    use_threads: bool = True  # Use threads vs processes
    chunk_size: int = 1  # Files per worker batch
    memory_limit_per_worker_gb: float = 1.0
    timeout_seconds: Optional[float] = None


class MemoryManager:
    """Advanced memory management for large dataset processing."""

    def __init__(
        self,
        memory_threshold: float = 0.8,
        cleanup_threshold: float = 0.9,
        gc_frequency: int = 100,
    ):
        """Initialize memory manager.

        Args:
            memory_threshold: Trigger optimization at this memory usage
            cleanup_threshold: Force garbage collection at this memory usage
            gc_frequency: Run GC every N operations
        """
        self.memory_threshold = memory_threshold
        self.cleanup_threshold = cleanup_threshold
        self.gc_frequency = gc_frequency
        self.operation_count = 0
        self.logger = logging.getLogger(__name__)
        self._weak_cache = weakref.WeakValueDictionary()

    def get_memory_metrics(self) -> MemoryMetrics:
        """Get current memory usage metrics."""
        if HAS_PSUTIL:
            memory = psutil.virtual_memory()
            process = psutil.Process()

            return MemoryMetrics(
                total_memory_gb=memory.total / (1024**3),
                available_memory_gb=memory.available / (1024**3),
                used_memory_gb=memory.used / (1024**3),
                memory_percent=memory.percent / 100.0,
                process_memory_mb=process.memory_info().rss / (1024**2),
            )
        else:
            return MemoryMetrics(
                total_memory_gb=0.0,
                available_memory_gb=0.0,
                used_memory_gb=0.0,
                memory_percent=0.0,
                process_memory_mb=0.0,
            )

    @contextmanager
    def memory_guard(self, operation_name: str = "operation"):
        """Context manager for memory-intensive operations."""
        before_metrics = self.get_memory_metrics()
        self.logger.debug(
            f"Starting {operation_name}, memory: {before_metrics.memory_percent:.1%}"
        )

        try:
            yield before_metrics
        finally:
            after_metrics = self.get_memory_metrics()
            delta = after_metrics.process_memory_mb - before_metrics.process_memory_mb

            self.logger.debug(
                f"Completed {operation_name}, memory: {after_metrics.memory_percent:.1%}, "
                f"delta: {delta:.1f}MB"
            )

            # Trigger cleanup if needed
            if after_metrics.memory_percent > self.cleanup_threshold:
                self.force_cleanup()
            elif after_metrics.memory_percent > self.memory_threshold:
                self.optimize_memory()

    def optimize_memory(self):
        """Optimize memory usage through various strategies."""
        self.logger.info("Optimizing memory usage")

        # Clear weak references that are no longer valid
        self._weak_cache.clear()

        # Suggest garbage collection
        if self.operation_count % self.gc_frequency == 0:
            collected = gc.collect()
            self.logger.debug(f"Garbage collection freed {collected} objects")

        self.operation_count += 1

    def force_cleanup(self):
        """Force aggressive memory cleanup."""
        self.logger.warning("Forcing memory cleanup due to high usage")

        # Clear all caches
        self._weak_cache.clear()

        # Force garbage collection
        for generation in range(3):
            collected = gc.collect()
            if collected == 0:
                break

        # Log final memory state
        metrics = self.get_memory_metrics()
        self.logger.info(f"Post-cleanup memory usage: {metrics.memory_percent:.1%}")

    def optimize_dataframe(
        self, df: pd.DataFrame, inplace: bool = False
    ) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        if not inplace:
            df = df.copy()

        initial_memory = df.memory_usage(deep=True).sum() / (1024**2)

        # Optimize object columns
        for col in df.select_dtypes(include=["object"]):
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:  # Convert to category if less than 50% unique
                df[col] = df[col].astype("category")

        # Optimize integer columns
        for col in df.select_dtypes(include=["int64"]):
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= 0:
                if col_max < 2**8:
                    df[col] = df[col].astype("uint8")
                elif col_max < 2**16:
                    df[col] = df[col].astype("uint16")
                elif col_max < 2**32:
                    df[col] = df[col].astype("uint32")
            else:
                if col_min >= -(2**7) and col_max < 2**7:
                    df[col] = df[col].astype("int8")
                elif col_min >= -(2**15) and col_max < 2**15:
                    df[col] = df[col].astype("int16")
                elif col_min >= -(2**31) and col_max < 2**31:
                    df[col] = df[col].astype("int32")

        # Optimize float columns
        for col in df.select_dtypes(include=["float64"]):
            df[col] = pd.to_numeric(df[col], downcast="float")

        final_memory = df.memory_usage(deep=True).sum() / (1024**2)
        reduction = (initial_memory - final_memory) / initial_memory

        self.logger.debug(
            f"DataFrame memory optimized: {initial_memory:.1f}MB -> {final_memory:.1f}MB "
            f"({reduction:.1%} reduction)"
        )

        return df


class ChunkedProcessor:
    """Chunked processing for large datasets."""

    def __init__(self, config: ChunkProcessingConfig, memory_manager: MemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)

    def process_file_chunks(
        self, file_path: Path, processor: Callable, **kwargs
    ) -> Generator[Any, None, None]:
        """Process a large file in chunks."""
        file_size_mb = file_path.stat().st_size / (1024**2)

        if file_size_mb < 50:  # Small file, process normally
            with self.memory_manager.memory_guard(f"processing {file_path.name}"):
                df = pd.read_csv(file_path, sep="\t")
                df = self.memory_manager.optimize_dataframe(df)
                yield processor(df, **kwargs)
            return

        self.logger.info(
            f"Processing large file {file_path.name} ({file_size_mb:.1f}MB) in chunks"
        )

        # Estimate chunk size based on memory constraints
        estimated_chunk_size = min(
            self.config.chunk_size,
            int(
                self.config.max_memory_usage_gb
                * 1024**3
                / (file_size_mb * 1024**2 / 1000)
            ),
        )

        chunk_results = []
        chunk_count = 0

        try:
            for chunk in pd.read_csv(
                file_path, sep="\t", chunksize=estimated_chunk_size
            ):
                chunk_count += 1

                with self.memory_manager.memory_guard(f"chunk {chunk_count}"):
                    # Optimize chunk memory usage
                    chunk = self.memory_manager.optimize_dataframe(chunk)

                    # Add overlap from previous chunk if needed
                    if self.config.overlap_rows > 0 and chunk_results:
                        previous_tail = chunk_results[-1].tail(self.config.overlap_rows)
                        chunk = pd.concat([previous_tail, chunk], ignore_index=True)

                    # Process chunk
                    result = processor(chunk, **kwargs)
                    chunk_results.append(result)

                    yield result

        except Exception as e:
            self.logger.error(
                f"Error processing chunk {chunk_count} of {file_path}: {e}"
            )
            raise

        self.logger.info(
            f"Completed processing {chunk_count} chunks from {file_path.name}"
        )

    def estimate_optimal_chunk_size(self, file_path: Path) -> int:
        """Estimate optimal chunk size based on file characteristics."""
        current_memory = self.memory_manager.get_memory_metrics()

        # Available memory for processing
        available_memory_gb = min(
            current_memory.available_memory_gb * 0.7,  # Use 70% of available
            self.config.max_memory_usage_gb,
        )

        # Estimate rows per MB (rough heuristic)
        rows_per_mb = 10000  # Conservative estimate

        # Calculate chunk size that fits in available memory
        max_chunk_size = int(available_memory_gb * 1024 * rows_per_mb)

        return min(max_chunk_size, self.config.chunk_size)


class LazyDataLoader:
    """Lazy loading strategies for efficient data access."""

    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)
        self._cache = {}
        self._access_times = {}
        self._max_cache_size = 10

    def lazy_load_files(
        self, file_paths: List[Path], preload_count: int = 5
    ) -> Iterator[Tuple[Path, pd.DataFrame]]:
        """Lazily load files with intelligent preloading."""

        # Sort files by size for better memory management
        file_info = [(path, path.stat().st_size) for path in file_paths]
        file_info.sort(key=lambda x: x[1])  # Sort by size, smallest first

        preloaded = {}

        for i, (file_path, file_size) in enumerate(file_info):
            # Preload next few files in background if they're small enough
            if i < len(file_info) - 1:
                next_files = file_info[i + 1 : i + preload_count + 1]
                for next_path, next_size in next_files:
                    if next_size < 10 * 1024 * 1024:  # Less than 10MB
                        if next_path not in preloaded:
                            try:
                                preloaded[next_path] = self._load_file_cached(next_path)
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to preload {next_path}: {e}"
                                )

            # Load current file
            if file_path in preloaded:
                df = preloaded.pop(file_path)
                self.logger.debug(f"Using preloaded data for {file_path.name}")
            else:
                df = self._load_file_cached(file_path)

            yield file_path, df

            # Clean up preloaded files that weren't used
            if len(preloaded) > preload_count:
                oldest_path = min(
                    preloaded.keys(), key=lambda p: self._access_times.get(p, 0)
                )
                del preloaded[oldest_path]

    def _load_file_cached(self, file_path: Path) -> pd.DataFrame:
        """Load file with caching."""
        cache_key = str(file_path)

        # Check cache first
        if cache_key in self._cache:
            self._access_times[cache_key] = time.time()
            self.logger.debug(f"Cache hit for {file_path.name}")
            return self._cache[cache_key]

        # Load file
        with self.memory_manager.memory_guard(f"loading {file_path.name}"):
            try:
                df = pd.read_csv(file_path, sep="\t")
                df = self.memory_manager.optimize_dataframe(df)
            except Exception as e:
                # Try CSV if TSV fails
                self.logger.warning(f"TSV load failed for {file_path}, trying CSV: {e}")
                df = pd.read_csv(file_path)
                df = self.memory_manager.optimize_dataframe(df)

        # Cache if small enough and cache has space
        file_size_mb = df.memory_usage(deep=True).sum() / (1024**2)
        if file_size_mb < 50 and len(self._cache) < self._max_cache_size:
            self._cache[cache_key] = df
            self._access_times[cache_key] = time.time()
            self.logger.debug(f"Cached {file_path.name} ({file_size_mb:.1f}MB)")
        elif len(self._cache) >= self._max_cache_size:
            # LRU eviction
            oldest_key = min(
                self._access_times.keys(), key=lambda k: self._access_times[k]
            )
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
            self._cache[cache_key] = df
            self._access_times[cache_key] = time.time()

        return df


class ParallelProcessor:
    """Advanced parallel processing for multiple files."""

    def __init__(self, config: ParallelProcessingConfig, memory_manager: MemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)

        # Auto-detect worker count if not specified
        if self.config.max_workers is None:
            self.config.max_workers = min(os.cpu_count() or 1, 8)

    async def process_files_parallel(
        self,
        file_paths: List[Path],
        processor: Callable,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> List[Any]:
        """Process files in parallel with intelligent load balancing."""

        # Group files by size for better load balancing
        file_groups = self._group_files_by_size(file_paths)

        if self.config.use_threads:
            executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        else:
            executor = ProcessPoolExecutor(max_workers=self.config.max_workers)

        results = []
        completed = 0

        try:
            # Submit tasks in size-balanced groups
            future_to_file = {}

            for file_group in file_groups:
                for file_path in file_group:
                    future = executor.submit(
                        self._process_single_file, file_path, processor, **kwargs
                    )
                    future_to_file[future] = file_path

            # Collect results as they complete
            for future in as_completed(
                future_to_file.keys(), timeout=self.config.timeout_seconds
            ):
                file_path = future_to_file[future]

                try:
                    result = future.result()
                    results.append(result)
                    completed += 1

                    if progress_callback:
                        await progress_callback(
                            completed, len(file_paths), file_path.name
                        )

                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {e}")
                    results.append(None)
                    completed += 1

        finally:
            executor.shutdown(wait=True)

        return results

    def _group_files_by_size(self, file_paths: List[Path]) -> List[List[Path]]:
        """Group files by size for balanced parallel processing."""

        # Get file sizes
        file_info = [(path, path.stat().st_size) for path in file_paths]
        file_info.sort(key=lambda x: x[1], reverse=True)  # Largest first

        # Create balanced groups
        groups = [[] for _ in range(self.config.max_workers)]
        group_sizes = [0] * self.config.max_workers

        for file_path, file_size in file_info:
            # Find group with smallest total size
            min_group = min(
                range(self.config.max_workers), key=lambda i: group_sizes[i]
            )

            groups[min_group].append(file_path)
            group_sizes[min_group] += file_size

        # Remove empty groups
        return [group for group in groups if group]

    def _process_single_file(
        self, file_path: Path, processor: Callable, **kwargs
    ) -> Any:
        """Process a single file with memory monitoring."""
        process_memory_manager = MemoryManager()

        with process_memory_manager.memory_guard(f"processing {file_path.name}"):
            try:
                df = pd.read_csv(file_path, sep="\t")
                df = process_memory_manager.optimize_dataframe(df)
                return processor(df, **kwargs)
            except Exception as e:
                # Try CSV if TSV fails
                self.logger.warning(f"TSV failed for {file_path}, trying CSV: {e}")
                df = pd.read_csv(file_path)
                df = process_memory_manager.optimize_dataframe(df)
                return processor(df, **kwargs)


class PerformanceBenchmark:
    """Performance benchmarking and profiling utilities."""

    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)
        self.metrics_history: List[PerformanceMetrics] = []

    @contextmanager
    def benchmark(self, operation_name: str, items_count: int = 0):
        """Context manager for benchmarking operations."""
        start_time = time.time()
        memory_before = self.memory_manager.get_memory_metrics()

        success = True
        error_message = None

        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            end_time = time.time()
            memory_after = self.memory_manager.get_memory_metrics()

            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                memory_before=memory_before,
                memory_after=memory_after,
                items_processed=items_count,
                success=success,
                error_message=error_message,
            )

            self.metrics_history.append(metrics)
            self._log_metrics(metrics)

    def _log_metrics(self, metrics: PerformanceMetrics):
        """Log performance metrics."""
        status = "✓" if metrics.success else "✗"

        self.logger.info(
            f"{status} {metrics.operation_name}: "
            f"{metrics.duration:.2f}s, "
            f"{metrics.throughput:.1f} items/s, "
            f"memory Δ: {metrics.memory_delta:+.1f}MB"
        )

        if not metrics.success:
            self.logger.error(f"Operation failed: {metrics.error_message}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all performance metrics."""
        if not self.metrics_history:
            return {"message": "No performance data available"}

        successful_metrics = [m for m in self.metrics_history if m.success]

        total_duration = sum(m.duration for m in successful_metrics)
        total_items = sum(m.items_processed for m in successful_metrics)

        return {
            "total_operations": len(self.metrics_history),
            "successful_operations": len(successful_metrics),
            "total_duration": total_duration,
            "total_items_processed": total_items,
            "average_throughput": total_items / total_duration
            if total_duration > 0
            else 0,
            "operations": [
                {
                    "name": m.operation_name,
                    "duration": m.duration,
                    "throughput": m.throughput,
                    "memory_delta": m.memory_delta,
                    "success": m.success,
                }
                for m in self.metrics_history[-10:]  # Last 10 operations
            ],
        }

    def generate_performance_report(self) -> str:
        """Generate a human-readable performance report."""
        summary = self.get_performance_summary()

        if "message" in summary:
            return summary["message"]

        report = [
            "Performance Report",
            "=" * 50,
            f"Total Operations: {summary['total_operations']}",
            f"Successful: {summary['successful_operations']}",
            f"Total Duration: {summary['total_duration']:.2f}s",
            f"Items Processed: {summary['total_items_processed']}",
            f"Average Throughput: {summary['average_throughput']:.1f} items/s",
            "",
            "Recent Operations:",
            "-" * 30,
        ]

        for op in summary["operations"]:
            status = "✓" if op["success"] else "✗"
            report.append(
                f"{status} {op['name']}: {op['duration']:.2f}s, "
                f"{op['throughput']:.1f} items/s, "
                f"memory Δ: {op['memory_delta']:+.1f}MB"
            )

        return "\n".join(report)


# Utility functions for easy integration
def create_optimized_config(
    memory_limit_gb: float = 2.0,
    max_workers: Optional[int] = None,
    chunk_size: int = 10000,
) -> Tuple[ChunkProcessingConfig, ParallelProcessingConfig]:
    """Create optimized configuration based on system resources."""

    # Get system memory info
    if HAS_PSUTIL:
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)
    else:
        available_memory_gb = 0.0

    # Adjust memory limit based on available memory
    safe_memory_limit = min(memory_limit_gb, available_memory_gb * 0.6)

    chunk_config = ChunkProcessingConfig(
        chunk_size=chunk_size,
        max_memory_usage_gb=safe_memory_limit,
        overlap_rows=100,  # Small overlap for temporal data
    )

    parallel_config = ParallelProcessingConfig(
        max_workers=max_workers or min(os.cpu_count() or 1, 8),
        memory_limit_per_worker_gb=safe_memory_limit / (max_workers or 4),
        timeout_seconds=300,  # 5 minute timeout
    )

    return chunk_config, parallel_config


def optimize_for_large_datasets(
    enable_chunking: bool = True,
    enable_parallel: bool = True,
    memory_aggressive: bool = False,
) -> Dict[str, Any]:
    """Get optimization settings for large datasets."""

    settings = {
        "memory_threshold": 0.6 if memory_aggressive else 0.8,
        "cleanup_threshold": 0.8 if memory_aggressive else 0.9,
        "gc_frequency": 50 if memory_aggressive else 100,
        "enable_chunking": enable_chunking,
        "enable_parallel": enable_parallel,
        "cache_size": 5 if memory_aggressive else 10,
    }

    return settings
