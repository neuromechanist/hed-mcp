"""Tests for the Performance Optimizer module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import time

from hed_tools.tools.performance_optimizer import (
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


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "onset": np.random.random(1000),
            "duration": np.ones(1000),
            "trial_type": ["go", "stop"] * 500,
            "response_time": np.random.normal(0.5, 0.1, 1000),
            "accuracy": np.random.choice([0, 1], 1000),
            "stimulus": ["red", "blue", "green"] * 333 + ["red"],
        }
    )


@pytest.fixture
def memory_manager():
    """Create a memory manager for testing."""
    return MemoryManager(memory_threshold=0.8, cleanup_threshold=0.9, gc_frequency=10)


@pytest.fixture
def chunk_config():
    """Create chunk processing configuration."""
    return ChunkProcessingConfig(
        chunk_size=100, max_memory_usage_gb=1.0, overlap_rows=10
    )


@pytest.fixture
def parallel_config():
    """Create parallel processing configuration."""
    return ParallelProcessingConfig(max_workers=2, use_threads=True, timeout_seconds=30)


class TestMemoryManager:
    """Test the MemoryManager class."""

    def test_initialization(self, memory_manager):
        """Test memory manager initialization."""
        assert memory_manager.memory_threshold == 0.8
        assert memory_manager.cleanup_threshold == 0.9
        assert memory_manager.gc_frequency == 10
        assert memory_manager.operation_count == 0

    def test_get_memory_metrics(self, memory_manager):
        """Test memory metrics collection."""
        metrics = memory_manager.get_memory_metrics()

        assert isinstance(metrics, MemoryMetrics)
        assert metrics.total_memory_gb > 0
        assert metrics.available_memory_gb >= 0
        assert metrics.used_memory_gb >= 0
        assert 0 <= metrics.memory_percent <= 1
        assert metrics.process_memory_mb > 0
        assert metrics.timestamp > 0

    def test_memory_guard_context_manager(self, memory_manager):
        """Test memory guard context manager."""
        with memory_manager.memory_guard("test_operation") as before_metrics:
            assert isinstance(before_metrics, MemoryMetrics)
            # Simulate some memory usage
            _ = list(range(1000))  # Use underscore to indicate intentionally unused

        # Should complete without error
        assert True

    def test_optimize_dataframe(self, memory_manager, sample_dataframe):
        """Test DataFrame optimization."""
        initial_memory = sample_dataframe.memory_usage(deep=True).sum()

        # Test in-place optimization
        optimized_df = memory_manager.optimize_dataframe(sample_dataframe, inplace=True)
        final_memory = optimized_df.memory_usage(deep=True).sum()

        assert (
            optimized_df is sample_dataframe
        )  # Should be same object for inplace=True
        assert final_memory <= initial_memory  # Should not increase memory

    def test_optimize_dataframe_copy(self, memory_manager, sample_dataframe):
        """Test DataFrame optimization with copy."""
        optimized_df = memory_manager.optimize_dataframe(
            sample_dataframe, inplace=False
        )

        assert optimized_df is not sample_dataframe  # Should be different object
        assert len(optimized_df) == len(sample_dataframe)
        assert list(optimized_df.columns) == list(sample_dataframe.columns)

    def test_force_cleanup(self, memory_manager):
        """Test force cleanup functionality."""
        # Should run without error
        memory_manager.force_cleanup()
        assert True

    def test_optimize_memory(self, memory_manager):
        """Test memory optimization."""
        initial_count = memory_manager.operation_count
        memory_manager.optimize_memory()
        assert memory_manager.operation_count == initial_count + 1


class TestChunkedProcessor:
    """Test the ChunkedProcessor class."""

    def test_initialization(self, chunk_config, memory_manager):
        """Test chunked processor initialization."""
        processor = ChunkedProcessor(chunk_config, memory_manager)

        assert processor.config == chunk_config
        assert processor.memory_manager == memory_manager

    def test_process_small_file_chunks(self, chunk_config, memory_manager, tmp_path):
        """Test processing small file (should not chunk)."""
        processor = ChunkedProcessor(chunk_config, memory_manager)

        # Create small test file
        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        test_file = tmp_path / "small_test.tsv"
        test_data.to_csv(test_file, sep="\t", index=False)

        def dummy_processor(df, **kwargs):
            return {"processed": len(df), "columns": list(df.columns)}

        results = list(processor.process_file_chunks(test_file, dummy_processor))

        assert len(results) == 1  # Should process as single chunk
        assert results[0]["processed"] == 3
        assert "col1" in results[0]["columns"]

    def test_estimate_optimal_chunk_size(self, chunk_config, memory_manager, tmp_path):
        """Test chunk size estimation."""
        processor = ChunkedProcessor(chunk_config, memory_manager)

        # Create test file
        test_file = tmp_path / "test.tsv"
        test_file.write_text("col1\tcol2\n1\ta\n2\tb\n")

        chunk_size = processor.estimate_optimal_chunk_size(test_file)

        assert isinstance(chunk_size, int)
        assert chunk_size > 0
        assert chunk_size <= chunk_config.chunk_size


class TestLazyDataLoader:
    """Test the LazyDataLoader class."""

    def test_initialization(self, memory_manager):
        """Test lazy data loader initialization."""
        loader = LazyDataLoader(memory_manager)

        assert loader.memory_manager == memory_manager
        assert isinstance(loader._cache, dict)
        assert isinstance(loader._access_times, dict)
        assert loader._max_cache_size == 10

    def test_load_file_cached(self, memory_manager, tmp_path):
        """Test cached file loading."""
        loader = LazyDataLoader(memory_manager)

        # Create test file
        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        test_file = tmp_path / "test.tsv"
        test_data.to_csv(test_file, sep="\t", index=False)

        # First load should cache
        df1 = loader._load_file_cached(test_file)
        assert len(df1) == 3

        # Second load should use cache
        df2 = loader._load_file_cached(test_file)
        assert len(df2) == 3

    def test_lazy_load_files(self, memory_manager, tmp_path):
        """Test lazy loading of multiple files."""
        loader = LazyDataLoader(memory_manager)

        # Create multiple test files
        files = []
        for i in range(3):
            test_data = pd.DataFrame({"col1": [i, i + 1], "col2": ["a", "b"]})
            test_file = tmp_path / f"test_{i}.tsv"
            test_data.to_csv(test_file, sep="\t", index=False)
            files.append(test_file)

        # Load files lazily
        results = list(loader.lazy_load_files(files, preload_count=2))

        assert len(results) == 3
        for file_path, df in results:
            assert isinstance(file_path, Path)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2


class TestParallelProcessor:
    """Test the ParallelProcessor class."""

    def test_initialization(self, parallel_config, memory_manager):
        """Test parallel processor initialization."""
        processor = ParallelProcessor(parallel_config, memory_manager)

        assert processor.config == parallel_config
        assert processor.memory_manager == memory_manager

    def test_group_files_by_size(self, parallel_config, memory_manager, tmp_path):
        """Test file grouping by size."""
        processor = ParallelProcessor(parallel_config, memory_manager)

        # Create files of different sizes
        files = []
        for i, size in enumerate([100, 200, 50, 300]):
            test_file = tmp_path / f"test_{i}.tsv"
            test_file.write_text("x" * size)
            files.append(test_file)

        groups = processor._group_files_by_size(files)

        assert len(groups) <= processor.config.max_workers
        # All files should be distributed
        total_files = sum(len(group) for group in groups)
        assert total_files == len(files)

    def test_process_single_file(self, parallel_config, memory_manager, tmp_path):
        """Test single file processing."""
        processor = ParallelProcessor(parallel_config, memory_manager)

        # Create test file
        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        test_file = tmp_path / "test.tsv"
        test_data.to_csv(test_file, sep="\t", index=False)

        def dummy_processor(df, **kwargs):
            return {"processed": len(df)}

        result = processor._process_single_file(test_file, dummy_processor)

        assert result["processed"] == 3


class TestPerformanceBenchmark:
    """Test the PerformanceBenchmark class."""

    def test_initialization(self, memory_manager):
        """Test performance benchmark initialization."""
        benchmark = PerformanceBenchmark(memory_manager)

        assert benchmark.memory_manager == memory_manager
        assert isinstance(benchmark.metrics_history, list)
        assert len(benchmark.metrics_history) == 0

    def test_benchmark_context_manager(self, memory_manager):
        """Test benchmark context manager."""
        benchmark = PerformanceBenchmark(memory_manager)

        with benchmark.benchmark("test_operation", 10):
            time.sleep(0.01)  # Small delay to measure

        assert len(benchmark.metrics_history) == 1
        metrics = benchmark.metrics_history[0]

        assert metrics.operation_name == "test_operation"
        assert metrics.items_processed == 10
        assert metrics.duration > 0
        assert metrics.success is True

    def test_benchmark_with_exception(self, memory_manager):
        """Test benchmark with exception handling."""
        benchmark = PerformanceBenchmark(memory_manager)

        with pytest.raises(ValueError):
            with benchmark.benchmark("failing_operation"):
                raise ValueError("Test error")

        assert len(benchmark.metrics_history) == 1
        metrics = benchmark.metrics_history[0]

        assert metrics.success is False
        assert "Test error" in metrics.error_message

    def test_get_performance_summary(self, memory_manager):
        """Test performance summary generation."""
        benchmark = PerformanceBenchmark(memory_manager)

        # Add some metrics
        with benchmark.benchmark("op1", 5):
            time.sleep(0.01)

        with benchmark.benchmark("op2", 10):
            time.sleep(0.01)

        summary = benchmark.get_performance_summary()

        assert summary["total_operations"] == 2
        assert summary["successful_operations"] == 2
        assert summary["total_items_processed"] == 15
        assert summary["total_duration"] > 0
        assert len(summary["operations"]) == 2

    def test_generate_performance_report(self, memory_manager):
        """Test performance report generation."""
        benchmark = PerformanceBenchmark(memory_manager)

        with benchmark.benchmark("test_op", 1):
            pass

        report = benchmark.generate_performance_report()

        assert isinstance(report, str)
        assert "Performance Report" in report
        assert "test_op" in report


class TestMetricsDataClasses:
    """Test the metrics data classes."""

    def test_memory_metrics(self):
        """Test MemoryMetrics dataclass."""
        metrics = MemoryMetrics(
            total_memory_gb=16.0,
            available_memory_gb=8.0,
            used_memory_gb=8.0,
            memory_percent=0.5,
            process_memory_mb=100.0,
        )

        assert metrics.total_memory_gb == 16.0
        assert metrics.memory_percent == 0.5
        assert metrics.timestamp > 0

    def test_performance_metrics(self):
        """Test PerformanceMetrics dataclass."""
        before_memory = MemoryMetrics(16.0, 8.0, 8.0, 0.5, 100.0)
        after_memory = MemoryMetrics(16.0, 7.0, 9.0, 0.56, 120.0)

        metrics = PerformanceMetrics(
            operation_name="test",
            start_time=1000.0,
            end_time=1001.0,
            duration=1.0,
            memory_before=before_memory,
            memory_after=after_memory,
            items_processed=10,
        )

        assert metrics.throughput == 10.0  # 10 items / 1 second
        assert metrics.memory_delta == 20.0  # 120 - 100 MB


class TestConfigurationDataClasses:
    """Test configuration data classes."""

    def test_chunk_processing_config(self):
        """Test ChunkProcessingConfig dataclass."""
        config = ChunkProcessingConfig(
            chunk_size=5000, max_memory_usage_gb=1.5, overlap_rows=50
        )

        assert config.chunk_size == 5000
        assert config.max_memory_usage_gb == 1.5
        assert config.overlap_rows == 50
        assert config.preserve_order is True

    def test_parallel_processing_config(self):
        """Test ParallelProcessingConfig dataclass."""
        config = ParallelProcessingConfig(
            max_workers=4, use_threads=False, timeout_seconds=60
        )

        assert config.max_workers == 4
        assert config.use_threads is False
        assert config.timeout_seconds == 60


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_optimized_config(self):
        """Test optimized configuration creation."""
        chunk_config, parallel_config = create_optimized_config(
            memory_limit_gb=2.0, max_workers=4, chunk_size=8000
        )

        assert isinstance(chunk_config, ChunkProcessingConfig)
        assert isinstance(parallel_config, ParallelProcessingConfig)
        assert chunk_config.chunk_size == 8000
        assert parallel_config.max_workers == 4

    def test_optimize_for_large_datasets(self):
        """Test large dataset optimization settings."""
        settings = optimize_for_large_datasets(
            enable_chunking=True, enable_parallel=True, memory_aggressive=True
        )

        assert isinstance(settings, dict)
        assert settings["enable_chunking"] is True
        assert settings["enable_parallel"] is True
        assert settings["memory_threshold"] == 0.6  # aggressive setting

    def test_optimize_for_large_datasets_conservative(self):
        """Test conservative optimization settings."""
        settings = optimize_for_large_datasets(memory_aggressive=False)

        assert settings["memory_threshold"] == 0.8  # conservative setting
        assert settings["gc_frequency"] == 100


class TestIntegration:
    """Integration tests for performance optimization components."""

    def test_memory_manager_with_chunked_processor(self, tmp_path):
        """Test memory manager integration with chunked processor."""
        memory_manager = MemoryManager()
        config = ChunkProcessingConfig(chunk_size=50)
        processor = ChunkedProcessor(config, memory_manager)

        # Create test file
        large_data = pd.DataFrame({"col1": range(200), "col2": ["value"] * 200})
        test_file = tmp_path / "large_test.tsv"
        large_data.to_csv(test_file, sep="\t", index=False)

        def counter_processor(df, **kwargs):
            return {"count": len(df)}

        results = list(processor.process_file_chunks(test_file, counter_processor))

        # Should process successfully
        assert len(results) >= 1
        total_processed = sum(r["count"] for r in results)
        assert total_processed == 200

    def test_performance_benchmark_with_chunked_processing(self, tmp_path):
        """Test performance benchmarking with chunked processing."""
        memory_manager = MemoryManager()
        benchmark = PerformanceBenchmark(memory_manager)
        config = ChunkProcessingConfig(chunk_size=30)
        processor = ChunkedProcessor(config, memory_manager)

        # Create test file
        test_data = pd.DataFrame({"col1": range(100), "col2": ["test"] * 100})
        test_file = tmp_path / "benchmark_test.tsv"
        test_data.to_csv(test_file, sep="\t", index=False)

        def timed_processor(df, **kwargs):
            return {"processed": len(df)}

        with benchmark.benchmark("chunked_processing", 100):
            _ = list(
                processor.process_file_chunks(test_file, timed_processor)
            )  # Use underscore

        assert len(benchmark.metrics_history) == 1
        metrics = benchmark.metrics_history[0]
        assert metrics.success is True
        assert metrics.items_processed == 100

    @pytest.mark.asyncio
    async def test_parallel_and_memory_integration(self, tmp_path):
        """Test parallel processing with memory management."""
        memory_manager = MemoryManager()
        config = ParallelProcessingConfig(max_workers=2, use_threads=True)
        processor = ParallelProcessor(config, memory_manager)

        # Create multiple test files
        files = []
        for i in range(4):
            test_data = pd.DataFrame({"col1": range(20), "col2": [f"file_{i}"] * 20})
            test_file = tmp_path / f"parallel_test_{i}.tsv"
            test_data.to_csv(test_file, sep="\t", index=False)
            files.append(test_file)

        def parallel_processor(df, **kwargs):
            return {"file_rows": len(df), "unique_values": df["col2"].nunique()}

        results = await processor.process_files_parallel(files, parallel_processor)

        assert len(results) == 4
        # All results should be successful (not None)
        successful_results = [r for r in results if r is not None]
        assert len(successful_results) == 4
