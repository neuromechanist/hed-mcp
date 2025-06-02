"""Main Column Analysis Engine for BIDS event files.

This module implements the BIDSColumnAnalysisEngine class that orchestrates
the entire column analysis workflow using the Facade pattern to provide
a unified interface for BIDS event file analysis.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .bids_parser import BIDSEventParser
from .enhanced_column_analyzer import EnhancedColumnAnalyzer, analyze_columns_enhanced
from .llm_preprocessor import create_llm_preprocessor, SamplingConfig
from .performance_optimizer import (
    MemoryManager,
    ChunkedProcessor,
    LazyDataLoader,
    ParallelProcessor,
    PerformanceBenchmark,
    create_optimized_config,
    optimize_for_large_datasets,
)


@dataclass
class AnalysisConfig:
    """Configuration for BIDS column analysis."""

    # Processing options
    enable_enhanced_analysis: bool = True
    enable_llm_preprocessing: bool = True
    enable_caching: bool = True
    parallel_processing: bool = False
    max_workers: Optional[int] = None

    # Performance optimization options
    enable_memory_optimization: bool = True
    enable_chunked_processing: bool = True
    enable_lazy_loading: bool = True
    enable_performance_benchmarking: bool = False
    memory_limit_gb: float = 2.0
    chunk_size: int = 10000
    memory_aggressive: bool = False

    # Analysis options
    sampling_config: Optional[SamplingConfig] = None

    # Output options
    output_format: str = "json"  # json, csv, pickle
    include_metadata: bool = True
    save_intermediate_results: bool = False

    def __post_init__(self):
        """Initialize sampling config if not provided."""
        if self.sampling_config is None:
            self.sampling_config = SamplingConfig()


@dataclass
class FileAnalysisResult:
    """Results from analyzing a single BIDS event file."""

    file_path: Path
    success: bool
    error_message: Optional[str] = None

    # File metadata
    file_size: int = 0
    row_count: int = 0
    column_count: int = 0

    # Parsing results
    bids_compliance: Optional[Dict[str, Any]] = None
    column_info: Optional[Dict[str, Any]] = None

    # Analysis results
    enhanced_analysis: Optional[Dict[str, Any]] = None
    llm_preprocessed: Optional[Dict[str, Any]] = None

    # Performance metrics
    processing_time: float = 0.0
    memory_usage: Optional[float] = None


@dataclass
class BatchAnalysisResult:
    """Results from batch analysis of multiple files."""

    total_files: int
    successful_files: int
    failed_files: int

    file_results: List[FileAnalysisResult]

    # Aggregated statistics
    total_processing_time: float
    average_processing_time: float

    # Summary statistics
    total_columns_analyzed: int
    column_type_distribution: Dict[str, int]
    hed_candidate_columns: List[str]


class BIDSColumnAnalysisEngine:
    """Main engine for BIDS column analysis workflow.

    This class implements the Facade pattern to provide a unified interface
    for the entire column analysis workflow, coordinating between:
    - BIDSEventParser for file parsing and validation
    - EnhancedColumnAnalyzer for detailed column analysis
    - LLMPreprocessor for LLM-ready data preparation
    """

    def __init__(self, config: AnalysisConfig):
        """Initialize the analysis engine with performance optimizations."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize performance optimization components
        if self.config.enable_memory_optimization:
            optimization_settings = optimize_for_large_datasets(
                enable_chunking=self.config.enable_chunked_processing,
                enable_parallel=self.config.parallel_processing,
                memory_aggressive=self.config.memory_aggressive,
            )

            self._memory_manager = MemoryManager(
                memory_threshold=optimization_settings["memory_threshold"],
                cleanup_threshold=optimization_settings["cleanup_threshold"],
                gc_frequency=optimization_settings["gc_frequency"],
            )
        else:
            self._memory_manager = MemoryManager()

        # Initialize chunked processor
        if self.config.enable_chunked_processing:
            chunk_config, parallel_config = create_optimized_config(
                memory_limit_gb=self.config.memory_limit_gb,
                max_workers=self.config.max_workers,
                chunk_size=self.config.chunk_size,
            )
            self._chunked_processor = ChunkedProcessor(
                chunk_config, self._memory_manager
            )

            if self.config.parallel_processing:
                self._parallel_processor = ParallelProcessor(
                    parallel_config, self._memory_manager
                )
        else:
            self._chunked_processor = None
            self._parallel_processor = None

        # Initialize lazy loader
        if self.config.enable_lazy_loading:
            self._lazy_loader = LazyDataLoader(self._memory_manager)
        else:
            self._lazy_loader = None

        # Initialize performance benchmarking
        if self.config.enable_performance_benchmarking:
            self._benchmark = PerformanceBenchmark(self._memory_manager)
        else:
            self._benchmark = None

        # Initialize core components
        self._parser = BIDSEventParser()
        self._analyzer = EnhancedColumnAnalyzer()
        self._preprocessor = create_llm_preprocessor(self.config.sampling_config)

        # Performance tracking
        self._performance_metrics = {
            "files_processed": 0,
            "total_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "memory_optimizations": 0,
            "chunks_processed": 0,
        }

        # Results cache
        self._results_cache = {} if self.config.enable_caching else None

    async def analyze_file(self, file_path: Path) -> FileAnalysisResult:
        """Analyze a single BIDS event file with performance optimizations."""

        # Use benchmarking if enabled
        if self._benchmark:
            with self._benchmark.benchmark(f"analyze_file_{file_path.name}", 1):
                return await self._analyze_file_internal(file_path)
        else:
            return await self._analyze_file_internal(file_path)

    async def _analyze_file_internal(self, file_path: Path) -> FileAnalysisResult:
        """Internal file analysis with memory management."""

        with self._memory_manager.memory_guard(f"analyzing {file_path.name}"):
            start_time = time.time()

            try:
                # Check if file exists first
                if not file_path.exists():
                    return FileAnalysisResult(
                        file_path=file_path,
                        success=False,
                        error_message=f"File not found: {file_path}",
                        processing_time=time.time() - start_time,
                    )

                # Check cache first
                if self._results_cache is not None:
                    cache_key = self._get_cache_key(file_path)
                    if cache_key in self._results_cache:
                        self._performance_metrics["cache_hits"] += 1
                        cached_result = self._results_cache[cache_key]
                        self.logger.debug(f"Cache hit for {file_path.name}")
                        return cached_result
                    else:
                        self._performance_metrics["cache_misses"] += 1

                # Check if file is large and needs chunked processing
                file_size_mb = file_path.stat().st_size / (1024**2)

                if (
                    self._chunked_processor
                    and file_size_mb > 100  # Files larger than 100MB
                    and self.config.enable_chunked_processing
                ):
                    return await self._analyze_large_file_chunked(file_path, start_time)
                else:
                    return await self._analyze_standard_file(file_path, start_time)

            except Exception as e:
                self.logger.error(f"Error analyzing {file_path}: {e}")
                return FileAnalysisResult(
                    file_path=file_path,
                    success=False,
                    error_message=str(e),
                    processing_time=time.time() - start_time,
                )
            finally:
                # Update performance tracking
                self._performance_metrics["files_processed"] += 1
                self._performance_metrics["total_processing_time"] += (
                    time.time() - start_time
                )

    async def _analyze_standard_file(
        self, file_path: Path, start_time: float
    ) -> FileAnalysisResult:
        """Analyze a standard-sized file."""

        # Parse BIDS events
        events_data = await self._parse_file(file_path)

        # Optimize DataFrame memory usage
        events_data.events_df = self._memory_manager.optimize_dataframe(
            events_data.events_df, inplace=True
        )

        # Perform enhanced analysis if enabled
        enhanced_analysis = None
        if self.config.enable_enhanced_analysis:
            try:
                # Run enhanced analysis asynchronously for chunks
                enhanced_result = await self._analyzer.analyze_dataframe(
                    events_data.events_df
                )
                enhanced_analysis = enhanced_result
            except Exception as e:
                self.logger.warning(f"Enhanced analysis failed for file: {e}")

        # Prepare for LLM if enabled
        llm_samples = None
        if self.config.enable_llm_preprocessing:
            llm_samples = self._prepare_for_llm(events_data.events_df)

        # Create result
        result = FileAnalysisResult(
            file_path=file_path,
            success=True,
            file_size=file_path.stat().st_size,
            row_count=len(events_data.events_df),
            column_count=len(events_data.events_df.columns),
            bids_compliance=events_data.validation_results,
            column_info=events_data.column_info,
            enhanced_analysis=enhanced_analysis,
            llm_preprocessed=llm_samples,
            processing_time=time.time() - start_time,
        )

        # Cache result if caching is enabled
        if self._results_cache is not None:
            cache_key = self._get_cache_key(file_path)
            self._results_cache[cache_key] = result

        return result

    async def _analyze_large_file_chunked(
        self, file_path: Path, start_time: float
    ) -> FileAnalysisResult:
        """Analyze a large file using chunked processing."""

        self.logger.info(
            f"Processing large file {file_path.name} with chunked processing"
        )

        chunk_results = []
        total_rows = 0
        combined_enhanced_analysis = {}
        combined_llm_samples = {}

        # Process file in chunks
        for chunk_result in self._chunked_processor.process_file_chunks(
            file_path, self._process_dataframe_chunk
        ):
            chunk_results.append(chunk_result)
            total_rows += len(chunk_result.get("events_df", pd.DataFrame()))

            # Combine enhanced analysis results
            if chunk_result.get("enhanced_analysis"):
                self._merge_enhanced_analysis(
                    combined_enhanced_analysis, chunk_result["enhanced_analysis"]
                )

            # Combine LLM preprocessing results
            if chunk_result.get("llm_samples"):
                combined_llm_samples.update(chunk_result["llm_samples"])

            self._performance_metrics["chunks_processed"] += 1

        # Create combined result
        result = FileAnalysisResult(
            file_path=file_path,
            success=True,
            file_size=file_path.stat().st_size,
            row_count=total_rows,
            column_count=len(chunk_results[0].get("events_df", pd.DataFrame()).columns)
            if chunk_results
            else 0,
            bids_compliance=chunk_results[0].get("validation_results", {})
            if chunk_results
            else {},
            column_info=chunk_results[0].get("column_info", {})
            if chunk_results
            else {},
            enhanced_analysis=combined_enhanced_analysis,
            llm_preprocessed=combined_llm_samples,
            processing_time=time.time() - start_time,
        )

        return result

    def _process_dataframe_chunk(self, chunk_df: pd.DataFrame) -> Dict[str, Any]:
        """Process a single DataFrame chunk."""

        # Optimize chunk memory
        chunk_df = self._memory_manager.optimize_dataframe(chunk_df, inplace=True)

        # Analyze chunk
        result = {"events_df": chunk_df}

        # Enhanced analysis if enabled
        if self.config.enable_enhanced_analysis:
            try:
                # Use the analyze_columns_enhanced function for chunks (async)
                import asyncio

                if asyncio.iscoroutinefunction(analyze_columns_enhanced):
                    enhanced_result = asyncio.run(analyze_columns_enhanced(chunk_df))
                else:
                    enhanced_result = analyze_columns_enhanced(chunk_df)
                result["enhanced_analysis"] = enhanced_result
            except Exception as e:
                self.logger.warning(f"Enhanced analysis failed for chunk: {e}")

        # LLM preprocessing if enabled
        if self.config.enable_llm_preprocessing:
            try:
                llm_result = self._preprocessor.process_dataframe(chunk_df)
                result["llm_samples"] = llm_result
            except Exception as e:
                self.logger.warning(f"LLM preprocessing failed for chunk: {e}")

        return result

    def _merge_enhanced_analysis(
        self, combined: Dict[str, Any], chunk_analysis: Dict[str, Any]
    ):
        """Merge enhanced analysis results from multiple chunks."""

        if not combined:
            combined.update(chunk_analysis)
            return

        # Merge type distributions
        if "type_distribution" in chunk_analysis:
            if "type_distribution" not in combined:
                combined["type_distribution"] = {}
            for col_type, count in chunk_analysis["type_distribution"].items():
                combined["type_distribution"][col_type] = (
                    combined["type_distribution"].get(col_type, 0) + count
                )

        # Merge HED candidates (avoid duplicates)
        if "hed_candidates" in chunk_analysis:
            if "hed_candidates" not in combined:
                combined["hed_candidates"] = []

            existing_columns = {c["column"] for c in combined["hed_candidates"]}
            for candidate in chunk_analysis["hed_candidates"]:
                if candidate["column"] not in existing_columns:
                    combined["hed_candidates"].append(candidate)

        # Merge other metrics as needed
        for key in ["bids_compliance_score", "data_quality_score"]:
            if key in chunk_analysis:
                if key not in combined:
                    combined[key] = chunk_analysis[key]
                else:
                    # Average the scores
                    combined[key] = (combined[key] + chunk_analysis[key]) / 2

    async def analyze_batch(
        self, file_paths: List[Union[str, Path]]
    ) -> BatchAnalysisResult:
        """Analyze multiple BIDS event files in batch.

        Args:
            file_paths: List of paths to BIDS event files

        Returns:
            BatchAnalysisResult with aggregated results
        """
        start_time = time.time()
        file_paths = [Path(p) for p in file_paths]

        self.logger.info(f"Starting batch analysis of {len(file_paths)} files")

        # Process files concurrently
        file_results = []

        if self.config.max_workers > 1:
            # Parallel processing
            semaphore = asyncio.Semaphore(self.config.max_workers)

            async def analyze_with_semaphore(file_path):
                async with semaphore:
                    return await self.analyze_file(file_path)

            tasks = [analyze_with_semaphore(fp) for fp in file_paths]

            # Process all tasks concurrently
            file_results = await asyncio.gather(*tasks, return_exceptions=True)
            # Convert exceptions to failed results
            for i, result in enumerate(file_results):
                if isinstance(result, Exception):
                    file_results[i] = FileAnalysisResult(
                        file_path=file_paths[i],
                        success=False,
                        error_message=str(result),
                    )
        else:
            # Sequential processing
            for i, file_path in enumerate(file_paths):
                result = await self.analyze_file(file_path)
                file_results.append(result)

        # Compile batch results
        successful_results = [r for r in file_results if r.success]
        failed_results = [r for r in file_results if not r.success]

        total_processing_time = time.time() - start_time

        # Calculate aggregated statistics
        total_columns = sum(r.column_count for r in successful_results)
        column_type_dist = self._calculate_column_type_distribution(successful_results)
        hed_candidates = self._extract_hed_candidates(successful_results)

        batch_result = BatchAnalysisResult(
            total_files=len(file_paths),
            successful_files=len(successful_results),
            failed_files=len(failed_results),
            file_results=file_results,
            total_processing_time=total_processing_time,
            average_processing_time=total_processing_time / len(file_paths)
            if file_paths
            else 0.0,
            total_columns_analyzed=total_columns,
            column_type_distribution=column_type_dist,
            hed_candidate_columns=hed_candidates,
        )

        self.logger.info(
            f"Batch analysis complete: {batch_result.successful_files}/{batch_result.total_files} successful"
        )

        return batch_result

    async def analyze_directory(
        self,
        directory_path: Union[str, Path],
        pattern: str = "**/*_events.tsv",
        recursive: bool = True,
    ) -> BatchAnalysisResult:
        """Analyze all BIDS event files in a directory.

        Args:
            directory_path: Path to directory containing BIDS files
            pattern: Glob pattern for finding event files
            recursive: Whether to search recursively

        Returns:
            BatchAnalysisResult with all found files analyzed
        """
        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # Find all matching files
        if recursive:
            file_paths = list(directory_path.rglob(pattern))
        else:
            file_paths = list(directory_path.glob(pattern))

        if not file_paths:
            self.logger.warning(
                f"No files found matching pattern '{pattern}' in {directory_path}"
            )
            return BatchAnalysisResult(
                total_files=0,
                successful_files=0,
                failed_files=0,
                file_results=[],
                total_processing_time=0.0,
                average_processing_time=0.0,
                total_columns_analyzed=0,
                column_type_distribution={},
                hed_candidate_columns=[],
            )

        self.logger.info(
            f"Found {len(file_paths)} files to analyze in {directory_path}"
        )

        return await self.analyze_batch(file_paths)

    def get_analysis_summary(self, result: BatchAnalysisResult) -> Dict[str, Any]:
        """Generate a summary of batch analysis results.

        Args:
            result: BatchAnalysisResult to summarize

        Returns:
            Dictionary with analysis summary
        """
        successful_files = [r for r in result.file_results if r.success]

        summary = {
            "overview": {
                "total_files": result.total_files,
                "successful_files": result.successful_files,
                "failed_files": result.failed_files,
                "success_rate": result.successful_files / result.total_files
                if result.total_files > 0
                else 0.0,
            },
            "performance": {
                "total_processing_time": result.total_processing_time,
                "average_processing_time": result.average_processing_time,
                "files_per_second": (
                    result.total_files / result.total_processing_time
                    if result.total_processing_time > 0
                    else 0.0
                ),
            },
            "data_statistics": {
                "total_columns_analyzed": result.total_columns_analyzed,
                "column_type_distribution": result.column_type_distribution,
                "hed_candidate_columns": len(result.hed_candidate_columns),
                "average_columns_per_file": (
                    result.total_columns_analyzed / result.successful_files
                    if result.successful_files > 0
                    else 0.0
                ),
            },
            "quality_metrics": {
                "bids_compliant_files": sum(
                    1 for r in successful_files if self._is_bids_compliant(r)
                ),
                "files_with_hed_candidates": sum(
                    1 for r in successful_files if self._has_hed_candidates(r)
                ),
            },
            "cache_performance": {
                "cache_hits": self._performance_metrics["cache_hits"],
                "cache_misses": self._performance_metrics["cache_misses"],
                "cache_hit_rate": (
                    self._performance_metrics["cache_hits"]
                    / (
                        self._performance_metrics["cache_hits"]
                        + self._performance_metrics["cache_misses"]
                    )
                    if (
                        self._performance_metrics["cache_hits"]
                        + self._performance_metrics["cache_misses"]
                    )
                    > 0
                    else 0.0
                ),
            },
        }

        return summary

    def save_results(
        self,
        result: Union[FileAnalysisResult, BatchAnalysisResult],
        output_path: Union[str, Path],
    ) -> None:
        """Save analysis results to file.

        Args:
            result: Analysis results to save
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.output_format == "json":
            import json

            # Convert dataclass to dict for JSON serialization
            if isinstance(result, BatchAnalysisResult):
                data = self._batch_result_to_dict(result)
            else:
                data = self._file_result_to_dict(result)

            with open(output_path.with_suffix(".json"), "w") as f:
                json.dump(data, f, indent=2, default=str)

        elif self.config.output_format == "pickle":
            import pickle

            with open(output_path.with_suffix(".pkl"), "wb") as f:
                pickle.dump(result, f)

        elif self.config.output_format == "csv":
            # Convert to DataFrame for CSV export
            if isinstance(result, BatchAnalysisResult):
                df = self._batch_result_to_dataframe(result)
            else:
                df = self._file_result_to_dataframe(result)

            df.to_csv(output_path.with_suffix(".csv"), index=False)

        self.logger.info(f"Results saved to {output_path}")

    # Private helper methods

    async def _parse_file(self, file_path: Path):
        """Parse BIDS event file."""
        return await self._parser.parse_file_async(file_path)

    async def _perform_enhanced_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform enhanced column analysis."""
        try:
            # Use the enhanced column analyzer (which is async)
            return await analyze_columns_enhanced(df)
        except Exception as e:
            self.logger.error(f"Enhanced analysis failed: {e}")
            return {}

    def _prepare_for_llm(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Preprocess data for LLM classification."""
        try:
            # Use the LLM preprocessor (synchronous operation)
            return self._preprocessor.process_dataframe(df)
        except Exception as e:
            self.logger.error(f"LLM preprocessing failed: {e}")
            return {}

    def _get_cache_key(self, file_path: Path) -> str:
        """Generate cache key for a file."""
        try:
            stat = file_path.stat()
            return f"{file_path}:{stat.st_mtime}:{stat.st_size}"
        except (OSError, FileNotFoundError):
            # For non-existent files, just use the path
            return str(file_path)

    def _calculate_column_type_distribution(
        self, results: List[FileAnalysisResult]
    ) -> Dict[str, int]:
        """Calculate distribution of column types across all files."""
        type_counts = {}

        for result in results:
            enhanced_analysis = result.enhanced_analysis
            if (
                enhanced_analysis
                and isinstance(enhanced_analysis, dict)
                and "type_distribution" in enhanced_analysis
            ):
                for col_type, count in enhanced_analysis["type_distribution"].items():
                    type_counts[col_type] = type_counts.get(col_type, 0) + count

        return type_counts

    def _extract_hed_candidates(self, results: List[FileAnalysisResult]) -> List[str]:
        """Extract HED candidate columns from all files."""
        candidates = set()

        for result in results:
            if (
                result.enhanced_analysis
                and "hed_candidates" in result.enhanced_analysis
            ):
                # Each HED candidate is a dict with 'column', 'type', 'priority', 'reason'
                for candidate in result.enhanced_analysis["hed_candidates"]:
                    if isinstance(candidate, dict) and "column" in candidate:
                        candidates.add(candidate["column"])

        return list(candidates)

    def _is_bids_compliant(self, result: FileAnalysisResult) -> bool:
        """Check if file result indicates BIDS compliance."""
        if result.bids_compliance:
            return result.bids_compliance.get("is_valid", False)
        return False

    def _has_hed_candidates(self, result: FileAnalysisResult) -> bool:
        """Check if file has HED candidate columns."""
        if result.enhanced_analysis and "hed_candidates" in result.enhanced_analysis:
            return len(result.enhanced_analysis["hed_candidates"]) > 0
        return False

    def _batch_result_to_dict(self, result: BatchAnalysisResult) -> Dict[str, Any]:
        """Convert BatchAnalysisResult to dictionary."""
        return {
            "summary": {
                "total_files": result.total_files,
                "successful_files": result.successful_files,
                "failed_files": result.failed_files,
                "total_processing_time": result.total_processing_time,
                "average_processing_time": result.average_processing_time,
                "total_columns_analyzed": result.total_columns_analyzed,
                "column_type_distribution": result.column_type_distribution,
                "hed_candidate_columns": result.hed_candidate_columns,
            },
            "file_results": [
                self._file_result_to_dict(fr) for fr in result.file_results
            ],
        }

    def _file_result_to_dict(self, result: FileAnalysisResult) -> Dict[str, Any]:
        """Convert FileAnalysisResult to dictionary."""
        data = {
            "file_path": str(result.file_path),
            "success": result.success,
            "processing_time": result.processing_time,
            "file_size": result.file_size,
            "row_count": result.row_count,
            "column_count": result.column_count,
        }

        if result.error_message:
            data["error_message"] = result.error_message

        if result.bids_compliance:
            data["bids_compliance"] = result.bids_compliance

        if result.enhanced_analysis:
            data["enhanced_analysis"] = result.enhanced_analysis

        if result.llm_preprocessed:
            data["llm_preprocessed"] = result.llm_preprocessed

        return data

    def _batch_result_to_dataframe(self, result: BatchAnalysisResult) -> pd.DataFrame:
        """Convert BatchAnalysisResult to DataFrame."""
        rows = []

        for fr in result.file_results:
            row = {
                "file_path": str(fr.file_path),
                "success": fr.success,
                "processing_time": fr.processing_time,
                "file_size": fr.file_size,
                "row_count": fr.row_count,
                "column_count": fr.column_count,
                "error_message": fr.error_message or "",
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def _file_result_to_dataframe(self, result: FileAnalysisResult) -> pd.DataFrame:
        """Convert FileAnalysisResult to DataFrame."""
        return pd.DataFrame([self._file_result_to_dict(result)])


# Convenience functions


def create_analysis_engine(
    config: Optional[AnalysisConfig] = None,
) -> BIDSColumnAnalysisEngine:
    """Create a new BIDSColumnAnalysisEngine instance.

    Args:
        config: Optional configuration for the engine

    Returns:
        Configured BIDSColumnAnalysisEngine instance
    """
    if config is None:
        config = AnalysisConfig()
    return BIDSColumnAnalysisEngine(config)


async def analyze_bids_files(
    file_paths: List[Union[str, Path]], config: Optional[AnalysisConfig] = None
) -> BatchAnalysisResult:
    """Convenience function to analyze multiple BIDS files.

    Args:
        file_paths: List of paths to BIDS event files
        config: Optional configuration for analysis

    Returns:
        BatchAnalysisResult with analysis results
    """
    engine = create_analysis_engine(config)
    return await engine.analyze_batch(file_paths)


async def analyze_bids_directory(
    directory_path: Union[str, Path],
    pattern: str = "**/*_events.tsv",
    config: Optional[AnalysisConfig] = None,
) -> BatchAnalysisResult:
    """Convenience function to analyze BIDS files in a directory.

    Args:
        directory_path: Path to directory containing BIDS files
        pattern: Glob pattern for finding event files
        config: Optional configuration for analysis

    Returns:
        BatchAnalysisResult with analysis results
    """
    engine = create_analysis_engine(config)
    return await engine.analyze_directory(directory_path, pattern)
