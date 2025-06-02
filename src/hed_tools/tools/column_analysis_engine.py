"""Main Column Analysis Engine for BIDS event files.

This module implements the BIDSColumnAnalysisEngine class that orchestrates
the entire column analysis workflow using the Facade pattern to provide
a unified interface for BIDS event file analysis.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .bids_parser import BIDSEventParser
from .enhanced_column_analyzer import EnhancedColumnAnalyzer, analyze_columns_enhanced
from .llm_preprocessor import create_llm_preprocessor, SamplingConfig


@dataclass
class AnalysisConfig:
    """Configuration for column analysis engine."""

    # Processing options
    max_workers: int = 4
    chunk_size: int = 10000
    enable_caching: bool = True

    # Analysis depth options
    enable_statistical_analysis: bool = True
    enable_pattern_recognition: bool = True
    enable_hed_detection: bool = True

    # Output options
    output_format: str = "json"  # json, csv, pickle
    include_detailed_stats: bool = True
    include_sample_data: bool = True

    # LLM preprocessing options
    sampling_config: SamplingConfig = field(default_factory=SamplingConfig)

    # Progress tracking
    enable_progress_tracking: bool = True
    progress_callback: Optional[callable] = None


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

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize the analysis engine.

        Args:
            config: Configuration options for the analysis engine
        """
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self._parser = BIDSEventParser()
        self._analyzer = EnhancedColumnAnalyzer()
        self._preprocessor = create_llm_preprocessor(self.config.sampling_config)

        # Results cache
        self._results_cache: Dict[str, FileAnalysisResult] = {}

        # Performance tracking
        self._performance_metrics = {
            "files_processed": 0,
            "total_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    async def analyze_file(self, file_path: Path) -> FileAnalysisResult:
        """Analyze a single BIDS event file."""
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
            cache_key = self._get_cache_key(file_path)
            if self.config.enable_caching and cache_key in self._results_cache:
                self._performance_metrics["cache_hits"] += 1
                return self._results_cache[cache_key]

            self._performance_metrics["cache_misses"] += 1

            # Parse the file
            events_data = await self._parse_file(file_path)

            # Perform enhanced analysis if enabled
            enhanced_analysis = None
            if self.config.enable_statistical_analysis:
                enhanced_analysis = await self._perform_enhanced_analysis(
                    events_data.events_df
                )

            # Prepare for LLM if enabled
            llm_samples = None
            if self.config.enable_hed_detection:
                llm_samples = self._prepare_for_llm(events_data.events_df)

            # Create result
            processing_time = time.time() - start_time
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
                processing_time=processing_time,
            )

            # Update performance metrics
            self._performance_metrics["files_processed"] += 1
            self._performance_metrics["total_processing_time"] += processing_time

            # Cache result
            if self.config.enable_caching:
                self._results_cache[cache_key] = result

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            # Still update metrics even for failed files
            self._performance_metrics["files_processed"] += 1
            self._performance_metrics["total_processing_time"] += processing_time

            return FileAnalysisResult(
                file_path=file_path,
                success=False,
                error_message=str(e),
                processing_time=processing_time,
            )

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

            # Process with progress tracking
            if self.config.enable_progress_tracking:
                for i, task in enumerate(asyncio.as_completed(tasks)):
                    result = await task
                    file_results.append(result)

                    if self.config.progress_callback:
                        await self.config.progress_callback(
                            i + 1, len(file_paths), result.file_path.name
                        )
            else:
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

                if (
                    self.config.enable_progress_tracking
                    and self.config.progress_callback
                ):
                    await self.config.progress_callback(
                        i + 1, len(file_paths), file_path.name
                    )

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
        return await self._parser.parse_events_file_async(file_path)

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
