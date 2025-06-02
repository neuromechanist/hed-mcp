"""TabularSummary integration module for HED operations.

This module provides a comprehensive wrapper around the HED TabularSummary class,
offering async/await patterns, batch processing capabilities, performance monitoring,
and memory optimization for handling tabular event data analysis.
"""

import asyncio
import hashlib
import logging
import time
import gc
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import wraps, lru_cache
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Union, Iterator, Callable, 
    Tuple, Type, Protocol
)

import pandas as pd

try:
    from hed.tools.analysis.tabular_summary import TabularSummary
    from hed.errors.exceptions import HedFileError, HedSchemaError
    HED_AVAILABLE = True
except ImportError:
    # Graceful fallback for development
    TabularSummary = None
    HedFileError = Exception
    HedSchemaError = Exception
    HED_AVAILABLE = False

from .models import (
    TabularSummaryConfig, EventsData, ColumnInfo, OperationResult,
    SidecarTemplate, ValidationResult
)
from .schema import SchemaHandler

logger = logging.getLogger(__name__)


class DataFormat(Enum):
    """Supported data formats for TabularSummary operations."""
    CSV = "csv"
    TSV = "tsv" 
    EXCEL = "excel"
    PARQUET = "parquet"


class ValidationError(Exception):
    """Custom exception for TabularSummary validation errors."""
    pass


@dataclass
class PerformanceMetrics:
    """Performance metrics for TabularSummary operations."""
    operation: str
    duration: float
    memory_usage_mb: float
    cache_hit: bool
    timestamp: float
    file_size_mb: Optional[float] = None
    row_count: Optional[int] = None


class ProgressCallback(Protocol):
    """Protocol for progress callback functions."""
    
    async def on_progress(self, current: int, total: int, operation: str) -> None:
        """Called when progress is made on an operation."""
        ...


def async_wrapper(func):
    """Decorator to make synchronous TabularSummary methods async."""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, func, self, *args, **kwargs)
    return wrapper


class DataValidator:
    """Validator for tabular data used in TabularSummary operations."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, 
                          required_columns: Optional[List[str]] = None) -> None:
        """Validate DataFrame for TabularSummary operations.
        
        Args:
            df: DataFrame to validate
            required_columns: Optional list of required column names
            
        Raises:
            ValidationError: If validation fails
        """
        if df.empty:
            raise ValidationError("DataFrame is empty")
        
        if df.shape[0] == 0:
            raise ValidationError("DataFrame has no rows")
        
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise ValidationError(f"Missing required columns: {missing_cols}")
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            logger.warning(f"Found completely empty columns: {empty_cols}")
    
    @staticmethod
    def validate_file_format(file_path: Path) -> DataFormat:
        """Validate and detect file format.
        
        Args:
            file_path: Path to file
            
        Returns:
            DataFormat enum value
            
        Raises:
            ValidationError: If format is not supported
        """
        if not file_path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
        
        suffix = file_path.suffix.lower()
        format_map = {
            '.csv': DataFormat.CSV,
            '.tsv': DataFormat.TSV,
            '.xlsx': DataFormat.EXCEL,
            '.xls': DataFormat.EXCEL,
            '.parquet': DataFormat.PARQUET
        }
        
        if suffix not in format_map:
            raise ValidationError(f"Unsupported file format: {suffix}")
        
        return format_map[suffix]


class MemoryManager:
    """Memory management utilities for TabularSummary operations."""
    
    def __init__(self, memory_threshold: float = 0.8):
        """Initialize memory manager.
        
        Args:
            memory_threshold: Memory usage threshold (0.0-1.0) to trigger cleanup
        """
        self.memory_threshold = memory_threshold
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def memory_guard(self):
        """Context manager for memory-intensive operations."""
        initial_memory = self._get_memory_usage()
        try:
            yield
        finally:
            current_memory = self._get_memory_usage()
            if current_memory > self.memory_threshold:
                gc.collect()
                self.logger.info(f"Memory cleanup: {initial_memory:.1%} -> {current_memory:.1%}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            # Fallback if psutil not available
            return 0.0
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Memory-optimized DataFrame
        """
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        optimized_df = df.copy()
        
        # Convert object columns to category if they have low cardinality
        for col in optimized_df.select_dtypes(include=['object']):
            if optimized_df[col].nunique() / len(optimized_df) < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')
        
        # Downcast numeric types where possible
        for col in optimized_df.select_dtypes(include=['int64']):
            col_min, col_max = optimized_df[col].min(), optimized_df[col].max()
            if col_min >= 0 and col_max < 2**32:
                optimized_df[col] = optimized_df[col].astype('uint32')
            elif col_min >= -2**31 and col_max < 2**31:
                optimized_df[col] = optimized_df[col].astype('int32')
        
        final_memory = optimized_df.memory_usage(deep=True).sum() / 1024**2
        reduction = (1 - final_memory / initial_memory) * 100
        
        if reduction > 5:  # Only log significant reductions
            self.logger.info(f"Memory optimization: {reduction:.1f}% reduction "
                           f"({initial_memory:.1f}MB -> {final_memory:.1f}MB)")
        
        return optimized_df


class CacheManager:
    """Cache manager for TabularSummary results."""
    
    def __init__(self, max_size: int = 128, ttl_seconds: int = 3600):
        """Initialize cache manager.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cached items in seconds
        """
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_times: Dict[str, float] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
    
    def _generate_key(self, data_hash: str, params: Dict[str, Any]) -> str:
        """Generate cache key from data hash and parameters.
        
        Args:
            data_hash: Hash of the input data
            params: Operation parameters
            
        Returns:
            Cache key string
        """
        # Sort parameters for consistent keys
        param_str = str(sorted(params.items()))
        combined = f"{data_hash}:{param_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, key: str) -> Tuple[Optional[Any], bool]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Tuple of (cached_value, cache_hit_boolean)
        """
        if key not in self._cache:
            return None, False
        
        value, timestamp = self._cache[key]
        
        # Check if expired
        if time.time() - timestamp > self._ttl_seconds:
            self._evict(key)
            return None, False
        
        # Update access time
        self._access_times[key] = time.time()
        return value, True
    
    def put(self, key: str, value: Any) -> None:
        """Store item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        current_time = time.time()
        
        # Evict if at capacity
        if len(self._cache) >= self._max_size:
            self._evict_lru()
        
        self._cache[key] = (value, current_time)
        self._access_times[key] = current_time
    
    def _evict(self, key: str) -> None:
        """Evict specific key from cache."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), 
                     key=lambda k: self._access_times[k])
        self._evict(lru_key)
    
    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        self._access_times.clear()


class TabularSummaryWrapper:
    """Comprehensive wrapper for HED TabularSummary operations.
    
    This class provides async/await patterns, caching, memory optimization,
    and enhanced error handling for TabularSummary operations.
    """
    
    def __init__(self, 
                 config: Optional[TabularSummaryConfig] = None,
                 schema_handler: Optional[SchemaHandler] = None,
                 executor_max_workers: int = 4):
        """Initialize TabularSummary wrapper.
        
        Args:
            config: Configuration for TabularSummary operations
            schema_handler: HED schema handler for validation
            executor_max_workers: Maximum workers for thread pool executor
        """
        self.config = config or TabularSummaryConfig()
        self.schema_handler = schema_handler
        self._executor = ThreadPoolExecutor(max_workers=executor_max_workers)
        
        # Initialize managers
        self.cache_manager = CacheManager(max_size=128)
        self.memory_manager = MemoryManager()
        self.validator = DataValidator()
        
        # Progress tracking
        self._progress_callbacks: List[ProgressCallback] = []
        
        if not HED_AVAILABLE:
            logger.warning("HED library not available - wrapper running in stub mode")
    
    def add_progress_callback(self, callback: ProgressCallback) -> None:
        """Add progress callback for batch operations.
        
        Args:
            callback: Progress callback function
        """
        self._progress_callbacks.append(callback)
    
    async def _emit_progress(self, current: int, total: int, operation: str) -> None:
        """Emit progress to all registered callbacks."""
        for callback in self._progress_callbacks:
            try:
                await callback.on_progress(current, total, operation)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Generate hash for DataFrame caching.
        
        Args:
            df: DataFrame to hash
            
        Returns:
            Hash string
        """
        # Use DataFrame content hash for caching
        content = pd.util.hash_pandas_object(df, index=True).values
        return hashlib.md5(content.tobytes()).hexdigest()
    
    async def load_data(self, 
                       source: Union[str, Path, pd.DataFrame],
                       **read_kwargs) -> pd.DataFrame:
        """Load data from various sources.
        
        Args:
            source: Data source (file path or DataFrame)
            **read_kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded DataFrame
            
        Raises:
            ValidationError: If loading fails or data is invalid
        """
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            file_path = Path(source)
            data_format = self.validator.validate_file_format(file_path)
            
            # Load based on format
            if data_format == DataFormat.CSV:
                df = await self._load_csv_async(file_path, **read_kwargs)
            elif data_format == DataFormat.TSV:
                kwargs = {'sep': '\t', **read_kwargs}
                df = await self._load_csv_async(file_path, **kwargs)
            elif data_format == DataFormat.EXCEL:
                df = await self._load_excel_async(file_path, **read_kwargs)
            elif data_format == DataFormat.PARQUET:
                df = await self._load_parquet_async(file_path, **read_kwargs)
            else:
                raise ValidationError(f"Unsupported data format: {data_format}")
        
        # Validate loaded data
        self.validator.validate_dataframe(df)
        
        # Optimize memory usage
        with self.memory_manager.memory_guard():
            df = self.memory_manager.optimize_dataframe(df)
        
        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    async def _load_csv_async(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load CSV file asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, 
            lambda: pd.read_csv(file_path, **kwargs)
        )
    
    async def _load_excel_async(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load Excel file asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, 
            lambda: pd.read_excel(file_path, **kwargs)
        )
    
    async def _load_parquet_async(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load Parquet file asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, 
            lambda: pd.read_parquet(file_path, **kwargs)
        )
    
    async def generate_summary(self, 
                             data: Union[pd.DataFrame, str, Path],
                             skip_columns: Optional[List[str]] = None,
                             value_columns: Optional[List[str]] = None,
                             name: str = "",
                             use_cache: bool = True) -> OperationResult:
        """Generate TabularSummary with caching and performance monitoring.
        
        Args:
            data: Input data (DataFrame or file path)
            skip_columns: Columns to skip in analysis
            value_columns: Columns to treat as value columns
            name: Name for the summary
            use_cache: Whether to use caching
            
        Returns:
            OperationResult with summary data and metadata
        """
        start_time = time.time()
        
        try:
            # Load data if needed
            if not isinstance(data, pd.DataFrame):
                df = await self.load_data(data)
            else:
                df = data.copy()
            
            # Prepare parameters
            params = {
                'skip_columns': skip_columns or self.config.skip_columns,
                'value_columns': value_columns or self.config.value_columns,
                'name': name or self.config.name
            }
            
            # Check cache if enabled
            cache_hit = False
            if use_cache:
                data_hash = self._hash_dataframe(df)
                cache_key = self.cache_manager._generate_key(data_hash, params)
                cached_result, cache_hit = self.cache_manager.get(cache_key)
                
                if cache_hit:
                    logger.debug(f"Cache hit for summary generation: {cache_key[:8]}...")
                    return OperationResult(
                        success=True,
                        data=cached_result,
                        processing_time=time.time() - start_time,
                        metadata={'cache_hit': True}
                    )
            
            # Generate summary
            summary_result = await self._generate_summary_async(df, **params)
            
            # Cache result if enabled
            if use_cache and not cache_hit:
                self.cache_manager.put(cache_key, summary_result)
            
            processing_time = time.time() - start_time
            
            return OperationResult(
                success=True,
                data=summary_result,
                processing_time=processing_time,
                metadata={
                    'cache_hit': cache_hit,
                    'row_count': len(df),
                    'column_count': len(df.columns)
                }
            )
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def _generate_summary_async(self, 
                                    df: pd.DataFrame,
                                    skip_columns: Optional[List[str]] = None,
                                    value_columns: Optional[List[str]] = None,
                                    name: str = "") -> Dict[str, Any]:
        """Generate TabularSummary asynchronously.
        
        Args:
            df: Input DataFrame
            skip_columns: Columns to skip
            value_columns: Value columns
            name: Summary name
            
        Returns:
            TabularSummary result dictionary
        """
        if not HED_AVAILABLE:
            # Return mock result for testing
            return {
                'summary': 'Mock summary - HED not available',
                'columns': df.columns.tolist(),
                'row_count': len(df)
            }
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._generate_summary_sync,
            df, skip_columns, value_columns, name
        )
    
    def _generate_summary_sync(self, 
                              df: pd.DataFrame,
                              skip_columns: Optional[List[str]] = None,
                              value_columns: Optional[List[str]] = None,
                              name: str = "") -> Dict[str, Any]:
        """Generate TabularSummary synchronously.
        
        This method runs in a thread pool to avoid blocking.
        """
        try:
            summary = TabularSummary(
                data_input=df,
                skip_columns=skip_columns,
                value_columns=value_columns,
                name=name
            )
            
            return {
                'summary': summary.get_summary(as_json=False),
                'template': summary.extract_sidecar_template(),
                'column_definitions': summary.get_column_def_names(),
                'unique_values': {col: summary.get_unique_column_values(col) 
                                for col in df.columns if col not in (skip_columns or [])}
            }
            
        except Exception as e:
            logger.error(f"TabularSummary generation failed: {e}")
            raise
    
    async def extract_sidecar_template(self,
                                     data: Union[pd.DataFrame, str, Path],
                                     columns: Optional[List[str]] = None,
                                     skip_columns: Optional[List[str]] = None,
                                     use_cache: bool = True) -> SidecarTemplate:
        """Extract HED sidecar template from data.
        
        Args:
            data: Input data (DataFrame or file path)
            columns: Specific columns to include
            skip_columns: Columns to skip
            use_cache: Whether to use caching
            
        Returns:
            SidecarTemplate object with template and metadata
        """
        start_time = time.time()
        
        try:
            # Load data if needed
            if not isinstance(data, pd.DataFrame):
                df = await self.load_data(data)
            else:
                df = data.copy()
            
            # Generate summary first
            summary_result = await self.generate_summary(
                df, skip_columns=skip_columns, use_cache=use_cache
            )
            
            if not summary_result.success:
                raise ValidationError(f"Failed to generate summary: {summary_result.error}")
            
            template_data = summary_result.data.get('template', {})
            
            return SidecarTemplate(
                template=template_data,
                generated_columns=list(template_data.keys()),
                schema_version=self.schema_handler.get_version() if self.schema_handler else "unknown",
                generation_time=time.time() - start_time,
                metadata={
                    'row_count': len(df),
                    'total_columns': len(df.columns),
                    'cache_hit': summary_result.metadata.get('cache_hit', False)
                }
            )
            
        except Exception as e:
            logger.error(f"Sidecar template extraction failed: {e}")
            raise
    
    async def process_batch(self,
                          file_paths: List[Union[str, Path]], 
                          chunk_size: int = 10,
                          continue_on_error: bool = True) -> Iterator[Dict[str, Any]]:
        """Process multiple files in batches.
        
        Args:
            file_paths: List of file paths to process
            chunk_size: Number of files to process simultaneously
            continue_on_error: Whether to continue if one file fails
            
        Yields:
            Processing results for each file
        """
        total_files = len(file_paths)
        processed = 0
        
        for i in range(0, total_files, chunk_size):
            chunk = file_paths[i:i + chunk_size]
            
            # Create tasks for chunk
            tasks = []
            for file_path in chunk:
                task = self._process_single_file(file_path, continue_on_error)
                tasks.append(task)
            
            # Process chunk
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Yield results and update progress
            for file_path, result in zip(chunk, results):
                processed += 1
                
                if isinstance(result, Exception):
                    if not continue_on_error:
                        raise result
                    logger.error(f"Failed to process {file_path}: {result}")
                    result = {
                        'file': str(file_path),
                        'success': False,
                        'error': str(result)
                    }
                
                # Emit progress
                await self._emit_progress(processed, total_files, f"processing {file_path}")
                
                yield result
    
    async def _process_single_file(self, 
                                  file_path: Union[str, Path],
                                  continue_on_error: bool = True) -> Dict[str, Any]:
        """Process a single file.
        
        Args:
            file_path: Path to file
            continue_on_error: Whether to continue on errors
            
        Returns:
            Processing result dictionary
        """
        try:
            summary_result = await self.generate_summary(file_path)
            
            return {
                'file': str(file_path),
                'success': summary_result.success,
                'summary': summary_result.data,
                'processing_time': summary_result.processing_time,
                'metadata': summary_result.metadata
            }
            
        except Exception as e:
            if not continue_on_error:
                raise
            
            return {
                'file': str(file_path),
                'success': False,
                'error': str(e)
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the wrapper.
        
        Returns:
            Dictionary with performance statistics
        """
        return {
            'cache_size': len(self.cache_manager._cache),
            'cache_hit_rate': getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1),
            'memory_usage': self.memory_manager._get_memory_usage(),
            'executor_threads': self._executor._max_workers
        }
    
    async def close(self) -> None:
        """Close the wrapper and clean up resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
        
        # Clear caches
        self.cache_manager.clear()
        
        logger.info("TabularSummaryWrapper closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Factory function for easy instantiation
def create_tabular_summary_wrapper(config: Optional[TabularSummaryConfig] = None,
                                  schema_handler: Optional[SchemaHandler] = None) -> TabularSummaryWrapper:
    """Create a TabularSummaryWrapper instance.
    
    Args:
        config: Optional configuration
        schema_handler: Optional schema handler
        
    Returns:
        Configured TabularSummaryWrapper instance
    """
    return TabularSummaryWrapper(config=config, schema_handler=schema_handler) 