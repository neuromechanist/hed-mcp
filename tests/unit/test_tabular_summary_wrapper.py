"""Unit tests for TabularSummaryWrapper implementation.

This module contains comprehensive tests for the TabularSummary integration,
including async operations, caching, memory management, and batch processing.
"""

import pytest
import pandas as pd
from unittest.mock import patch

from src.hed_tools.hed_integration.tabular_summary import (
    TabularSummaryWrapper, create_tabular_summary_wrapper,
    DataValidator, MemoryManager, CacheManager,
    ValidationError, DataFormat
)
from src.hed_tools.hed_integration.models import (
    TabularSummaryConfig, OperationResult, SidecarTemplate
)


class TestDataValidator:
    """Test cases for DataValidator class."""
    
    def test_validate_dataframe_success(self):
        """Test successful DataFrame validation."""
        df = pd.DataFrame({
            'onset': [1.0, 2.0, 3.0],
            'duration': [0.5, 0.5, 0.5],
            'trial_type': ['go', 'stop', 'go']
        })
        
        # Should not raise any exceptions
        DataValidator.validate_dataframe(df)
    
    def test_validate_dataframe_empty_fails(self):
        """Test that empty DataFrame validation fails."""
        df = pd.DataFrame()
        
        with pytest.raises(ValidationError, match="DataFrame is empty"):
            DataValidator.validate_dataframe(df)
    
    def test_validate_dataframe_no_rows_fails(self):
        """Test that DataFrame with no rows fails validation."""
        df = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])
        
        with pytest.raises(ValidationError, match="DataFrame has no rows"):
            DataValidator.validate_dataframe(df)
    
    def test_validate_dataframe_missing_required_columns(self):
        """Test validation fails when required columns are missing."""
        df = pd.DataFrame({'onset': [1.0, 2.0]})
        required_columns = ['onset', 'duration', 'trial_type']
        
        with pytest.raises(ValidationError, match="Missing required columns"):
            DataValidator.validate_dataframe(df, required_columns)
    
    def test_validate_file_format_csv(self, tmp_path):
        """Test CSV file format validation."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("onset,duration,trial_type\n1.0,0.5,go\n")
        
        format_result = DataValidator.validate_file_format(csv_file)
        assert format_result == DataFormat.CSV
    
    def test_validate_file_format_excel(self, tmp_path):
        """Test Excel file format validation."""
        excel_file = tmp_path / "test.xlsx"
        excel_file.touch()  # Create empty file
        
        format_result = DataValidator.validate_file_format(excel_file)
        assert format_result == DataFormat.EXCEL
    
    def test_validate_file_format_unsupported(self, tmp_path):
        """Test unsupported file format raises error."""
        unsupported_file = tmp_path / "test.json"
        unsupported_file.write_text("{}")
        
        with pytest.raises(ValidationError, match="Unsupported file format"):
            DataValidator.validate_file_format(unsupported_file)
    
    def test_validate_file_format_nonexistent(self, tmp_path):
        """Test nonexistent file raises error."""
        nonexistent_file = tmp_path / "nonexistent.csv"
        
        with pytest.raises(ValidationError, match="File does not exist"):
            DataValidator.validate_file_format(nonexistent_file)


class TestMemoryManager:
    """Test cases for MemoryManager class."""
    
    def test_memory_manager_initialization(self):
        """Test MemoryManager initialization."""
        manager = MemoryManager(memory_threshold=0.9)
        assert manager.memory_threshold == 0.9
    
    def test_optimize_dataframe_category_conversion(self):
        """Test DataFrame optimization with category conversion."""
        # Create DataFrame with low-cardinality string column
        df = pd.DataFrame({
            'category_col': ['A'] * 50 + ['B'] * 50,  # Low cardinality
            'unique_col': [f'unique_{i}' for i in range(100)],  # High cardinality
            'numeric_col': range(100)
        })
        
        manager = MemoryManager()
        optimized_df = manager.optimize_dataframe(df)
        
        # Low cardinality column should be converted to category
        assert optimized_df['category_col'].dtype.name == 'category'
        # High cardinality column should remain object
        assert optimized_df['unique_col'].dtype.name == 'object'
    
    def test_optimize_dataframe_numeric_downcast(self):
        """Test DataFrame optimization with numeric downcasting."""
        df = pd.DataFrame({
            'small_int': [1, 2, 3, 4, 5],  # Can be downcasted
            'large_int': [2**40, 2**41, 2**42],  # Cannot be downcasted
        })
        
        manager = MemoryManager()
        optimized_df = manager.optimize_dataframe(df)
        
        # Small integers should be downcasted
        assert optimized_df['small_int'].dtype == 'uint32'
        # Large integers should remain int64
        assert optimized_df['large_int'].dtype == 'int64'
    
    @patch('hed_tools.hed_integration.tabular_summary.gc.collect')
    def test_memory_guard_cleanup(self, mock_gc_collect):
        """Test memory guard triggers cleanup when threshold exceeded."""
        manager = MemoryManager(memory_threshold=0.5)
        
        with patch.object(manager, '_get_memory_usage', side_effect=[0.3, 0.8]):
            with manager.memory_guard():
                pass
        
        mock_gc_collect.assert_called_once()


class TestCacheManager:
    """Test cases for CacheManager class."""
    
    def test_cache_manager_initialization(self):
        """Test CacheManager initialization."""
        cache = CacheManager(max_size=64, ttl_seconds=1800)
        assert cache._max_size == 64
        assert cache._ttl_seconds == 1800
    
    def test_cache_put_get_success(self):
        """Test successful cache put and get operations."""
        cache = CacheManager()
        key = "test_key"
        value = {"test": "data"}
        
        cache.put(key, value)
        retrieved_value, cache_hit = cache.get(key)
        
        assert cache_hit is True
        assert retrieved_value == value
    
    def test_cache_get_miss(self):
        """Test cache miss returns None and False."""
        cache = CacheManager()
        
        value, cache_hit = cache.get("nonexistent_key")
        
        assert value is None
        assert cache_hit is False
    
    def test_cache_generate_key_consistency(self):
        """Test cache key generation is consistent."""
        cache = CacheManager()
        data_hash = "abc123"
        params1 = {'skip_cols': ['onset'], 'name': 'test'}
        params2 = {'name': 'test', 'skip_cols': ['onset']}  # Different order
        
        key1 = cache._generate_key(data_hash, params1)
        key2 = cache._generate_key(data_hash, params2)
        
        assert key1 == key2  # Should be same despite different order
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = CacheManager(max_size=2)
        
        # Fill cache to capacity
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access key1 to make it more recently used
        cache.get("key1")
        
        # Add third item, should evict key2 (least recently used)
        cache.put("key3", "value3")
        
        # key1 and key3 should exist, key2 should be evicted
        assert cache.get("key1")[1] is True
        assert cache.get("key3")[1] is True
        assert cache.get("key2")[1] is False
    
    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        import time
        cache = CacheManager(ttl_seconds=0.1)  # Very short TTL
        
        cache.put("key", "value")
        
        # Should be available immediately
        value, hit = cache.get("key")
        assert hit is True
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired now
        value, hit = cache.get("key")
        assert hit is False


@pytest.fixture
def sample_dataframe():
    """Fixture providing sample DataFrame for testing."""
    return pd.DataFrame({
        'onset': [1.0, 2.0, 3.0, 4.0, 5.0],
        'duration': [0.5, 0.5, 0.5, 0.5, 0.5],
        'trial_type': ['go', 'stop', 'go', 'stop', 'go'],
        'response': ['left', 'right', 'left', 'right', 'left'],
        'accuracy': [1, 0, 1, 1, 0]
    })


@pytest.fixture
def sample_config():
    """Fixture providing sample configuration."""
    return TabularSummaryConfig(
        skip_columns=['onset', 'duration'],
        name='test_summary',
        max_unique_values=10
    )


class TestTabularSummaryWrapper:
    """Test cases for TabularSummaryWrapper class."""
    
    def test_wrapper_initialization(self, sample_config):
        """Test TabularSummaryWrapper initialization."""
        wrapper = TabularSummaryWrapper(config=sample_config)
        
        assert wrapper.config == sample_config
        assert wrapper.cache_manager is not None
        assert wrapper.memory_manager is not None
        assert wrapper.validator is not None
    
    def test_wrapper_initialization_defaults(self):
        """Test TabularSummaryWrapper initialization with defaults."""
        wrapper = TabularSummaryWrapper()
        
        assert wrapper.config is not None
        assert isinstance(wrapper.config, TabularSummaryConfig)
    
    @pytest.mark.asyncio
    async def test_load_data_from_dataframe(self, sample_dataframe):
        """Test loading data from DataFrame."""
        wrapper = TabularSummaryWrapper()
        
        result_df = await wrapper.load_data(sample_dataframe)
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(sample_dataframe)
        pd.testing.assert_frame_equal(result_df, sample_dataframe)
    
    @pytest.mark.asyncio
    async def test_load_data_from_csv_file(self, tmp_path, sample_dataframe):
        """Test loading data from CSV file."""
        # Create CSV file
        csv_file = tmp_path / "test_events.csv"
        sample_dataframe.to_csv(csv_file, index=False)
        
        wrapper = TabularSummaryWrapper()
        result_df = await wrapper.load_data(csv_file)
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(sample_dataframe)
        # Check columns match (allowing for potential optimizations)
        assert set(result_df.columns) == set(sample_dataframe.columns)
    
    def test_hash_dataframe_consistency(self, sample_dataframe):
        """Test DataFrame hashing is consistent."""
        wrapper = TabularSummaryWrapper()
        
        hash1 = wrapper._hash_dataframe(sample_dataframe)
        hash2 = wrapper._hash_dataframe(sample_dataframe.copy())
        
        assert hash1 == hash2
    
    def test_hash_dataframe_different_data(self, sample_dataframe):
        """Test DataFrame hashing produces different hashes for different data."""
        wrapper = TabularSummaryWrapper()
        
        modified_df = sample_dataframe.copy()
        modified_df.iloc[0, 0] = 999  # Change one value
        
        hash1 = wrapper._hash_dataframe(sample_dataframe)
        hash2 = wrapper._hash_dataframe(modified_df)
        
        assert hash1 != hash2
    
    @pytest.mark.asyncio
    async def test_generate_summary_success(self, sample_dataframe):
        """Test successful summary generation."""
        wrapper = TabularSummaryWrapper()
        
        # Mock the HED TabularSummary since it might not be available in tests
        with patch.object(wrapper, '_generate_summary_async') as mock_generate:
            mock_generate.return_value = {
                'summary': {'test': 'summary'},
                'template': {'trial_type': {'HED': {}}},
                'column_definitions': ['trial_type'],
                'unique_values': {'trial_type': ['go', 'stop']}
            }
            
            result = await wrapper.generate_summary(sample_dataframe)
            
            assert result.success is True
            assert result.data is not None
            assert 'summary' in result.data
            assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_generate_summary_caching(self, sample_dataframe):
        """Test summary generation uses caching."""
        wrapper = TabularSummaryWrapper()
        
        with patch.object(wrapper, '_generate_summary_async') as mock_generate:
            mock_generate.return_value = {'summary': 'test'}
            
            # First call should hit the actual method
            result1 = await wrapper.generate_summary(sample_dataframe, use_cache=True)
            assert result1.metadata.get('cache_hit') is False
            
            # Second call should hit the cache
            result2 = await wrapper.generate_summary(sample_dataframe, use_cache=True)
            assert result2.metadata.get('cache_hit') is True
            
            # Should only call the actual method once
            assert mock_generate.call_count == 1
    
    @pytest.mark.asyncio
    async def test_generate_summary_error_handling(self, sample_dataframe):
        """Test error handling in summary generation."""
        wrapper = TabularSummaryWrapper()
        
        with patch.object(wrapper, '_generate_summary_async') as mock_generate:
            mock_generate.side_effect = Exception("Test error")
            
            result = await wrapper.generate_summary(sample_dataframe)
            
            assert result.success is False
            assert "Test error" in result.error
    
    @pytest.mark.asyncio
    async def test_extract_sidecar_template(self, sample_dataframe):
        """Test sidecar template extraction."""
        wrapper = TabularSummaryWrapper()
        
        with patch.object(wrapper, 'generate_summary') as mock_generate:
            mock_result = OperationResult(
                success=True,
                data={'template': {'trial_type': {'HED': {'go': 'Event/go', 'stop': 'Event/stop'}}}},
                processing_time=0.1,
                metadata={'cache_hit': False}
            )
            mock_generate.return_value = mock_result
            
            template = await wrapper.extract_sidecar_template(sample_dataframe)
            
            assert isinstance(template, SidecarTemplate)
            assert 'trial_type' in template.template
            assert 'trial_type' in template.generated_columns
            assert template.generation_time > 0
    
    @pytest.mark.asyncio
    async def test_process_batch_success(self, tmp_path, sample_dataframe):
        """Test successful batch processing."""
        # Create multiple CSV files
        files = []
        for i in range(3):
            csv_file = tmp_path / f"test_{i}.csv"
            sample_dataframe.to_csv(csv_file, index=False)
            files.append(csv_file)
        
        wrapper = TabularSummaryWrapper()
        
        with patch.object(wrapper, 'generate_summary') as mock_generate:
            mock_generate.return_value = OperationResult(
                success=True,
                data={'summary': 'test'},
                processing_time=0.1
            )
            
            results = []
            async for result in wrapper.process_batch(files, chunk_size=2):
                results.append(result)
            
            assert len(results) == 3
            assert all(result['success'] for result in results)
    
    @pytest.mark.asyncio
    async def test_process_batch_with_errors(self, tmp_path, sample_dataframe):
        """Test batch processing with some errors."""
        # Create files (one will fail)
        files = []
        for i in range(2):
            csv_file = tmp_path / f"test_{i}.csv"
            sample_dataframe.to_csv(csv_file, index=False)
            files.append(csv_file)
        
        wrapper = TabularSummaryWrapper()
        
        with patch.object(wrapper, 'generate_summary') as mock_generate:
            # First file succeeds, second fails
            mock_generate.side_effect = [
                OperationResult(success=True, data={'summary': 'test'}, processing_time=0.1),
                Exception("Processing failed")
            ]
            
            results = []
            async for result in wrapper.process_batch(files, continue_on_error=True):
                results.append(result)
            
            assert len(results) == 2
            assert results[0]['success'] is True
            assert results[1]['success'] is False
            assert 'error' in results[1]
    
    def test_get_performance_metrics(self):
        """Test performance metrics retrieval."""
        wrapper = TabularSummaryWrapper()
        
        metrics = wrapper.get_performance_metrics()
        
        assert 'cache_size' in metrics
        assert 'cache_hit_rate' in metrics
        assert 'memory_usage' in metrics
        assert 'executor_threads' in metrics
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test wrapper as async context manager."""
        async with TabularSummaryWrapper() as wrapper:
            assert wrapper is not None
            assert hasattr(wrapper, '_executor')
        
        # After exiting context, resources should be cleaned up
        assert wrapper._executor._shutdown
    
    @pytest.mark.asyncio
    async def test_progress_callbacks(self, tmp_path, sample_dataframe):
        """Test progress callbacks during batch processing."""
        # Create test files
        files = []
        for i in range(3):
            csv_file = tmp_path / f"test_{i}.csv"
            sample_dataframe.to_csv(csv_file, index=False)
            files.append(csv_file)
        
        # Mock progress callback
        progress_calls = []
        
        class MockProgressCallback:
            async def on_progress(self, current: int, total: int, operation: str):
                progress_calls.append((current, total, operation))
        
        wrapper = TabularSummaryWrapper()
        wrapper.add_progress_callback(MockProgressCallback())
        
        with patch.object(wrapper, 'generate_summary') as mock_generate:
            mock_generate.return_value = OperationResult(
                success=True,
                data={'summary': 'test'},
                processing_time=0.1
            )
            
            results = []
            async for result in wrapper.process_batch(files):
                results.append(result)
        
        # Should have received progress callbacks
        assert len(progress_calls) == 3
        assert progress_calls[-1][0] == 3  # Final call should be for 3rd file
        assert all(call[1] == 3 for call in progress_calls)  # Total should always be 3


class TestFactoryFunction:
    """Test cases for factory function."""
    
    def test_create_tabular_summary_wrapper_defaults(self):
        """Test factory function with defaults."""
        wrapper = create_tabular_summary_wrapper()
        
        assert isinstance(wrapper, TabularSummaryWrapper)
        assert wrapper.config is not None
    
    def test_create_tabular_summary_wrapper_with_config(self, sample_config):
        """Test factory function with custom config."""
        wrapper = create_tabular_summary_wrapper(config=sample_config)
        
        assert isinstance(wrapper, TabularSummaryWrapper)
        assert wrapper.config == sample_config


class TestIntegrationScenarios:
    """Integration test scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, tmp_path, sample_dataframe):
        """Test complete workflow from file to sidecar template."""
        # Create CSV file
        csv_file = tmp_path / "events.csv"
        sample_dataframe.to_csv(csv_file, index=False)
        
        # Create wrapper
        config = TabularSummaryConfig(skip_columns=['onset', 'duration'])
        async with create_tabular_summary_wrapper(config=config) as wrapper:
            # Load data
            df = await wrapper.load_data(csv_file)
            assert isinstance(df, pd.DataFrame)
            
            # Generate summary (mocked for testing)
            with patch.object(wrapper, '_generate_summary_async') as mock_generate:
                mock_generate.return_value = {
                    'summary': {'columns': ['trial_type', 'response']},
                    'template': {
                        'trial_type': {'HED': {'go': 'Event/go', 'stop': 'Event/stop'}},
                        'response': {'HED': {'left': 'Response/left', 'right': 'Response/right'}}
                    },
                    'column_definitions': ['trial_type', 'response'],
                    'unique_values': {
                        'trial_type': ['go', 'stop'],
                        'response': ['left', 'right']
                    }
                }
                
                summary_result = await wrapper.generate_summary(df)
                assert summary_result.success
                
                # Extract sidecar template
                template = await wrapper.extract_sidecar_template(df)
                assert isinstance(template, SidecarTemplate)
                assert len(template.generated_columns) == 2
                assert 'trial_type' in template.template
                assert 'response' in template.template


if __name__ == "__main__":
    pytest.main([__file__]) 