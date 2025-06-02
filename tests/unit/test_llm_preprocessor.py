"""Unit tests for LLM Preprocessor."""

import pytest
import pandas as pd
import numpy as np

from hed_tools.tools.llm_preprocessor import (
    LLMPreprocessor,
    SamplingConfig,
    ColumnSample,
    ColumnClassification,
    create_llm_preprocessor,
    process_for_llm_classification,
)


class TestSamplingConfig:
    """Test the SamplingConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SamplingConfig()

        assert config.max_tokens == 512
        assert config.max_samples_per_column == 50
        assert config.min_samples_per_column == 5
        assert config.quantiles == [0.0, 0.25, 0.5, 0.75, 1.0]
        assert config.include_extremes is True
        assert config.preserve_distribution is True
        assert config.random_seed == 42

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SamplingConfig(
            max_tokens=1024, max_samples_per_column=100, quantiles=[0.1, 0.5, 0.9]
        )

        assert config.max_tokens == 1024
        assert config.max_samples_per_column == 100
        assert config.quantiles == [0.1, 0.5, 0.9]


class TestColumnClassification:
    """Test column classification functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = LLMPreprocessor()

    def test_skip_patterns(self):
        """Test skip pattern detection."""
        # Test BIDS timing columns
        onset_data = pd.Series([0.1, 0.5, 1.2, 2.0])
        assert (
            self.preprocessor.classify_column("onset", onset_data)
            == ColumnClassification.SKIP_COLUMN
        )

        duration_data = pd.Series([0.2, 0.3, 0.1, 0.4])
        assert (
            self.preprocessor.classify_column("duration", duration_data)
            == ColumnClassification.SKIP_COLUMN
        )

        # Test ID columns
        id_data = pd.Series([1, 2, 3, 4])
        assert (
            self.preprocessor.classify_column("subject_id", id_data)
            == ColumnClassification.SKIP_COLUMN
        )

        # Test index columns
        index_data = pd.Series([0, 1, 2, 3])
        assert (
            self.preprocessor.classify_column("sample", index_data)
            == ColumnClassification.SKIP_COLUMN
        )

    def test_value_patterns(self):
        """Test value pattern detection."""
        # Test trial type
        trial_data = pd.Series(["go", "nogo", "go", "nogo"])
        assert (
            self.preprocessor.classify_column("trial_type", trial_data)
            == ColumnClassification.VALUE_COLUMN
        )

        # Test response columns
        response_data = pd.Series(["left", "right", "left", "right"])
        assert (
            self.preprocessor.classify_column("response", response_data)
            == ColumnClassification.VALUE_COLUMN
        )

        # Test stimulus columns
        stim_data = pd.Series(["face", "house", "face", "house"])
        assert (
            self.preprocessor.classify_column("stimulus_type", stim_data)
            == ColumnClassification.VALUE_COLUMN
        )

    def test_data_driven_classification(self):
        """Test data-driven classification based on uniqueness."""
        # High uniqueness -> skip
        unique_data = pd.Series(range(100))  # 100 unique values out of 100
        assert (
            self.preprocessor.classify_column("unknown_col", unique_data)
            == ColumnClassification.SKIP_COLUMN
        )

        # Low uniqueness -> value
        categorical_data = pd.Series(
            ["A"] * 50 + ["B"] * 50
        )  # 2 unique values out of 100
        assert (
            self.preprocessor.classify_column("unknown_col", categorical_data)
            == ColumnClassification.VALUE_COLUMN
        )

        # Medium uniqueness -> value (default)
        medium_data = pd.Series(
            ["A"] * 30 + ["B"] * 30 + ["C"] * 20 + ["D"] * 20
        )  # 4 unique out of 100
        assert (
            self.preprocessor.classify_column("unknown_col", medium_data)
            == ColumnClassification.VALUE_COLUMN
        )


class TestSamplingStrategies:
    """Test different sampling strategies."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = LLMPreprocessor(SamplingConfig(random_seed=42))

    def test_stratified_sampling(self):
        """Test stratified sampling for categorical data."""
        # Create categorical data with different frequencies
        data = pd.Series(["A"] * 50 + ["B"] * 30 + ["C"] * 20)

        samples, metadata = self.preprocessor.stratified_sample(data, max_samples=10)

        assert len(samples) <= 10
        assert metadata["method"] == "stratified"
        assert metadata["total_unique"] == 3
        assert "distribution" in metadata

        # Check that most frequent category is represented
        assert "A" in samples

    def test_quantile_sampling(self):
        """Test quantile-based sampling for numeric data."""
        # Create numeric data
        np.random.seed(42)
        data = pd.Series(np.random.normal(100, 15, 1000))

        samples, metadata = self.preprocessor.quantile_sample(data, max_samples=20)

        assert len(samples) <= 20
        assert metadata["method"] == "quantile"
        assert "min" in metadata
        assert "max" in metadata
        assert "mean" in metadata
        assert "std" in metadata

        # Check that quantiles are included
        assert len(set(samples)) > 1  # Should have different values

    def test_reservoir_sampling(self):
        """Test reservoir sampling for large datasets."""
        # Create large dataset
        data = pd.Series(range(1000))

        samples, metadata = self.preprocessor.reservoir_sample(data, max_samples=50)

        assert len(samples) == 50
        assert metadata["method"] == "reservoir"
        assert metadata["original_size"] == 1000
        assert metadata["sample_size"] == 50
        assert "sampling_ratio" in metadata

    def test_text_sampling(self):
        """Test text sampling strategy."""
        # Create text data with different frequencies
        texts = (
            ["hello world"] * 30
            + ["goodbye"] * 20
            + ["test"] * 10
            + ["unique text here"] * 5
        )
        data = pd.Series(texts)

        samples, metadata = self.preprocessor.text_sample(data, max_samples=10)

        assert len(samples) <= 10
        assert metadata["method"] == "text"
        assert "unique_values" in metadata
        assert "avg_length" in metadata
        assert "avg_words" in metadata
        assert "length_range" in metadata

        # Check that most common text is included
        assert "hello world" in samples

    def test_empty_data_handling(self):
        """Test handling of empty or null data."""
        empty_data = pd.Series([])
        samples, metadata = self.preprocessor.stratified_sample(
            empty_data, max_samples=10
        )
        assert samples == []
        assert metadata["empty"] is True

        null_data = pd.Series([None, np.nan, None])
        samples, metadata = self.preprocessor.quantile_sample(null_data, max_samples=10)
        assert samples == []
        assert metadata["empty"] is True


class TestColumnSampling:
    """Test column sampling functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = LLMPreprocessor()

        # Create sample DataFrame
        self.df = pd.DataFrame(
            {
                "onset": [0.1, 0.5, 1.2, 2.0, 3.1],
                "duration": [0.2, 0.3, 0.1, 0.4, 0.2],
                "trial_type": ["go", "nogo", "go", "nogo", "go"],
                "response_time": [0.45, 0.67, 0.52, 0.78, 0.43],
                "accuracy": [1, 0, 1, 1, 1],
                "subject_id": [1, 1, 1, 1, 1],
                "stimulus_file": [
                    "face1.jpg",
                    "house1.jpg",
                    "face2.jpg",
                    "house2.jpg",
                    "face1.jpg",
                ],
            }
        )

    def test_sample_numeric_column(self):
        """Test sampling of numeric columns."""
        sample = self.preprocessor.sample_column(
            "response_time", self.df["response_time"]
        )

        assert sample.name == "response_time"
        assert sample.column_type == "numeric"
        assert sample.classification == ColumnClassification.SKIP_COLUMN
        assert len(sample.sample_values) > 0
        assert sample.statistics["method"] == "quantile"
        assert sample.token_count > 0

    def test_sample_categorical_column(self):
        """Test sampling of categorical columns."""
        sample = self.preprocessor.sample_column("trial_type", self.df["trial_type"])

        assert sample.name == "trial_type"
        assert sample.column_type == "categorical"
        assert sample.classification == ColumnClassification.VALUE_COLUMN
        assert len(sample.sample_values) > 0
        assert sample.token_count > 0

    def test_sample_skip_column(self):
        """Test sampling of columns that should be skipped."""
        sample = self.preprocessor.sample_column("onset", self.df["onset"])

        assert sample.name == "onset"
        assert sample.classification == ColumnClassification.SKIP_COLUMN
        assert len(sample.sample_values) > 0

    def test_sample_text_column(self):
        """Test sampling of text columns."""
        sample = self.preprocessor.sample_column(
            "stimulus_file", self.df["stimulus_file"]
        )

        assert sample.name == "stimulus_file"
        assert sample.column_type in ["text", "categorical"]
        assert sample.classification == ColumnClassification.VALUE_COLUMN
        assert len(sample.sample_values) > 0
        assert sample.token_count > 0


class TestDataFrameProcessing:
    """Test DataFrame processing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = LLMPreprocessor()

        # Create comprehensive test DataFrame
        self.df = pd.DataFrame(
            {
                "onset": [0.1, 0.5, 1.2, 2.0, 3.1],
                "duration": [0.2, 0.3, 0.1, 0.4, 0.2],
                "trial_type": ["go", "nogo", "go", "nogo", "go"],
                "response_time": [0.45, 0.67, 0.52, 0.78, 0.43],
                "accuracy": [1, 0, 1, 1, 1],
                "subject_id": [1, 1, 1, 1, 1],
                "stimulus_file": [
                    "face1.jpg",
                    "house1.jpg",
                    "face2.jpg",
                    "house2.jpg",
                    "face1.jpg",
                ],
                "condition": ["A", "B", "A", "B", "A"],
                "response": ["left", "right", "left", "right", "left"],
            }
        )

    def test_process_dataframe(self):
        """Test processing of entire DataFrame."""
        samples = self.preprocessor.process_dataframe(self.df)

        assert len(samples) == len(self.df.columns)

        # Check that all columns are represented
        column_names = [sample.name for sample in samples]
        assert set(column_names) == set(self.df.columns)

        # Check classifications
        value_columns = [
            s for s in samples if s.classification == ColumnClassification.VALUE_COLUMN
        ]
        skip_columns = [
            s for s in samples if s.classification == ColumnClassification.SKIP_COLUMN
        ]

        assert len(value_columns) > 0
        assert len(skip_columns) > 0

        # Specific checks
        trial_type_sample = next(s for s in samples if s.name == "trial_type")
        assert trial_type_sample.classification == ColumnClassification.VALUE_COLUMN

        onset_sample = next(s for s in samples if s.name == "onset")
        assert onset_sample.classification == ColumnClassification.SKIP_COLUMN

    def test_error_handling(self):
        """Test error handling for problematic columns."""
        # Create DataFrame with problematic column
        df_with_issues = pd.DataFrame(
            {
                "normal_col": [1, 2, 3],
                "problem_col": [{"a": 1}, {"b": 2}, {"c": 3}],  # Complex objects
            }
        )

        samples = self.preprocessor.process_dataframe(df_with_issues)

        assert len(samples) == 2

        # Check that error is handled gracefully
        problem_sample = next(s for s in samples if s.name == "problem_col")
        # Should still create a sample, even if it's marked as error
        assert problem_sample is not None


class TestLLMFormatting:
    """Test LLM formatting functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = LLMPreprocessor(SamplingConfig(max_tokens=500))

        self.df = pd.DataFrame(
            {
                "onset": [0.1, 0.5, 1.2, 2.0],
                "trial_type": ["go", "nogo", "go", "nogo"],
                "response_time": [0.45, 0.67, 0.52, 0.78],
                "subject_id": [1, 1, 1, 1],
            }
        )

    def test_format_for_llm(self):
        """Test formatting samples for LLM consumption."""
        samples = self.preprocessor.process_dataframe(self.df)
        formatted_output = self.preprocessor.format_for_llm(samples)

        assert isinstance(formatted_output, str)
        assert len(formatted_output) > 0

        # Check structure
        assert "COLUMN ANALYSIS FOR LLM CLASSIFICATION" in formatted_output
        assert "VALUE COLUMNS" in formatted_output
        assert "SKIP COLUMNS" in formatted_output
        assert "Total estimated tokens:" in formatted_output

        # Check that value columns come first
        value_index = formatted_output.find("VALUE COLUMNS")
        skip_index = formatted_output.find("SKIP COLUMNS")
        assert value_index < skip_index

    def test_token_limit_respected(self):
        """Test that token limits are respected."""
        # Create large dataset
        large_df = pd.DataFrame(
            {f"col_{i}": [f"value_{i}_{j}" for j in range(100)] for i in range(20)}
        )

        config = SamplingConfig(max_tokens=100)  # Very small limit
        preprocessor = LLMPreprocessor(config)

        samples = preprocessor.process_dataframe(large_df)
        formatted_output = preprocessor.format_for_llm(samples)

        # Should not exceed token limit significantly
        estimated_tokens = len(formatted_output) // 4  # Rough estimate
        assert estimated_tokens <= config.max_tokens * 1.2  # Allow some overhead

    def test_column_section_formatting(self):
        """Test individual column section formatting."""
        sample = ColumnSample(
            name="test_column",
            column_type="categorical",
            classification=ColumnClassification.VALUE_COLUMN,
            sample_values=["A", "B", "C"],
            statistics={"count": 100, "unique": 3, "non_null": 95},
            metadata={},
            token_count=50,
        )

        section = self.preprocessor._format_column_section(sample)

        assert "Column: test_column" in section
        assert "Type: categorical" in section
        assert "Classification: value_column" in section
        assert "Count: 100" in section
        assert "Sample values: A, B, C" in section

    def test_token_estimation(self):
        """Test token count estimation."""
        tokens = self.preprocessor._estimate_tokens(
            "test_col", ["value1", "value2", "value3"], {"count": 100, "unique": 3}
        )

        assert isinstance(tokens, int)
        assert tokens > 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_llm_preprocessor(self):
        """Test preprocessor creation function."""
        preprocessor = create_llm_preprocessor()
        assert isinstance(preprocessor, LLMPreprocessor)

        config = SamplingConfig(max_tokens=1024)
        preprocessor_with_config = create_llm_preprocessor(config)
        assert preprocessor_with_config.config.max_tokens == 1024

    def test_process_for_llm_classification(self):
        """Test convenience function for LLM classification."""
        df = pd.DataFrame(
            {
                "trial_type": ["go", "nogo", "go"],
                "response_time": [0.5, 0.7, 0.6],
                "onset": [0.1, 0.5, 1.0],
            }
        )

        formatted_output, samples = process_for_llm_classification(df, max_tokens=256)

        assert isinstance(formatted_output, str)
        assert isinstance(samples, list)
        assert len(samples) == 3
        assert "COLUMN ANALYSIS" in formatted_output


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        preprocessor = LLMPreprocessor()

        samples = preprocessor.process_dataframe(empty_df)
        assert samples == []

    def test_single_column_dataframe(self):
        """Test handling of single column DataFrame."""
        single_col_df = pd.DataFrame({"col1": [1, 2, 3]})
        preprocessor = LLMPreprocessor()

        samples = preprocessor.process_dataframe(single_col_df)
        assert len(samples) == 1
        assert samples[0].name == "col1"

    def test_all_null_column(self):
        """Test handling of column with all null values."""
        null_df = pd.DataFrame({"null_col": [None, None, None]})
        preprocessor = LLMPreprocessor()

        samples = preprocessor.process_dataframe(null_df)
        assert len(samples) == 1
        sample = samples[0]
        assert sample.statistics["non_null"] == 0

    def test_very_small_token_limit(self):
        """Test behavior with very small token limits."""
        df = pd.DataFrame(
            {"col1": ["a", "b", "c"], "col2": [1, 2, 3], "col3": ["x", "y", "z"]}
        )

        config = SamplingConfig(max_tokens=10)  # Extremely small
        preprocessor = LLMPreprocessor(config)

        samples = preprocessor.process_dataframe(df)
        formatted_output = preprocessor.format_for_llm(samples)

        # Should still produce some output
        assert len(formatted_output) > 0
        assert "COLUMN ANALYSIS" in formatted_output


if __name__ == "__main__":
    pytest.main([__file__])
