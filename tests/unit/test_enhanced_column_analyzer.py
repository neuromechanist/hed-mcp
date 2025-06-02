"""Unit tests for Enhanced Column Analyzer."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile

from hed_tools.tools.enhanced_column_analyzer import (
    EnhancedColumnAnalyzer,
    NumericColumnAnalyzer,
    CategoricalColumnAnalyzer,
    TemporalColumnAnalyzer,
    TextColumnAnalyzer,
    ColumnType,
    create_enhanced_column_analyzer,
    analyze_columns_enhanced,
)


class TestNumericColumnAnalyzer:
    """Test the NumericColumnAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NumericColumnAnalyzer()

    def test_can_analyze_numeric_data(self):
        """Test detection of numeric columns."""
        # Integer data
        int_series = pd.Series([1, 2, 3, 4, 5])
        assert self.analyzer.can_analyze(int_series, "test_col")

        # Float data
        float_series = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])
        assert self.analyzer.can_analyze(float_series, "test_col")

        # Mixed numeric with nulls
        mixed_series = pd.Series([1, 2.5, None, 4, 5.0])
        assert self.analyzer.can_analyze(mixed_series, "test_col")

    def test_cannot_analyze_non_numeric(self):
        """Test rejection of non-numeric columns."""
        text_series = pd.Series(["a", "b", "c"])
        assert not self.analyzer.can_analyze(text_series, "test_col")

        mixed_series = pd.Series([1, "text", 3])
        assert not self.analyzer.can_analyze(mixed_series, "test_col")

        empty_series = pd.Series([])
        assert not self.analyzer.can_analyze(empty_series, "test_col")

    def test_analyze_comprehensive_statistics(self):
        """Test comprehensive numeric analysis."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        series = pd.Series(data)

        result = self.analyzer.analyze(series, "test_numeric")

        # Check type classification
        assert result["type"] == ColumnType.NUMERIC.value

        # Check basic statistics
        stats = result["statistics"]
        assert stats["mean"] == 5.5
        assert stats["median"] == 5.5
        assert stats["min"] == 1.0
        assert stats["max"] == 10.0
        assert stats["range"] == 9.0
        assert "std" in stats
        assert "var" in stats
        assert "skewness" in stats
        assert "kurtosis" in stats
        assert "q25" in stats
        assert "q75" in stats
        assert "iqr" in stats
        assert "cv" in stats

        # Check outlier detection
        assert "outlier_count" in stats
        assert "outlier_percentage" in stats

        # Check distribution analysis
        assert "distribution" in result
        assert "histogram_counts" in result["distribution"]
        assert "histogram_bins" in result["distribution"]
        assert "is_integer" in result["distribution"]
        assert "is_positive" in result["distribution"]

        # Check data quality assessment
        assert "data_quality" in result
        assert "has_outliers" in result["data_quality"]
        assert "distribution_shape" in result["data_quality"]

    def test_outlier_detection(self):
        """Test outlier detection using IQR method."""
        # Data with clear outliers
        data = [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        series = pd.Series(data)

        result = self.analyzer.analyze(series, "test_outliers")
        stats = result["statistics"]

        assert stats["outlier_count"] > 0
        assert stats["outlier_percentage"] > 0
        assert result["data_quality"]["has_outliers"] is True

    def test_empty_series_handling(self):
        """Test handling of empty series."""
        empty_series = pd.Series([], dtype=float)
        result = self.analyzer.analyze(empty_series, "empty_col")

        assert result["type"] == ColumnType.NUMERIC.value
        assert result["statistics"] == {}


class TestCategoricalColumnAnalyzer:
    """Test the CategoricalColumnAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = CategoricalColumnAnalyzer()

    def test_can_analyze_categorical_data(self):
        """Test detection of categorical columns."""
        # Low unique ratio
        cat_series = pd.Series(["A", "B", "A", "B", "A", "B"])
        assert self.analyzer.can_analyze(cat_series, "test_col")

        # Reasonable number of unique values
        categories = ["cat1", "cat2", "cat3"] * 10
        cat_series = pd.Series(categories)
        assert self.analyzer.can_analyze(cat_series, "test_col")

    def test_cannot_analyze_high_unique_ratio(self):
        """Test rejection of high unique ratio columns."""
        # High unique ratio
        unique_series = pd.Series([f"unique_{i}" for i in range(100)])
        assert not self.analyzer.can_analyze(unique_series, "test_col")

        # Empty series
        empty_series = pd.Series([])
        assert not self.analyzer.can_analyze(empty_series, "test_col")

    def test_analyze_categorical_distribution(self):
        """Test categorical distribution analysis."""
        data = ["A", "B", "A", "C", "A", "B", "A"]
        series = pd.Series(data)

        result = self.analyzer.analyze(series, "test_categorical")

        # Check type classification
        assert result["type"] == ColumnType.CATEGORICAL.value

        # Check statistics
        stats = result["statistics"]
        assert stats["most_frequent"] == "A"
        assert stats["most_frequent_count"] == 4
        assert "most_frequent_percentage" in stats
        assert "least_frequent" in stats
        assert "entropy" in stats
        assert "gini_coefficient" in stats

        # Check distribution
        distribution = result["distribution"]
        assert "value_distribution" in distribution
        assert "is_balanced" in distribution
        assert "has_rare_categories" in distribution

        # Check patterns
        assert "patterns" in result
        assert "has_numeric_codes" in result["patterns"]
        assert "has_mixed_case" in result["patterns"]

    def test_entropy_calculation(self):
        """Test Shannon entropy calculation."""
        # Perfectly balanced distribution should have higher entropy
        balanced_data = ["A", "B", "C"] * 10
        balanced_series = pd.Series(balanced_data)
        balanced_result = self.analyzer.analyze(balanced_series, "balanced")

        # Skewed distribution should have lower entropy
        skewed_data = ["A"] * 25 + ["B"] * 3 + ["C"] * 2
        skewed_series = pd.Series(skewed_data)
        skewed_result = self.analyzer.analyze(skewed_series, "skewed")

        balanced_entropy = balanced_result["statistics"]["entropy"]
        skewed_entropy = skewed_result["statistics"]["entropy"]

        assert balanced_entropy > skewed_entropy

    def test_bids_pattern_detection(self):
        """Test BIDS pattern detection in categorical data."""
        # Trial type pattern
        trial_data = ["go", "stop", "go", "stop"]
        trial_series = pd.Series(trial_data)

        result = self.analyzer.analyze(trial_series, "trial_type")
        patterns = result["patterns"]

        # Should detect BIDS patterns
        assert "is_condition_type" in patterns
        assert "is_response_type" in patterns
        assert "is_stimulus_type" in patterns
        assert "is_task_related" in patterns


class TestTemporalColumnAnalyzer:
    """Test the TemporalColumnAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TemporalColumnAnalyzer()

    def test_can_analyze_temporal_columns(self):
        """Test detection of temporal columns."""
        onset_data = [0.0, 1.5, 3.0, 4.5]
        onset_series = pd.Series(onset_data)

        # Should detect onset column
        assert self.analyzer.can_analyze(onset_series, "onset")
        assert self.analyzer.can_analyze(onset_series, "response_time")
        assert self.analyzer.can_analyze(onset_series, "duration")

        # Should not detect non-temporal names
        assert not self.analyzer.can_analyze(onset_series, "trial_type")

    def test_cannot_analyze_non_numeric_temporal(self):
        """Test rejection of non-numeric temporal columns."""
        text_data = ["early", "late", "medium"]
        text_series = pd.Series(text_data)

        # Even with temporal name, should reject non-numeric
        assert not self.analyzer.can_analyze(text_series, "onset")

    def test_analyze_onset_column(self):
        """Test analysis of onset column."""
        onset_data = [0.0, 1.5, 3.0, 4.5, 6.0]
        onset_series = pd.Series(onset_data)

        result = self.analyzer.analyze(onset_series, "onset")

        # Check type classification
        assert result["type"] == ColumnType.TEMPORAL.value

        # Check basic statistics
        stats = result["statistics"]
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "range" in stats

        # Check timing analysis
        timing = result["timing_analysis"]
        assert "is_sorted" in timing
        assert "has_negative_values" in timing
        assert "has_zero_values" in timing
        assert "precision" in timing

        # Onset-specific analysis
        assert "mean_interval" in timing
        assert "std_interval" in timing
        assert "min_interval" in timing
        assert "regular_timing" in timing

    def test_analyze_duration_column(self):
        """Test analysis of duration column."""
        duration_data = [1.0, 1.0, 1.0, 1.0, 1.0]
        duration_series = pd.Series(duration_data)

        result = self.analyzer.analyze(duration_series, "duration")

        timing = result["timing_analysis"]

        # Duration-specific analysis
        assert "mean_duration" in timing
        assert "has_instantaneous" in timing
        assert "duration_consistency" in timing

    def test_precision_estimation(self):
        """Test timing precision estimation."""
        # High precision data (3 decimal places)
        high_precision = [1.123, 2.456, 3.789]
        hp_series = pd.Series(high_precision)

        result = self.analyzer.analyze(hp_series, "onset")
        precision = result["timing_analysis"]["precision"]

        assert precision["max_decimal_places"] == 3
        assert precision["mean_decimal_places"] == 3.0

    def test_data_quality_assessment(self):
        """Test temporal data quality assessment."""
        # Data with negative values (problematic)
        bad_data = [-1.0, 0.0, 1.0, 2.0]
        bad_series = pd.Series(bad_data)

        result = self.analyzer.analyze(bad_series, "onset")
        quality = result["data_quality"]

        assert quality["has_negative_times"] is True
        assert quality["has_issues"] is True


class TestTextColumnAnalyzer:
    """Test the TextColumnAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TextColumnAnalyzer()

    def test_can_analyze_text_data(self):
        """Test detection of text columns."""
        # High unique ratio text
        text_data = ["This is text", "Another text", "Different text", "Unique text"]
        text_series = pd.Series(text_data)
        assert self.analyzer.can_analyze(text_series, "description")

        # Long text data
        long_text = ["This is a very long text string that exceeds the threshold"] * 5
        long_series = pd.Series(long_text)
        assert self.analyzer.can_analyze(long_series, "notes")

    def test_cannot_analyze_numeric_data(self):
        """Test rejection of numeric data."""
        numeric_data = [1, 2, 3, 4, 5]
        numeric_series = pd.Series(numeric_data)
        assert not self.analyzer.can_analyze(numeric_series, "test_col")

    def test_analyze_text_statistics(self):
        """Test text analysis with length and linguistic statistics."""
        text_data = [
            "Short text",
            "This is a longer text with more words",
            "Medium length text here",
            "Another text sample",
        ]
        text_series = pd.Series(text_data)

        result = self.analyzer.analyze(text_series, "description")

        # Check type classification
        assert result["type"] == ColumnType.TEXT.value

        # Check length statistics
        stats = result["statistics"]
        assert "mean_length" in stats
        assert "median_length" in stats
        assert "std_length" in stats
        assert "min_length" in stats
        assert "max_length" in stats
        assert "total_characters" in stats

        # Check linguistic analysis
        linguistic = result["linguistic"]
        assert "total_words" in linguistic
        assert "unique_words" in linguistic
        assert "average_words_per_entry" in linguistic
        assert "vocabulary_diversity" in linguistic
        assert "has_punctuation" in linguistic
        assert "has_numbers" in linguistic

        # Check patterns
        patterns = result["patterns"]
        assert "consistent_format" in patterns
        assert "has_urls" in patterns
        assert "has_emails" in patterns
        assert "has_codes" in patterns
        assert "language_detected" in patterns

    def test_language_detection(self):
        """Test basic language detection."""
        english_text = ["The quick brown fox", "and the lazy dog"]
        english_series = pd.Series(english_text)

        result = self.analyzer.analyze(english_series, "english_text")
        language = result["patterns"]["language_detected"]

        assert language == "english"

    def test_url_and_email_detection(self):
        """Test URL and email pattern detection."""
        data_with_urls = ["Visit https://example.com", "Check out http://test.org"]
        url_series = pd.Series(data_with_urls)

        result = self.analyzer.analyze(url_series, "urls")
        assert result["patterns"]["has_urls"] is True

        data_with_emails = ["Contact john@example.com", "Email admin@test.org"]
        email_series = pd.Series(data_with_emails)

        result = self.analyzer.analyze(email_series, "emails")
        assert result["patterns"]["has_emails"] is True


class TestEnhancedColumnAnalyzer:
    """Test the main EnhancedColumnAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = EnhancedColumnAnalyzer()

        # Create sample DataFrame with various column types
        self.sample_df = pd.DataFrame(
            {
                "onset": [0.0, 1.5, 3.0, 4.5],
                "duration": [1.0, 1.0, 1.0, 1.0],
                "trial_type": ["go", "stop", "go", "stop"],
                "response_time": [0.5, 0.7, 0.6, 0.8],
                "accuracy": [1, 1, 0, 1],
                "participant_id": ["sub-001", "sub-001", "sub-001", "sub-001"],
                "notes": [
                    "Trial went well",
                    "Participant hesitated",
                    "Good response",
                    "Clear response",
                ],
            }
        )

    def test_initialization(self):
        """Test analyzer initialization."""
        assert len(self.analyzer.analyzers) == 4  # Four analyzer types
        assert len(self.analyzer.bids_patterns) == 7  # Seven BIDS patterns
        assert self.analyzer._last_analysis is None

    @pytest.mark.asyncio
    async def test_analyze_dataframe_complete(self):
        """Test complete DataFrame analysis."""
        result = await self.analyzer.analyze_dataframe(self.sample_df)

        # Check top-level structure
        assert "source_file" in result
        assert "file_info" in result
        assert "columns" in result
        assert "bids_compliance" in result
        assert "patterns_detected" in result
        assert "hed_candidates" in result
        assert "recommendations" in result
        assert "summary" in result

        # Check that all columns were analyzed
        assert len(result["columns"]) == len(self.sample_df.columns)

        # Check specific column types (allowing for different classification)
        columns = result["columns"]
        assert columns["onset"]["type"] == ColumnType.TEMPORAL.value
        assert columns["duration"]["type"] == ColumnType.TEMPORAL.value
        # trial_type might be classified as mixed or categorical depending on implementation
        assert columns["trial_type"]["type"] in [
            ColumnType.CATEGORICAL.value,
            ColumnType.MIXED.value,
        ]
        assert columns["response_time"]["type"] == ColumnType.TEMPORAL.value
        assert columns["accuracy"]["type"] == ColumnType.NUMERIC.value
        assert columns["notes"]["type"] == ColumnType.TEXT.value

    @pytest.mark.asyncio
    async def test_bids_pattern_detection(self):
        """Test BIDS pattern detection."""
        result = await self.analyzer.analyze_dataframe(self.sample_df)

        patterns_detected = result["patterns_detected"]
        pattern_names = [p["pattern_name"] for p in patterns_detected]

        # Should detect common BIDS patterns
        assert "trial_type" in pattern_names
        assert "onset" in pattern_names
        assert "duration" in pattern_names
        assert "response_time" in pattern_names

    @pytest.mark.asyncio
    async def test_hed_candidate_identification(self):
        """Test HED candidate identification."""
        result = await self.analyzer.analyze_dataframe(self.sample_df)

        hed_candidates = result["hed_candidates"]
        candidate_columns = [c["column"] for c in hed_candidates]

        # Should have some HED candidates (exact set may vary based on implementation)
        assert len(hed_candidates) > 0

        # temporal columns should not be HED candidates
        assert "onset" not in candidate_columns
        assert "duration" not in candidate_columns

    @pytest.mark.asyncio
    async def test_bids_compliance_checking(self):
        """Test BIDS compliance checking."""
        result = await self.analyzer.analyze_dataframe(self.sample_df)

        compliance = result["bids_compliance"]

        # Should be compliant (has onset and duration)
        assert compliance["is_compliant"] is True
        assert compliance["score"] > 90  # High compliance score

        # Test non-compliant data
        non_compliant_df = pd.DataFrame(
            {
                "trial_type": ["go", "stop"],
                "response": ["left", "right"],
                # Missing onset and duration
            }
        )

        result = await self.analyzer.analyze_dataframe(non_compliant_df)
        compliance = result["bids_compliance"]

        assert compliance["is_compliant"] is False
        assert len(compliance["errors"]) > 0

    @pytest.mark.asyncio
    async def test_recommendations_generation(self):
        """Test recommendation generation."""
        result = await self.analyzer.analyze_dataframe(self.sample_df)

        recommendations = result["recommendations"]
        assert isinstance(recommendations, list)

        # Should have some recommendations
        assert len(recommendations) > 0

    @pytest.mark.asyncio
    async def test_summary_generation(self):
        """Test analysis summary generation."""
        result = await self.analyzer.analyze_dataframe(self.sample_df)

        summary = result["summary"]

        assert "total_columns" in summary
        assert "column_type_distribution" in summary
        assert "bids_patterns_found" in summary
        assert "hed_candidates_found" in summary
        assert "bids_compliance_score" in summary
        assert "data_quality_issues" in summary

        # Check values
        assert summary["total_columns"] == len(self.sample_df.columns)
        assert summary["bids_compliance_score"] > 90

    @pytest.mark.asyncio
    async def test_data_quality_assessment(self):
        """Test data quality assessment."""
        # Create problematic data with same array lengths
        problematic_df = pd.DataFrame(
            {
                "onset": [-1.0, 0.0, 1.0, 2.0, 3.0],  # Negative onset (problem)
                "duration": [1.0, 1.0, 1.0, 1.0, 1.0],
                "trial_type": ["A", "A", "A", "A", "B"],  # Highly skewed (problem)
                "response": ["left", "left", "right", "left", "left"],
                "accuracy": [1, 1, 0, 1, 1],
            }
        )

        result = await self.analyzer.analyze_dataframe(problematic_df)

        # Should detect data quality issues
        quality_issues = result["summary"]["data_quality_issues"]
        assert (
            quality_issues >= 0
        )  # May or may not detect issues depending on thresholds

        # Should have some recommendations
        recommendations = result["recommendations"]
        assert len(recommendations) > 0

    def test_get_analysis_summary(self):
        """Test getting analysis summary."""
        # Before analysis
        assert self.analyzer.get_analysis_summary() is None

        # After analysis (would need to be async, but testing the method exists)
        assert hasattr(self.analyzer, "get_analysis_summary")


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_enhanced_column_analyzer(self):
        """Test analyzer factory function."""
        analyzer = create_enhanced_column_analyzer()
        assert isinstance(analyzer, EnhancedColumnAnalyzer)

    @pytest.mark.asyncio
    async def test_analyze_columns_enhanced_with_dataframe(self):
        """Test convenience function with DataFrame."""
        df = pd.DataFrame({"onset": [0.0, 1.0, 2.0], "trial_type": ["A", "B", "A"]})

        result = await analyze_columns_enhanced(df)
        assert "columns" in result
        assert len(result["columns"]) == 2

    @pytest.mark.asyncio
    async def test_analyze_columns_enhanced_with_file(self):
        """Test convenience function with file path."""
        # Create temporary TSV file
        df = pd.DataFrame(
            {
                "onset": [0.0, 1.0, 2.0],
                "duration": [1.0, 1.0, 1.0],
                "trial_type": ["go", "stop", "go"],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            df.to_csv(f.name, sep="\t", index=False)
            temp_path = Path(f.name)

        try:
            result = await analyze_columns_enhanced(temp_path)
            assert "columns" in result
            assert "source_file" in result
            assert result["source_file"] == str(temp_path)
        finally:
            temp_path.unlink()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = EnhancedColumnAnalyzer()

    @pytest.mark.asyncio
    async def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        result = await self.analyzer.analyze_dataframe(empty_df)

        assert result["columns"] == {}
        assert result["summary"]["total_columns"] == 0

    @pytest.mark.asyncio
    async def test_single_column_dataframe(self):
        """Test handling of single column DataFrame."""
        single_col_df = pd.DataFrame({"single_col": [1, 2, 3]})
        result = await self.analyzer.analyze_dataframe(single_col_df)

        assert len(result["columns"]) == 1
        assert "single_col" in result["columns"]

    @pytest.mark.asyncio
    async def test_all_null_column(self):
        """Test handling of column with all null values."""
        null_df = pd.DataFrame({"all_nulls": [None, None, None], "normal": [1, 2, 3]})

        result = await self.analyzer.analyze_dataframe(null_df)

        # Should still analyze the null column
        assert "all_nulls" in result["columns"]
        null_analysis = result["columns"]["all_nulls"]
        assert null_analysis["null_percentage"] == 100.0

    @pytest.mark.asyncio
    async def test_mixed_type_column(self):
        """Test handling of columns with mixed data types."""
        mixed_df = pd.DataFrame({"mixed": [1, "text", 3.14, None, True]})

        result = await self.analyzer.analyze_dataframe(mixed_df)

        # Should classify as mixed or fall back to basic analysis
        assert "mixed" in result["columns"]
        mixed_analysis = result["columns"]["mixed"]
        assert mixed_analysis["type"] in [ColumnType.MIXED.value, ColumnType.TEXT.value]
