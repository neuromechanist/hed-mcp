"""LLM Preprocessor for column data sampling and preparation.

This module provides intelligent sampling techniques to prepare column data
for LLM classification while respecting context limits and maintaining
representative data distributions.
"""

import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class ColumnClassification(Enum):
    """Column classification for LLM processing."""

    VALUE_COLUMN = "value_column"
    SKIP_COLUMN = "skip_column"


@dataclass
class SamplingConfig:
    """Configuration for sampling parameters."""

    max_tokens: int = 512
    max_samples_per_column: int = 50
    min_samples_per_column: int = 5
    quantiles: List[float] = None
    include_extremes: bool = True
    preserve_distribution: bool = True
    random_seed: Optional[int] = 42

    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]


@dataclass
class ColumnSample:
    """Sampled data for a single column."""

    name: str
    column_type: str
    classification: ColumnClassification
    sample_values: List[Any]
    statistics: Dict[str, Any]
    metadata: Dict[str, Any]
    token_count: int


class LLMPreprocessor:
    """Preprocessor for preparing column data for LLM classification.

    This class implements various sampling strategies to create representative
    samples of column data while maintaining distributional properties and
    respecting token limits for LLM processing.
    """

    def __init__(self, config: Optional[SamplingConfig] = None):
        """Initialize the LLM preprocessor.

        Args:
            config: Sampling configuration parameters
        """
        self.config = config or SamplingConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

        # Pattern-based classification rules
        self.skip_patterns = [
            r"^(onset|duration)$",  # BIDS timing columns
            r"^(sample|index|row)$",  # Index columns
            r"_id$",  # ID columns
            r"^(file|path|url)$",  # File reference columns
        ]

        self.value_patterns = [
            r"^(trial_type|condition)$",  # Experimental conditions
            r"^(response|key|button)$",  # Response columns
            r"^(stimulus|stim)_",  # Stimulus columns
            r"^(accuracy|correct)$",  # Accuracy columns
            r"_time$",  # Response time columns
        ]

    def classify_column(self, name: str, data: pd.Series) -> ColumnClassification:
        """Classify a column as value or skip column.

        Args:
            name: Column name
            data: Column data

        Returns:
            Column classification
        """
        name_lower = name.lower()

        # Check skip patterns first
        for pattern in self.skip_patterns:
            if re.match(pattern, name_lower):
                return ColumnClassification.SKIP_COLUMN

        # Check value patterns
        for pattern in self.value_patterns:
            if re.match(pattern, name_lower):
                return ColumnClassification.VALUE_COLUMN

        # Data-driven classification
        unique_ratio = data.nunique() / len(data)

        # High uniqueness suggests skip column (IDs, timestamps, etc.)
        if unique_ratio > 0.9:
            return ColumnClassification.SKIP_COLUMN

        # Low uniqueness suggests value column (categorical)
        if unique_ratio < 0.1:
            return ColumnClassification.VALUE_COLUMN

        # Default to value column for moderate uniqueness
        return ColumnClassification.VALUE_COLUMN

    def stratified_sample(
        self, data: pd.Series, max_samples: int
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Perform stratified sampling for categorical data.

        Args:
            data: Categorical data series
            max_samples: Maximum number of samples to return

        Returns:
            Tuple of (sampled values, sampling metadata)
        """
        value_counts = data.value_counts()
        total_samples = min(max_samples, len(value_counts))

        if total_samples <= 0:
            return [], {"method": "stratified", "empty": True}

        # Calculate proportional sample sizes
        proportions = value_counts / value_counts.sum()
        samples = []
        metadata = {
            "method": "stratified",
            "total_unique": len(value_counts),
            "sample_size": total_samples,
            "distribution": dict(proportions.head(10)),
        }

        # Sample proportionally, ensuring at least one of each major category
        for value, count in value_counts.head(total_samples).items():
            if self.config.preserve_distribution:
                n_samples = max(1, int(proportions[value] * total_samples))
            else:
                n_samples = 1

            samples.extend(
                [value] * min(n_samples, len(samples) + n_samples - len(samples))
            )

            if len(samples) >= total_samples:
                break

        return samples[:total_samples], metadata

    def quantile_sample(
        self, data: pd.Series, max_samples: int
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Perform quantile-based sampling for numeric data.

        Args:
            data: Numeric data series
            max_samples: Maximum number of samples to return

        Returns:
            Tuple of (sampled values, sampling metadata)
        """
        # Remove NaN values for quantile calculation
        clean_data = data.dropna()

        if len(clean_data) == 0:
            return [], {"method": "quantile", "empty": True}

        samples = []
        metadata = {
            "method": "quantile",
            "quantiles_used": self.config.quantiles,
            "min": float(clean_data.min()),
            "max": float(clean_data.max()),
            "mean": float(clean_data.mean()),
            "std": float(clean_data.std()),
        }

        # Get quantile values
        quantile_values = clean_data.quantile(self.config.quantiles).tolist()
        samples.extend(quantile_values)

        # Add some random samples if we have room
        remaining_samples = max_samples - len(samples)
        if remaining_samples > 0:
            random_samples = clean_data.sample(
                min(remaining_samples, len(clean_data))
            ).tolist()
            samples.extend(random_samples)

        # Remove duplicates while preserving order
        seen = set()
        unique_samples = []
        for value in samples:
            if value not in seen:
                seen.add(value)
                unique_samples.append(value)

        return unique_samples[:max_samples], metadata

    def reservoir_sample(
        self, data: pd.Series, max_samples: int
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Perform reservoir sampling for large datasets.

        Args:
            data: Data series
            max_samples: Maximum number of samples to return

        Returns:
            Tuple of (sampled values, sampling metadata)
        """
        if len(data) <= max_samples:
            return data.dropna().tolist(), {
                "method": "reservoir",
                "full_sample": True,
                "original_size": len(data),
            }

        # Reservoir sampling algorithm
        reservoir = []

        for i, value in enumerate(data):
            if pd.isna(value):
                continue

            if len(reservoir) < max_samples:
                reservoir.append(value)
            else:
                # Replace with decreasing probability
                j = random.randint(0, i)
                if j < max_samples:
                    reservoir[j] = value

        metadata = {
            "method": "reservoir",
            "original_size": len(data),
            "sample_size": len(reservoir),
            "sampling_ratio": len(reservoir) / len(data),
        }

        return reservoir, metadata

    def text_sample(
        self, data: pd.Series, max_samples: int
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Sample text data including both common and rare examples.

        Args:
            data: Text data series
            max_samples: Maximum number of samples to return

        Returns:
            Tuple of (sampled values, sampling metadata)
        """
        # Clean and filter text data
        clean_data = data.dropna().astype(str)
        clean_data = clean_data[clean_data.str.strip() != ""]

        if len(clean_data) == 0:
            return [], {"method": "text", "empty": True}

        # Calculate text statistics
        lengths = clean_data.str.len()
        word_counts = clean_data.str.split().str.len()

        # Get frequency distribution
        value_counts = clean_data.value_counts()

        samples = []
        metadata = {
            "method": "text",
            "unique_values": len(value_counts),
            "avg_length": float(lengths.mean()),
            "avg_words": float(word_counts.mean()),
            "length_range": [int(lengths.min()), int(lengths.max())],
        }

        # Include most common examples
        common_samples = min(max_samples // 2, len(value_counts))
        samples.extend(value_counts.head(common_samples).index.tolist())

        # Include rare examples (if we have room)
        remaining = max_samples - len(samples)
        if remaining > 0 and len(value_counts) > common_samples:
            rare_samples = min(remaining, len(value_counts) - common_samples)
            samples.extend(value_counts.tail(rare_samples).index.tolist())

        return samples[:max_samples], metadata

    def sample_column(self, name: str, data: pd.Series) -> ColumnSample:
        """Sample a single column based on its characteristics.

        Args:
            name: Column name
            data: Column data

        Returns:
            ColumnSample with sampled data and metadata
        """
        classification = self.classify_column(name, data)

        # Determine column type
        if pd.api.types.is_numeric_dtype(data):
            column_type = "numeric"
        elif isinstance(data.dtype, pd.CategoricalDtype) or data.nunique() < 50:
            column_type = "categorical"
        else:
            column_type = "text"

        # Determine sampling strategy
        if column_type == "numeric":
            samples, stats = self.quantile_sample(
                data, self.config.max_samples_per_column
            )
        elif column_type == "categorical":
            samples, stats = self.stratified_sample(
                data, self.config.max_samples_per_column
            )
        elif column_type == "text":
            samples, stats = self.text_sample(data, self.config.max_samples_per_column)
        else:
            # Fallback to reservoir sampling
            samples, stats = self.reservoir_sample(
                data, self.config.max_samples_per_column
            )

        # Calculate basic statistics
        basic_stats = {
            "count": len(data),
            "non_null": data.notna().sum(),
            "unique": data.nunique(),
            "missing_ratio": data.isna().mean(),
        }

        # Combine statistics
        all_stats = {**basic_stats, **stats}

        # Estimate token count (rough approximation)
        token_count = self._estimate_tokens(name, samples, all_stats)

        return ColumnSample(
            name=name,
            column_type=column_type,
            classification=classification,
            sample_values=samples,
            statistics=all_stats,
            metadata={"sampling_config": self.config.__dict__},
            token_count=token_count,
        )

    def process_dataframe(self, df: pd.DataFrame) -> List[ColumnSample]:
        """Process an entire DataFrame and sample all columns.

        Args:
            df: Input DataFrame

        Returns:
            List of ColumnSample objects
        """
        results = []

        for column in df.columns:
            try:
                sample = self.sample_column(column, df[column])
                results.append(sample)
            except Exception as e:
                # Create error sample for problematic columns
                error_sample = ColumnSample(
                    name=column,
                    column_type="error",
                    classification=ColumnClassification.SKIP_COLUMN,
                    sample_values=[],
                    statistics={"error": str(e)},
                    metadata={},
                    token_count=0,
                )
                results.append(error_sample)

        return results

    def format_for_llm(self, samples: List[ColumnSample]) -> str:
        """Format sampled data for LLM consumption.

        Args:
            samples: List of column samples

        Returns:
            Formatted string for LLM input
        """
        output_parts = []
        total_tokens = 0

        # Sort by classification and importance
        value_columns = [
            s for s in samples if s.classification == ColumnClassification.VALUE_COLUMN
        ]
        skip_columns = [
            s for s in samples if s.classification == ColumnClassification.SKIP_COLUMN
        ]

        output_parts.append("COLUMN ANALYSIS FOR LLM CLASSIFICATION\n")
        output_parts.append("=" * 50 + "\n")

        # Process value columns first (more important)
        if value_columns:
            output_parts.append("\nVALUE COLUMNS (recommended for HED annotation):\n")
            for sample in value_columns:
                section = self._format_column_section(sample)
                if total_tokens + sample.token_count < self.config.max_tokens:
                    output_parts.append(section)
                    total_tokens += sample.token_count
                else:
                    break

        # Add skip columns if we have room
        if skip_columns and total_tokens < self.config.max_tokens * 0.8:
            output_parts.append(
                "\nSKIP COLUMNS (not recommended for HED annotation):\n"
            )
            for sample in skip_columns:
                section = self._format_column_section(sample)
                if total_tokens + sample.token_count < self.config.max_tokens:
                    output_parts.append(section)
                    total_tokens += sample.token_count
                else:
                    break

        output_parts.append(f"\n\nTotal estimated tokens: {total_tokens}")
        return "".join(output_parts)

    def _format_column_section(self, sample: ColumnSample) -> str:
        """Format a single column sample for LLM display.

        Args:
            sample: Column sample to format

        Returns:
            Formatted string section
        """
        lines = [f"\nColumn: {sample.name}"]
        lines.append(f"Type: {sample.column_type}")
        lines.append(f"Classification: {sample.classification.value}")

        # Add key statistics
        stats = sample.statistics
        if "count" in stats:
            lines.append(f"Count: {stats['count']} (non-null: {stats['non_null']})")
        if "unique" in stats:
            lines.append(f"Unique values: {stats['unique']}")

        # Add type-specific statistics
        if sample.column_type == "numeric" and "mean" in stats:
            lines.append(
                f"Range: {stats.get('min', 'N/A')} - {stats.get('max', 'N/A')}"
            )
            lines.append(f"Mean: {stats['mean']:.3f}")

        # Add sample values
        if sample.sample_values:
            sample_str = ", ".join([str(v)[:50] for v in sample.sample_values[:10]])
            lines.append(f"Sample values: {sample_str}")

        lines.append("-" * 30)
        return "\n".join(lines) + "\n"

    def _estimate_tokens(
        self, name: str, samples: List[Any], stats: Dict[str, Any]
    ) -> int:
        """Estimate token count for a column sample.

        Args:
            name: Column name
            samples: Sample values
            stats: Statistics dictionary

        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token ≈ 4 characters
        text_content = f"{name} "
        text_content += " ".join([str(v) for v in samples[:10]])
        text_content += " ".join(
            [f"{k}:{v}" for k, v in stats.items() if isinstance(v, (int, float, str))]
        )

        return len(text_content) // 4


def create_llm_preprocessor(config: Optional[SamplingConfig] = None) -> LLMPreprocessor:
    """Create an LLM preprocessor instance.

    Args:
        config: Optional sampling configuration

    Returns:
        LLMPreprocessor instance
    """
    return LLMPreprocessor(config)


def process_for_llm_classification(
    df: pd.DataFrame, max_tokens: int = 512
) -> Tuple[str, List[ColumnSample]]:
    """Convenience function to process DataFrame for LLM classification.

    Args:
        df: Input DataFrame
        max_tokens: Maximum tokens for LLM input

    Returns:
        Tuple of (formatted string, list of column samples)
    """
    config = SamplingConfig(max_tokens=max_tokens)
    preprocessor = create_llm_preprocessor(config)
    samples = preprocessor.process_dataframe(df)
    formatted_output = preprocessor.format_for_llm(samples)

    return formatted_output, samples
