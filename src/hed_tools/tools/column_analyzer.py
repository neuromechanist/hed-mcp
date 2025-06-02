"""BIDS events file column analysis and HED annotation tools.

This module provides functionality to analyze BIDS events files, classify columns,
and prepare them for HED annotation through automated sidecar generation.
Enhanced with Strategy pattern and specialized column type analyzers.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class ColumnType(Enum):
    """Enhanced column type classification."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXT = "text"
    HED_TAGS = "hed_tags"
    BOOLEAN = "boolean"
    IDENTIFIER = "identifier"
    MIXED = "mixed"
    EMPTY = "empty"


@dataclass
class ColumnPattern:
    """BIDS column pattern definition."""

    name: str
    patterns: List[str]  # Regex patterns to match column names
    type_hint: ColumnType
    priority: str  # "high", "medium", "low"
    description: str


class ColumnAnalysisStrategy(Protocol):
    """Protocol for column analysis strategies."""

    def can_analyze(self, series: pd.Series, column_name: str) -> bool:
        """Check if this strategy can analyze the given column."""
        ...

    def analyze(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Perform specialized analysis for this column type."""
        ...


class BaseColumnAnalyzer(ABC):
    """Base class for column type analyzers."""

    @abstractmethod
    def can_analyze(self, series: pd.Series, column_name: str) -> bool:
        """Check if this analyzer can handle the given column."""
        pass

    @abstractmethod
    def analyze(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Perform specialized analysis for this column type."""
        pass

    def _get_basic_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Get basic statistics common to all column types."""
        return {
            "total_count": len(series),
            "null_count": int(series.isnull().sum()),
            "non_null_count": int(series.count()),
            "null_percentage": float(series.isnull().sum() / len(series) * 100),
            "unique_count": int(series.nunique()),
            "unique_percentage": float(series.nunique() / len(series) * 100)
            if len(series) > 0
            else 0.0,
        }


class NumericColumnAnalyzer(BaseColumnAnalyzer):
    """Analyzer for numeric columns with vectorized numpy operations."""

    def can_analyze(self, series: pd.Series, column_name: str) -> bool:
        """Check if column contains numeric data."""
        if series.empty:
            return False
        try:
            pd.to_numeric(series.dropna(), errors="raise")
            return True
        except (ValueError, TypeError):
            return False

    def analyze(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Analyze numeric column with comprehensive statistics."""
        clean_series = pd.to_numeric(series.dropna(), errors="coerce")
        values = clean_series.values

        basic_stats = self._get_basic_stats(series)

        if len(values) == 0:
            return {**basic_stats, "type": ColumnType.NUMERIC.value, "statistics": {}}

        # Vectorized numpy operations for efficiency
        statistics = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "var": float(np.var(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
            "skewness": float(self._calculate_skewness(values)),
            "kurtosis": float(self._calculate_kurtosis(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "cv": float(np.std(values) / np.mean(values))
            if np.mean(values) != 0
            else 0.0,
        }

        # Check for outliers using IQR method
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        outlier_mask = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
        statistics["outlier_count"] = int(np.sum(outlier_mask))
        statistics["outlier_percentage"] = float(
            np.sum(outlier_mask) / len(values) * 100
        )

        # Distribution analysis
        distribution_info = self._analyze_distribution(values)

        return {
            **basic_stats,
            "type": ColumnType.NUMERIC.value,
            "statistics": statistics,
            "distribution": distribution_info,
            "data_quality": self._assess_numeric_quality(values, statistics),
        }

    def _calculate_skewness(self, values: np.ndarray) -> float:
        """Calculate skewness using numpy."""
        if len(values) < 3:
            return 0.0
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        return np.mean(((values - mean) / std) ** 3)

    def _calculate_kurtosis(self, values: np.ndarray) -> float:
        """Calculate kurtosis using numpy."""
        if len(values) < 4:
            return 0.0
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        return np.mean(((values - mean) / std) ** 4) - 3

    def _analyze_distribution(self, values: np.ndarray) -> Dict[str, Any]:
        """Analyze distribution characteristics."""
        hist, bin_edges = np.histogram(values, bins=min(50, len(np.unique(values))))

        return {
            "histogram_counts": hist.tolist(),
            "histogram_bins": bin_edges.tolist(),
            "is_integer": bool(np.all(values == values.astype(int))),
            "is_positive": bool(np.all(values >= 0)),
            "is_binary": bool(set(values.astype(int)) <= {0, 1})
            if np.all(values == values.astype(int))
            else False,
        }

    def _assess_numeric_quality(
        self, values: np.ndarray, stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess data quality for numeric columns."""
        return {
            "has_outliers": stats["outlier_count"] > 0,
            "outlier_severity": "high" if stats["outlier_percentage"] > 5 else "low",
            "distribution_shape": self._classify_distribution_shape(stats),
            "precision_issues": bool(
                np.any(np.abs(values - np.round(values, 10)) > 1e-10)
            ),
        }

    def _classify_distribution_shape(self, stats: Dict[str, Any]) -> str:
        """Classify distribution shape based on skewness and kurtosis."""
        skew = abs(stats["skewness"])

        if skew < 0.5:
            return "normal"
        elif skew < 1.0:
            return "moderately_skewed"
        elif skew < 2.0:
            return "highly_skewed"
        else:
            return "extremely_skewed"


class CategoricalColumnAnalyzer(BaseColumnAnalyzer):
    """Analyzer for categorical columns."""

    def can_analyze(self, series: pd.Series, column_name: str) -> bool:
        """Check if column contains categorical data."""
        if series.empty:
            return False

        clean_series = series.dropna()
        unique_ratio = (
            clean_series.nunique() / len(clean_series) if len(clean_series) > 0 else 0
        )

        # Consider categorical if:
        # 1. Low unique ratio (< 0.5)
        # 2. Reasonable number of unique values (2-100)
        # 3. Not purely numeric (unless looks like coded categories)
        return (
            unique_ratio < 0.5
            and 2 <= clean_series.nunique() <= 100
            and (
                not pd.api.types.is_numeric_dtype(clean_series)
                or clean_series.nunique() <= 20
            )
        )

    def analyze(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Analyze categorical column with distribution and pattern analysis."""
        basic_stats = self._get_basic_stats(series)
        clean_series = series.dropna()

        if clean_series.empty:
            return {
                **basic_stats,
                "type": ColumnType.CATEGORICAL.value,
                "statistics": {},
            }

        # Value distribution analysis
        value_counts = clean_series.value_counts()

        statistics = {
            "most_frequent": str(value_counts.index[0])
            if not value_counts.empty
            else None,
            "most_frequent_count": int(value_counts.iloc[0])
            if not value_counts.empty
            else 0,
            "most_frequent_percentage": (
                float(value_counts.iloc[0] / len(clean_series) * 100)
                if not value_counts.empty
                else 0.0
            ),
            "least_frequent": str(value_counts.index[-1])
            if not value_counts.empty
            else None,
            "least_frequent_count": int(value_counts.iloc[-1])
            if not value_counts.empty
            else 0,
            "entropy": float(self._calculate_entropy(value_counts.values)),
            "gini_coefficient": float(self._calculate_gini(value_counts.values)),
        }

        # Distribution analysis
        distribution = {
            "value_distribution": dict(value_counts.to_dict()),
            "is_balanced": self._is_balanced_distribution(value_counts),
            "has_rare_categories": bool(
                np.any(value_counts < len(clean_series) * 0.01)
            ),  # < 1% frequency
            "dominant_category_percentage": float(
                value_counts.iloc[0] / len(clean_series) * 100
            ),
        }

        # Pattern analysis
        patterns = self._analyze_categorical_patterns(clean_series)

        return {
            **basic_stats,
            "type": ColumnType.CATEGORICAL.value,
            "statistics": statistics,
            "distribution": distribution,
            "patterns": patterns,
            "data_quality": self._assess_categorical_quality(clean_series, statistics),
        }

    def _calculate_entropy(self, counts: np.ndarray) -> float:
        """Calculate Shannon entropy for categorical distribution."""
        if len(counts) <= 1:
            return 0.0
        probabilities = counts / np.sum(counts)
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        return -np.sum(probabilities * np.log2(probabilities))

    def _calculate_gini(self, counts: np.ndarray) -> float:
        """Calculate Gini coefficient for distribution inequality."""
        if len(counts) <= 1:
            return 0.0
        sorted_counts = np.sort(counts)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    def _is_balanced_distribution(self, value_counts: pd.Series) -> bool:
        """Check if categorical distribution is reasonably balanced."""
        if len(value_counts) <= 1:
            return True

        # Calculate coefficient of variation for counts
        cv = value_counts.std() / value_counts.mean()
        return cv < 1.0  # Arbitrary threshold for "balanced"

    def _analyze_categorical_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze patterns in categorical values."""
        values = series.astype(str).tolist()

        patterns = {
            "has_numeric_codes": bool(re.search(r"^\d+$", "|".join(values))),
            "has_mixed_case": any(v != v.lower() and v != v.upper() for v in values),
            "has_special_chars": bool(re.search(r"[^\w\s]", "|".join(values))),
            "average_length": float(np.mean([len(str(v)) for v in values])),
            "max_length": max(len(str(v)) for v in values),
            "min_length": min(len(str(v)) for v in values),
        }

        # Check for common BIDS patterns
        patterns.update(self._detect_bids_patterns(values))

        return patterns

    def _detect_bids_patterns(self, values: List[str]) -> Dict[str, bool]:
        """Detect common BIDS categorical patterns."""
        values_str = "|".join(values).lower()

        return {
            "is_condition_type": any(
                term in values_str for term in ["condition", "trial_type", "stimulus"]
            ),
            "is_response_type": any(
                term in values_str for term in ["response", "key", "button"]
            ),
            "is_stimulus_type": any(
                term in values_str for term in ["stim", "image", "sound", "visual"]
            ),
            "is_task_related": any(
                term in values_str for term in ["task", "block", "run"]
            ),
        }

    def _assess_categorical_quality(
        self, series: pd.Series, stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess data quality for categorical columns."""
        return {
            "has_rare_categories": stats["least_frequent_count"] < len(series) * 0.01,
            "is_highly_skewed": stats["most_frequent_percentage"] > 90,
            "entropy_level": "high"
            if stats["entropy"] > 2.0
            else "medium"
            if stats["entropy"] > 1.0
            else "low",
            "distribution_quality": "good"
            if 10 <= stats["most_frequent_percentage"] <= 90
            else "poor",
        }


class TemporalColumnAnalyzer(BaseColumnAnalyzer):
    """Analyzer for temporal columns (onset, duration, timing)."""

    TEMPORAL_PATTERNS = [
        r".*onset.*",
        r".*duration.*",
        r".*time.*",
        r".*latency.*",
        r".*delay.*",
        r".*interval.*",
        r".*timestamp.*",
    ]

    def can_analyze(self, series: pd.Series, column_name: str) -> bool:
        """Check if column contains temporal data."""
        # Check column name patterns
        name_match = any(
            re.match(pattern, column_name.lower()) for pattern in self.TEMPORAL_PATTERNS
        )

        # Check if numeric (temporal data should be numeric)
        try:
            pd.to_numeric(series.dropna(), errors="raise")
            is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False

        return name_match and is_numeric

    def analyze(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Analyze temporal column with timing-specific metrics."""
        basic_stats = self._get_basic_stats(series)
        clean_series = pd.to_numeric(series.dropna(), errors="coerce")

        if clean_series.empty:
            return {**basic_stats, "type": ColumnType.TEMPORAL.value, "statistics": {}}

        values = clean_series.values

        # Basic temporal statistics
        statistics = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
            "total_duration": float(np.sum(values))
            if "duration" in column_name.lower()
            else None,
        }

        # Timing-specific analysis
        timing_analysis = self._analyze_timing_patterns(values, column_name)

        return {
            **basic_stats,
            "type": ColumnType.TEMPORAL.value,
            "statistics": statistics,
            "timing_analysis": timing_analysis,
            "data_quality": self._assess_temporal_quality(values, column_name),
        }

    def _analyze_timing_patterns(
        self, values: np.ndarray, column_name: str
    ) -> Dict[str, Any]:
        """Analyze timing-specific patterns."""
        analysis = {
            "is_sorted": bool(np.all(values[:-1] <= values[1:])),
            "has_negative_values": bool(np.any(values < 0)),
            "has_zero_values": bool(np.any(values == 0)),
            "precision": self._estimate_precision(values),
        }

        if "onset" in column_name.lower():
            # Analyze inter-onset intervals
            if len(values) > 1:
                intervals = np.diff(np.sort(values))
                analysis.update(
                    {
                        "mean_interval": float(np.mean(intervals)),
                        "std_interval": float(np.std(intervals)),
                        "min_interval": float(np.min(intervals)),
                        "regular_timing": bool(
                            np.std(intervals) < np.mean(intervals) * 0.1
                        ),
                    }
                )

        elif "duration" in column_name.lower():
            # Duration-specific analysis
            analysis.update(
                {
                    "mean_duration": float(np.mean(values)),
                    "has_instantaneous": bool(np.any(values == 0)),
                    "duration_consistency": float(1 - np.std(values) / np.mean(values))
                    if np.mean(values) > 0
                    else 0,
                }
            )

        return analysis

    def _estimate_precision(self, values: np.ndarray) -> Dict[str, Any]:
        """Estimate timing precision from decimal places."""
        decimal_places = []
        for val in values:
            val_str = f"{val:.10f}".rstrip("0")
            if "." in val_str:
                decimal_places.append(len(val_str.split(".")[1]))
            else:
                decimal_places.append(0)

        return {
            "mean_decimal_places": float(np.mean(decimal_places)),
            "max_decimal_places": int(np.max(decimal_places)),
            "estimated_precision": float(10 ** -np.mean(decimal_places)),
        }

    def _assess_temporal_quality(
        self, values: np.ndarray, column_name: str
    ) -> Dict[str, Any]:
        """Assess data quality for temporal columns."""
        quality = {
            "has_negative_times": bool(np.any(values < 0)),
            "timing_precision": "high"
            if np.std(values) < 0.001
            else "medium"
            if np.std(values) < 0.01
            else "low",
        }

        if "onset" in column_name.lower():
            quality["onset_order"] = (
                "sorted" if np.all(values[:-1] <= values[1:]) else "unsorted"
            )

        return quality


class TextColumnAnalyzer(BaseColumnAnalyzer):
    """Analyzer for text columns."""

    def can_analyze(self, series: pd.Series, column_name: str) -> bool:
        """Check if column contains text data."""
        if series.empty:
            return False

        clean_series = series.dropna()

        # Consider text if:
        # 1. Not numeric
        # 2. High unique ratio (> 0.5) OR very long average string length
        # 3. Contains actual text (not just single characters/codes)
        try:
            pd.to_numeric(clean_series, errors="raise")
            return False  # Numeric data
        except (ValueError, TypeError):
            pass

        unique_ratio = (
            clean_series.nunique() / len(clean_series) if len(clean_series) > 0 else 0
        )
        avg_length = np.mean([len(str(v)) for v in clean_series])

        return unique_ratio > 0.5 or avg_length > 10

    def analyze(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Analyze text column with linguistic and pattern analysis."""
        basic_stats = self._get_basic_stats(series)
        clean_series = series.dropna().astype(str)

        if clean_series.empty:
            return {**basic_stats, "type": ColumnType.TEXT.value, "statistics": {}}

        # Text length analysis
        lengths = np.array([len(text) for text in clean_series])

        statistics = {
            "mean_length": float(np.mean(lengths)),
            "median_length": float(np.median(lengths)),
            "std_length": float(np.std(lengths)),
            "min_length": int(np.min(lengths)),
            "max_length": int(np.max(lengths)),
            "total_characters": int(np.sum(lengths)),
        }

        # Linguistic analysis
        linguistic = self._analyze_linguistic_patterns(clean_series.tolist())

        # Pattern analysis
        patterns = self._analyze_text_patterns(clean_series.tolist())

        return {
            **basic_stats,
            "type": ColumnType.TEXT.value,
            "statistics": statistics,
            "linguistic": linguistic,
            "patterns": patterns,
            "data_quality": self._assess_text_quality(clean_series, statistics),
        }

    def _analyze_linguistic_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze linguistic patterns in text."""
        all_text = " ".join(texts)
        words = all_text.split()

        return {
            "total_words": len(words),
            "unique_words": len(set(words)),
            "average_words_per_entry": float(len(words) / len(texts))
            if len(texts) > 0
            else 0,
            "vocabulary_diversity": float(len(set(words)) / len(words))
            if len(words) > 0
            else 0,
            "has_punctuation": bool(re.search(r"[^\w\s]", all_text)),
            "has_numbers": bool(re.search(r"\d", all_text)),
        }

    def _analyze_text_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze structural patterns in text."""
        return {
            "consistent_format": self._check_format_consistency(texts),
            "has_urls": bool(any(re.search(r"https?://", text) for text in texts)),
            "has_emails": bool(any(re.search(r"\S+@\S+", text) for text in texts)),
            "has_codes": bool(any(re.search(r"^[A-Z0-9_-]+$", text) for text in texts)),
            "language_detected": self._detect_language_hints(texts),
        }

    def _check_format_consistency(self, texts: List[str]) -> bool:
        """Check if texts follow a consistent format."""
        if len(texts) < 2:
            return True

        # Check length consistency
        lengths = [len(text) for text in texts]
        length_cv = np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0

        return length_cv < 0.5  # Arbitrary threshold

    def _detect_language_hints(self, texts: List[str]) -> str:
        """Detect basic language hints from text."""
        all_text = " ".join(texts).lower()

        # Simple heuristics
        if any(word in all_text for word in ["the", "and", "or", "but"]):
            return "english"
        elif any(word in all_text for word in ["le", "la", "et", "ou"]):
            return "french"
        elif any(word in all_text for word in ["der", "die", "das", "und"]):
            return "german"
        else:
            return "unknown"

    def _assess_text_quality(
        self, series: pd.Series, stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess data quality for text columns."""
        return {
            "length_consistency": "high"
            if stats["std_length"] / stats["mean_length"] < 0.5
            else "low",
            "has_very_short_entries": bool(stats["min_length"] < 3),
            "has_very_long_entries": bool(stats["max_length"] > 1000),
            "completeness": "good"
            if series.notna().sum() / len(series) > 0.95
            else "poor",
        }


class HedTagColumnAnalyzer(BaseColumnAnalyzer):
    """Analyzer for HED tag columns."""

    HED_PATTERNS = [r".*hed.*", r".*tag.*", r".*annotation.*"]

    def can_analyze(self, series: pd.Series, column_name: str) -> bool:
        """Check if column contains HED tags."""
        # Check column name
        name_match = any(
            re.match(pattern, column_name.lower()) for pattern in self.HED_PATTERNS
        )

        if not name_match:
            return False

        # Check content for HED tag patterns
        clean_series = series.dropna().astype(str)
        if clean_series.empty:
            return False

        # Look for HED tag patterns (hierarchical structure with slashes)
        sample_text = " ".join(clean_series.head(10))
        has_hed_structure = bool(re.search(r"\w+/\w+", sample_text))

        return has_hed_structure

    def analyze(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Analyze HED tag column with tag-specific metrics."""
        basic_stats = self._get_basic_stats(series)
        clean_series = series.dropna().astype(str)

        if clean_series.empty:
            return {**basic_stats, "type": ColumnType.HED_TAGS.value, "statistics": {}}

        # HED tag analysis
        tag_analysis = self._analyze_hed_tags(clean_series.tolist())

        return {
            **basic_stats,
            "type": ColumnType.HED_TAGS.value,
            "hed_analysis": tag_analysis,
            "data_quality": self._assess_hed_quality(clean_series, tag_analysis),
        }

    def _analyze_hed_tags(self, tags: List[str]) -> Dict[str, Any]:
        """Analyze HED tag structure and content."""
        all_tags = []
        hierarchies = []

        for tag_string in tags:
            # Split by commas to get individual tags
            individual_tags = [tag.strip() for tag in tag_string.split(",")]
            all_tags.extend(individual_tags)

            # Analyze hierarchy depth
            for tag in individual_tags:
                hierarchy_depth = tag.count("/")
                hierarchies.append(hierarchy_depth)

        return {
            "total_tags": len(all_tags),
            "unique_tags": len(set(all_tags)),
            "tag_diversity": float(len(set(all_tags)) / len(all_tags))
            if len(all_tags) > 0
            else 0,
            "average_hierarchy_depth": float(np.mean(hierarchies))
            if hierarchies
            else 0,
            "max_hierarchy_depth": int(np.max(hierarchies)) if hierarchies else 0,
            "common_root_tags": self._find_common_roots(all_tags),
            "tag_complexity": self._assess_tag_complexity(all_tags),
        }

    def _find_common_roots(self, tags: List[str]) -> List[str]:
        """Find most common root tags."""
        roots = []
        for tag in tags:
            if "/" in tag:
                root = tag.split("/")[0]
                roots.append(root)
            else:
                roots.append(tag)

        from collections import Counter

        common_roots = Counter(roots).most_common(5)
        return [root for root, count in common_roots]

    def _assess_tag_complexity(self, tags: List[str]) -> str:
        """Assess complexity of HED tag structure."""
        avg_depth = np.mean([tag.count("/") for tag in tags]) if tags else 0

        if avg_depth < 1:
            return "simple"
        elif avg_depth < 2:
            return "moderate"
        else:
            return "complex"

    def _assess_hed_quality(
        self, series: pd.Series, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess data quality for HED tag columns."""
        return {
            "tag_diversity": analysis.get("tag_diversity", 0),
            "hierarchy_consistency": "good"
            if analysis.get("average_hierarchy_depth", 0) > 0
            else "poor",
            "complexity_level": analysis.get("tag_complexity", "unknown"),
        }


class ColumnAnalyzer:
    """Enhanced column analyzer with strategy pattern and BIDS pattern recognition."""

    def __init__(self):
        """Initialize the enhanced column analyzer."""
        self.analyzers = [
            TemporalColumnAnalyzer(),
            HedTagColumnAnalyzer(),
            NumericColumnAnalyzer(),
            CategoricalColumnAnalyzer(),
            TextColumnAnalyzer(),
        ]

        self.bids_patterns = self._initialize_bids_patterns()
        self._last_analysis = None

    def _initialize_bids_patterns(self) -> List[ColumnPattern]:
        """Initialize BIDS column pattern definitions."""
        return [
            ColumnPattern(
                name="trial_type",
                patterns=[r"^trial_type$", r"^condition$", r"^trial_condition$"],
                type_hint=ColumnType.CATEGORICAL,
                priority="high",
                description="Main experimental condition identifier",
            ),
            ColumnPattern(
                name="onset",
                patterns=[r"^onset$", r"^start_time$"],
                type_hint=ColumnType.TEMPORAL,
                priority="high",
                description="Event onset time in seconds",
            ),
            ColumnPattern(
                name="duration",
                patterns=[r"^duration$", r"^event_duration$"],
                type_hint=ColumnType.TEMPORAL,
                priority="high",
                description="Event duration in seconds",
            ),
            ColumnPattern(
                name="response_time",
                patterns=[r"^response_time$", r"^rt$", r"^reaction_time$"],
                type_hint=ColumnType.TEMPORAL,
                priority="medium",
                description="Response time measurement",
            ),
            ColumnPattern(
                name="accuracy",
                patterns=[r"^accuracy$", r"^correct$", r"^acc$"],
                type_hint=ColumnType.CATEGORICAL,
                priority="medium",
                description="Response accuracy indicator",
            ),
            ColumnPattern(
                name="response",
                patterns=[r"^response$", r"^key_press$", r"^button$"],
                type_hint=ColumnType.CATEGORICAL,
                priority="medium",
                description="Response identifier",
            ),
            ColumnPattern(
                name="stimulus",
                patterns=[r"^stimulus$", r"^stim_type$", r"^stimulus_type$"],
                type_hint=ColumnType.CATEGORICAL,
                priority="high",
                description="Stimulus identifier or type",
            ),
        ]

    async def analyze_dataframe(
        self, df: pd.DataFrame, source_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Analyze complete DataFrame with enhanced column detection."""
        analysis = {
            "source_file": str(source_file) if source_file else None,
            "file_info": self._get_file_info(df, source_file),
            "columns": {},
            "bids_compliance": self._check_bids_compliance(df),
            "patterns_detected": [],
            "hed_candidates": [],
            "recommendations": [],
            "summary": {},
        }

        # Analyze each column
        for column_name in df.columns:
            column_analysis = await self._analyze_single_column(
                df[column_name], column_name
            )
            analysis["columns"][column_name] = column_analysis

            # Detect BIDS patterns
            pattern_match = self._match_bids_pattern(column_name, column_analysis)
            if pattern_match:
                analysis["patterns_detected"].append(pattern_match)

            # Identify HED candidates
            if self._is_hed_candidate(column_name, column_analysis):
                analysis["hed_candidates"].append(
                    {
                        "column": column_name,
                        "type": column_analysis["type"],
                        "priority": self._get_hed_priority(
                            column_name, column_analysis
                        ),
                        "reason": self._get_hed_reason(column_name, column_analysis),
                    }
                )

        # Add summary and recommendations
        analysis["summary"] = self._generate_summary(analysis)
        analysis["recommendations"] = self._generate_recommendations(analysis)

        # Store analysis for later access
        self._last_analysis = analysis

        return analysis

    async def _analyze_single_column(
        self, series: pd.Series, column_name: str
    ) -> Dict[str, Any]:
        """Analyze a single column using appropriate strategy."""
        # Find the first analyzer that can handle this column
        for analyzer in self.analyzers:
            if analyzer.can_analyze(series, column_name):
                return analyzer.analyze(series, column_name)

        # Fallback to basic analysis
        return self._basic_analysis(series, column_name)

    def _basic_analysis(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Fallback basic analysis for unclassified columns."""
        basic_stats = {
            "total_count": len(series),
            "null_count": int(series.isnull().sum()),
            "non_null_count": int(series.count()),
            "null_percentage": float(series.isnull().sum() / len(series) * 100),
            "unique_count": int(series.nunique()),
            "unique_percentage": float(series.nunique() / len(series) * 100)
            if len(series) > 0
            else 0.0,
        }

        return {
            **basic_stats,
            "type": ColumnType.MIXED.value,
            "data_quality": {"classification": "unclassified"},
        }

    def _get_file_info(
        self, df: pd.DataFrame, source_file: Optional[Path]
    ) -> Dict[str, Any]:
        """Get basic file information."""
        info = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
        }

        if source_file and source_file.exists():
            info.update(
                {
                    "file_size_mb": float(source_file.stat().st_size / 1024 / 1024),
                    "file_name": source_file.name,
                }
            )

        return info

    def _match_bids_pattern(
        self, column_name: str, analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Match column against BIDS patterns."""
        for pattern in self.bids_patterns:
            if any(re.match(p, column_name.lower()) for p in pattern.patterns):
                return {
                    "pattern_name": pattern.name,
                    "column_name": column_name,
                    "expected_type": pattern.type_hint.value,
                    "actual_type": analysis["type"],
                    "priority": pattern.priority,
                    "description": pattern.description,
                    "type_match": analysis["type"] == pattern.type_hint.value,
                }
        return None

    def _check_bids_compliance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced BIDS compliance checking."""
        required_columns = ["onset", "duration"]
        compliance = {
            "is_compliant": True,
            "errors": [],
            "warnings": [],
            "score": 100.0,
        }

        # Check required columns
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            compliance["is_compliant"] = False
            compliance["errors"].append(f"Missing required columns: {missing_required}")
            compliance["score"] -= 50

        # Check column naming conventions
        for col in df.columns:
            if not col.islower():
                compliance["warnings"].append(f"Column '{col}' should be lowercase")
                compliance["score"] -= 5

            if " " in col:
                compliance["warnings"].append(
                    f"Column '{col}' should not contain spaces"
                )
                compliance["score"] -= 5

        compliance["score"] = max(0, compliance["score"])
        return compliance

    def _is_hed_candidate(self, column_name: str, analysis: Dict[str, Any]) -> bool:
        """Enhanced HED candidate detection."""
        # Skip temporal and identifier columns
        if analysis["type"] in [ColumnType.TEMPORAL.value, ColumnType.IDENTIFIER.value]:
            return False

        # Skip columns with too many unique values
        if analysis.get("unique_count", 0) > 100:
            return False

        # Skip columns with too few unique values
        if analysis.get("unique_count", 0) < 2:
            return False

        # Prefer categorical columns
        if analysis["type"] == ColumnType.CATEGORICAL.value:
            return True

        # Accept some numeric columns if they look categorical
        if analysis["type"] == ColumnType.NUMERIC.value:
            return analysis.get("unique_count", 0) <= 20

        # Accept text columns with reasonable unique count
        if analysis["type"] == ColumnType.TEXT.value:
            return 2 <= analysis.get("unique_count", 0) <= 50

        return False

    def _get_hed_priority(self, column_name: str, analysis: Dict[str, Any]) -> str:
        """Get HED annotation priority."""
        # Check BIDS patterns first
        pattern_match = self._match_bids_pattern(column_name, analysis)
        if pattern_match:
            return pattern_match["priority"]

        # Default priority based on type and characteristics
        if analysis["type"] == ColumnType.CATEGORICAL.value:
            unique_count = analysis.get("unique_count", 0)
            if 2 <= unique_count <= 10:
                return "high"
            elif unique_count <= 20:
                return "medium"

        return "low"

    def _get_hed_reason(self, column_name: str, analysis: Dict[str, Any]) -> str:
        """Get reason for HED candidacy."""
        if analysis["type"] == ColumnType.CATEGORICAL.value:
            return (
                f"Categorical column with {analysis.get('unique_count', 0)} categories"
            )
        elif analysis["type"] == ColumnType.NUMERIC.value:
            return f"Numeric column with discrete values ({analysis.get('unique_count', 0)} unique)"
        elif analysis["type"] == ColumnType.TEXT.value:
            return f"Text column with structured values ({analysis.get('unique_count', 0)} unique)"
        else:
            return "Column suitable for semantic annotation"

    def _generate_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary."""
        columns = analysis["columns"]

        type_distribution = {}
        for col_analysis in columns.values():
            col_type = col_analysis["type"]
            type_distribution[col_type] = type_distribution.get(col_type, 0) + 1

        return {
            "total_columns": len(columns),
            "column_type_distribution": type_distribution,
            "bids_patterns_found": len(analysis["patterns_detected"]),
            "hed_candidates_found": len(analysis["hed_candidates"]),
            "bids_compliance_score": analysis["bids_compliance"]["score"],
            "data_quality_issues": sum(
                1
                for col in columns.values()
                if col.get("data_quality", {}).get("has_issues", False)
            ),
        }

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # BIDS compliance recommendations
        if not analysis["bids_compliance"]["is_compliant"]:
            recommendations.append(
                "Fix BIDS compliance issues before proceeding with HED annotation"
            )

        # HED annotation recommendations
        high_priority_candidates = [
            c for c in analysis["hed_candidates"] if c["priority"] == "high"
        ]
        if high_priority_candidates:
            recommendations.append(
                f"Prioritize HED annotation for: {', '.join([c['column'] for c in high_priority_candidates])}"
            )

        # Data quality recommendations
        for col_name, col_analysis in analysis["columns"].items():
            quality = col_analysis.get("data_quality", {})
            if quality.get("has_issues"):
                recommendations.append(
                    f"Review data quality issues in column '{col_name}'"
                )

        return recommendations

    def get_analysis_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of the last analysis performed."""
        return getattr(self, "_last_analysis", None)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of the last analysis performed.

        Returns:
            Summary dictionary from the last analysis, or empty dict if no analysis
        """
        last_analysis = getattr(self, "_last_analysis", None)
        if last_analysis and "summary" in last_analysis:
            return last_analysis["summary"]
        return {}

    def suggest_hed_annotations(
        self, analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """Suggest HED annotations for columns based on analysis.

        Args:
            analysis: Optional analysis results. If None, uses last analysis.

        Returns:
            Dictionary mapping column names to suggested HED annotations
        """
        if analysis is None:
            analysis = getattr(self, "_last_analysis", None)

        if not analysis or "columns" not in analysis:
            return {}

        suggestions = {}

        for column_name, column_data in analysis["columns"].items():
            column_suggestions = []

            # Get basic suggestions based on column type
            column_type = column_data.get("type", "")

            if column_type == ColumnType.CATEGORICAL.value:
                # Suggest value-based HED tags for categorical columns
                unique_values = column_data.get("statistics", {}).get(
                    "unique_values", []
                )
                for value in unique_values[:5]:  # Limit to first 5 values
                    column_suggestions.append(f"Label/{value}")

            elif column_type == ColumnType.NUMERIC.value:
                # Suggest numeric HED patterns
                if "reaction" in column_name.lower() or "rt" in column_name.lower():
                    column_suggestions.append("Response-time")
                elif "accuracy" in column_name.lower():
                    column_suggestions.append("Performance")
                else:
                    column_suggestions.append("Parameter")

            elif column_type == ColumnType.TEMPORAL.value:
                # Suggest temporal HED patterns
                if "onset" in column_name.lower():
                    column_suggestions.append("Onset")
                elif "duration" in column_name.lower():
                    column_suggestions.append("Duration")
                else:
                    column_suggestions.append("Temporal-attribute")

            # Add BIDS-specific suggestions
            bids_match = column_data.get("bids_pattern")
            if bids_match:
                pattern_name = bids_match.get("name", "")
                if "trial_type" in pattern_name.lower():
                    column_suggestions.append("Condition")
                elif "response" in pattern_name.lower():
                    column_suggestions.append("Agent-action")
                elif "stimulus" in pattern_name.lower():
                    column_suggestions.append("Sensory-event")

            # Add HED candidate suggestions
            if column_data.get("hed_candidate", {}).get("is_candidate", False):
                reason = column_data.get("hed_candidate", {}).get("reason", "")
                if "categorical" in reason:
                    column_suggestions.append("(Condition, Label/#)")
                elif "temporal" in reason:
                    column_suggestions.append("(Event, (Onset, Label/#))")

            # Ensure we have at least a basic suggestion
            if not column_suggestions:
                column_suggestions.append("Label/#")

            suggestions[column_name] = column_suggestions

        return suggestions

    async def analyze_events_file(self, file_path: Union[Path, str]) -> Dict[str, Any]:
        """Analyze a BIDS events file for column patterns and HED annotation opportunities.

        Args:
            file_path: Path to the events TSV file

        Returns:
            Dictionary containing analysis results
        """
        file_path = Path(file_path)
        df = pd.read_csv(file_path, sep="\t")

        # Perform analysis and store results
        analysis = await self.analyze_dataframe(df, file_path)
        self._last_analysis = analysis

        return analysis


# Convenience functions
def create_column_analyzer() -> ColumnAnalyzer:
    """Create a new ColumnAnalyzer instance.

    Returns:
        Configured ColumnAnalyzer instance
    """
    return ColumnAnalyzer()


# Create alias for backward compatibility
BIDSColumnAnalyzer = ColumnAnalyzer


async def analyze_columns(
    data: Union[pd.DataFrame, Path, str], source_file: Optional[Path] = None
) -> Dict[str, Any]:
    """Convenience function for column analysis.

    Args:
        data: DataFrame, file path, or file content to analyze
        source_file: Optional source file path for metadata

    Returns:
        Complete analysis results
    """
    analyzer = create_column_analyzer()

    if isinstance(data, pd.DataFrame):
        return await analyzer.analyze_dataframe(data, source_file)
    elif isinstance(data, (Path, str)):
        return await analyzer.analyze_file(Path(data))
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


async def analyze_events_file(file_path: Union[Path, str]) -> Dict[str, Any]:
    """Analyze events file for BIDS compatibility.

    Args:
        file_path: Path to the events file

    Returns:
        Analysis results including column recommendations
    """
    analyzer = create_column_analyzer()
    return await analyzer.analyze_file(Path(file_path))
