"""Column classification stage for HED sidecar generation pipeline.

This stage handles:
- Automatic column type detection and classification
- Column content analysis and pattern recognition
- HED mapping potential assessment
- Statistical analysis of column data
"""

import logging
import pandas as pd
import re
from typing import Any, Dict, List
from dataclasses import dataclass
from collections import Counter

from . import (
    PipelineStage,
    StageInput,
    StageOutput,
    create_stage_output,
    register_stage,
)

logger = logging.getLogger(__name__)


@dataclass
class ColumnClassification:
    """Classification result for a single column."""

    name: str
    data_type: str
    semantic_type: str
    hed_relevance: float  # 0.0 to 1.0
    value_examples: List[str]
    unique_count: int
    null_count: int
    confidence: float
    patterns: List[str]
    recommendations: List[str]


class ColumnClassificationStage(PipelineStage):
    """Stage for classifying tabular data columns.

    This stage analyzes each column to determine its type, content patterns,
    and potential for HED mapping, providing structured information for
    subsequent pipeline stages.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("column_classification", config)

        # Configuration with defaults
        self.classification_threshold = self.get_config_value(
            "classification_threshold", 0.8
        )
        self.max_examples = self.get_config_value("max_examples", 10)
        self.enable_pattern_detection = self.get_config_value(
            "enable_pattern_detection", True
        )
        self.enable_semantic_analysis = self.get_config_value(
            "enable_semantic_analysis", True
        )

        # Semantic type patterns
        self._semantic_patterns = self._initialize_semantic_patterns()

        # HED-relevant column indicators
        self._hed_keywords = {
            "event",
            "stimulus",
            "response",
            "condition",
            "trial",
            "onset",
            "duration",
            "type",
            "category",
            "label",
            "trigger",
            "marker",
            "tag",
            "description",
            "participant",
            "subject",
            "session",
            "run",
            "task",
            "block",
        }

    def _initialize_semantic_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for semantic type detection."""
        return {
            "identifier": [
                r"^id$",
                r"_id$",
                r"^.*_id$",
                r"^participant",
                r"^subject",
                r"^session",
                r"^run",
                r"^trial",
            ],
            "temporal": [
                r"time",
                r"onset",
                r"duration",
                r"latency",
                r"rt",
                r"response_time",
                r"timestamp",
                r"start",
                r"end",
                r"interval",
            ],
            "categorical": [
                r"type",
                r"category",
                r"condition",
                r"group",
                r"class",
                r"label",
                r"status",
                r"state",
                r"phase",
                r"block",
            ],
            "numerical": [
                r"score",
                r"rating",
                r"value",
                r"amount",
                r"count",
                r"number",
                r"measurement",
                r"metric",
                r"index",
            ],
            "behavioral": [
                r"response",
                r"choice",
                r"accuracy",
                r"correct",
                r"error",
                r"performance",
                r"behavior",
            ],
            "stimulus": [
                r"stimulus",
                r"stim",
                r"image",
                r"sound",
                r"video",
                r"text",
                r"word",
                r"picture",
                r"file",
                r"item",
            ],
            "physiological": [
                r"eeg",
                r"ecg",
                r"emg",
                r"eog",
                r"heart_rate",
                r"blood_pressure",
                r"temperature",
                r"oxygen",
                r"pupil",
            ],
        }

    async def _initialize_implementation(self) -> None:
        """Initialize the column classification stage."""
        self.logger.info(
            f"Initializing column classification: threshold={self.classification_threshold}, "
            f"pattern_detection={self.enable_pattern_detection}"
        )

    async def _execute_implementation(self, stage_input: StageInput) -> StageOutput:
        """Execute column classification for the given DataFrame."""
        dataframe = stage_input.get_data()

        if not isinstance(dataframe, pd.DataFrame):
            return create_stage_output(
                data=None,
                metadata=stage_input.metadata,
                context=stage_input.context,
                errors=["Expected pandas DataFrame input for column classification"],
            )

        try:
            # Classify each column
            classifications = {}
            warnings = []

            for column_name in dataframe.columns:
                classification = await self._classify_column(dataframe, column_name)
                classifications[column_name] = classification

                # Add warnings for low confidence classifications
                if classification.confidence < self.classification_threshold:
                    warnings.append(
                        f"Low confidence classification for column '{column_name}': "
                        f"{classification.confidence:.2f}"
                    )

            # Generate summary statistics
            summary = self._generate_classification_summary(classifications)

            # Update metadata
            classification_metadata = {
                "total_columns": len(classifications),
                "high_confidence_count": sum(
                    1
                    for c in classifications.values()
                    if c.confidence >= self.classification_threshold
                ),
                "hed_relevant_count": sum(
                    1 for c in classifications.values() if c.hed_relevance > 0.5
                ),
                "semantic_type_distribution": summary["semantic_types"],
                "data_type_distribution": summary["data_types"],
            }
            stage_input.metadata.update(classification_metadata)

            # Update context for next stages
            stage_input.context.update(
                {
                    "column_classifications": classifications,
                    "classification_summary": summary,
                    "hed_relevant_columns": [
                        name
                        for name, c in classifications.items()
                        if c.hed_relevance > 0.5
                    ],
                }
            )

            self.logger.info(
                f"Classified {len(classifications)} columns: "
                f"{classification_metadata['hed_relevant_count']} HED-relevant, "
                f"{classification_metadata['high_confidence_count']} high confidence"
            )

            return create_stage_output(
                data=dataframe,  # Pass through the original data
                metadata=stage_input.metadata,
                context=stage_input.context,
                warnings=warnings,
            )

        except Exception as e:
            self.logger.error(f"Column classification failed: {e}")
            return create_stage_output(
                data=None,
                metadata=stage_input.metadata,
                context=stage_input.context,
                errors=[f"Column classification failed: {str(e)}"],
            )

    async def _classify_column(
        self, df: pd.DataFrame, column_name: str
    ) -> ColumnClassification:
        """Classify a single column."""
        column_data = df[column_name]

        # Basic statistics
        unique_count = column_data.nunique()
        null_count = column_data.isnull().sum()
        total_count = len(column_data)

        # Data type detection
        data_type = self._detect_data_type(column_data)

        # Semantic type detection
        semantic_type = self._detect_semantic_type(column_name, column_data)

        # HED relevance assessment
        hed_relevance = self._assess_hed_relevance(
            column_name, column_data, semantic_type
        )

        # Pattern detection
        patterns = []
        if self.enable_pattern_detection:
            patterns = self._detect_patterns(column_data)

        # Generate examples
        value_examples = self._get_value_examples(column_data)

        # Calculate confidence
        confidence = self._calculate_confidence(
            column_name, column_data, semantic_type, data_type, patterns
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            column_name,
            semantic_type,
            hed_relevance,
            unique_count,
            null_count,
            total_count,
        )

        return ColumnClassification(
            name=column_name,
            data_type=data_type,
            semantic_type=semantic_type,
            hed_relevance=hed_relevance,
            value_examples=value_examples,
            unique_count=unique_count,
            null_count=null_count,
            confidence=confidence,
            patterns=patterns,
            recommendations=recommendations,
        )

    def _detect_data_type(self, column_data: pd.Series) -> str:
        """Detect the basic data type of a column."""
        # Check pandas dtype first

        if pd.api.types.is_numeric_dtype(column_data):
            if pd.api.types.is_integer_dtype(column_data):
                return "integer"
            else:
                return "float"
        elif pd.api.types.is_datetime64_any_dtype(column_data):
            return "datetime"
        elif pd.api.types.is_bool_dtype(column_data):
            return "boolean"
        else:
            # Try to infer more specific types for object columns
            non_null_data = column_data.dropna()

            if len(non_null_data) == 0:
                return "empty"

            # Check if all values can be converted to numbers
            try:
                pd.to_numeric(non_null_data)
                return "numeric_string"
            except (ValueError, TypeError):
                pass

            # Check if all values are boolean-like
            boolean_values = {"true", "false", "1", "0", "yes", "no", "y", "n"}
            if all(str(val).lower() in boolean_values for val in non_null_data):
                return "boolean_string"

            # Check for categorical data (low unique count relative to total)
            unique_ratio = non_null_data.nunique() / len(non_null_data)
            if unique_ratio < 0.1 and non_null_data.nunique() < 20:
                return "categorical"

            return "text"

    def _detect_semantic_type(self, column_name: str, column_data: pd.Series) -> str:
        """Detect the semantic type based on column name and content."""
        column_name_lower = column_name.lower()

        # Check name-based patterns
        for semantic_type, patterns in self._semantic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, column_name_lower, re.IGNORECASE):
                    return semantic_type

        # Content-based detection for numeric columns
        if pd.api.types.is_numeric_dtype(column_data):
            non_null_data = column_data.dropna()

            if len(non_null_data) == 0:
                return "unknown"

            # Check value ranges for specific types
            min_val, max_val = non_null_data.min(), non_null_data.max()

            # Time-like values (could be timestamps or durations)
            if min_val >= 0 and max_val > 1000000:  # Large numbers might be timestamps
                return "temporal"

            # Rating scales
            if min_val >= 1 and max_val <= 10 and non_null_data.nunique() <= 10:
                return "numerical"

            # Binary indicators
            if set(non_null_data.unique()).issubset({0, 1}):
                return "categorical"

        # Content-based detection for text columns
        elif column_data.dtype == "object":
            non_null_data = column_data.dropna().astype(str)

            if len(non_null_data) == 0:
                return "unknown"

            # Check for file paths or URLs
            if any(
                "/" in str(val) or "\\" in str(val) or "http" in str(val)
                for val in non_null_data.head(10)
            ):
                return "stimulus"

            # Check for codes or identifiers
            if all(
                len(str(val)) < 20 and any(c.isdigit() for c in str(val))
                for val in non_null_data.head(10)
            ):
                return "identifier"

        return "unknown"

    def _assess_hed_relevance(
        self, column_name: str, column_data: pd.Series, semantic_type: str
    ) -> float:
        """Assess how relevant a column is for HED mapping."""
        relevance_score = 0.0

        # Name-based relevance
        column_name_lower = column_name.lower()
        name_keywords = sum(
            1 for keyword in self._hed_keywords if keyword in column_name_lower
        )
        relevance_score += min(name_keywords * 0.3, 0.6)

        # Semantic type relevance
        semantic_relevance = {
            "stimulus": 0.9,
            "behavioral": 0.8,
            "categorical": 0.7,
            "temporal": 0.6,
            "physiological": 0.8,
            "identifier": 0.3,
            "numerical": 0.4,
            "unknown": 0.1,
        }
        relevance_score += semantic_relevance.get(semantic_type, 0.1)

        # Content-based relevance
        if column_data.dtype == "object":
            non_null_data = column_data.dropna().astype(str)

            # Check for HED-like terms in the data
            hed_terms_in_data = 0
            for val in non_null_data.head(20):
                val_lower = str(val).lower()
                if any(keyword in val_lower for keyword in self._hed_keywords):
                    hed_terms_in_data += 1

            content_relevance = min(
                hed_terms_in_data / min(len(non_null_data), 20) * 0.4, 0.4
            )
            relevance_score += content_relevance

        return min(relevance_score, 1.0)

    def _detect_patterns(self, column_data: pd.Series) -> List[str]:
        """Detect patterns in column data."""
        patterns = []

        if column_data.dtype == "object":
            non_null_data = column_data.dropna().astype(str)

            if len(non_null_data) == 0:
                return patterns

            # Common patterns
            sample_values = non_null_data.head(20)

            # Check for consistent length
            lengths = [len(val) for val in sample_values]
            if len(set(lengths)) == 1:
                patterns.append(f"fixed_length_{lengths[0]}")

            # Check for common separators
            for separator in ["-", "_", "/", "\\", ".", ":"]:
                if all(separator in val for val in sample_values):
                    patterns.append(f"contains_{separator}")

            # Check for numeric patterns
            if all(any(c.isdigit() for c in val) for val in sample_values):
                patterns.append("contains_digits")

            # Check for alphabetic patterns
            if all(any(c.isalpha() for c in val) for val in sample_values):
                patterns.append("contains_letters")

            # Check for uppercase/lowercase patterns
            if all(val.isupper() for val in sample_values):
                patterns.append("all_uppercase")
            elif all(val.islower() for val in sample_values):
                patterns.append("all_lowercase")

        return patterns

    def _get_value_examples(self, column_data: pd.Series) -> List[str]:
        """Get representative examples of column values."""
        non_null_data = column_data.dropna()

        if len(non_null_data) == 0:
            return []

        # Get unique values, prioritizing most common
        value_counts = non_null_data.value_counts()
        examples = []

        for value, count in value_counts.head(self.max_examples).items():
            examples.append(str(value))

        return examples

    def _calculate_confidence(
        self,
        column_name: str,
        column_data: pd.Series,
        semantic_type: str,
        data_type: str,
        patterns: List[str],
    ) -> float:
        """Calculate confidence in the classification."""
        confidence = 0.5  # Base confidence

        # Higher confidence for clear semantic types
        if semantic_type != "unknown":
            confidence += 0.3

        # Higher confidence for consistent data types
        if data_type in ["integer", "float", "datetime", "boolean"]:
            confidence += 0.2

        # Higher confidence for clear patterns
        if patterns:
            confidence += min(len(patterns) * 0.05, 0.2)

        # Higher confidence for clear column names
        if any(keyword in column_name.lower() for keyword in self._hed_keywords):
            confidence += 0.1

        # Lower confidence for high null rates
        null_rate = column_data.isnull().sum() / len(column_data)
        confidence -= null_rate * 0.2

        return min(max(confidence, 0.0), 1.0)

    def _generate_recommendations(
        self,
        column_name: str,
        semantic_type: str,
        hed_relevance: float,
        unique_count: int,
        null_count: int,
        total_count: int,
    ) -> List[str]:
        """Generate recommendations for column handling."""
        recommendations = []

        # HED mapping recommendations
        if hed_relevance > 0.7:
            recommendations.append("High priority for HED mapping")
        elif hed_relevance > 0.4:
            recommendations.append("Consider for HED mapping")
        else:
            recommendations.append("Low priority for HED mapping")

        # Data quality recommendations
        null_rate = null_count / total_count if total_count > 0 else 0
        if null_rate > 0.5:
            recommendations.append("High null rate - consider data cleaning")
        elif null_rate > 0.1:
            recommendations.append("Some missing values - review data quality")

        # Uniqueness recommendations
        unique_rate = unique_count / total_count if total_count > 0 else 0
        if unique_rate > 0.95:
            recommendations.append("Highly unique values - likely identifier")
        elif unique_rate < 0.05:
            recommendations.append("Low unique values - likely categorical")

        # Semantic type specific recommendations
        if semantic_type == "temporal":
            recommendations.append("Validate time format and units")
        elif semantic_type == "stimulus":
            recommendations.append("Check file paths and stimulus definitions")
        elif semantic_type == "categorical":
            recommendations.append("Consider creating value-to-HED mappings")

        return recommendations

    def _generate_classification_summary(
        self, classifications: Dict[str, ColumnClassification]
    ) -> Dict[str, Any]:
        """Generate summary statistics for all classifications."""
        semantic_types = Counter(c.semantic_type for c in classifications.values())
        data_types = Counter(c.data_type for c in classifications.values())

        avg_confidence = sum(c.confidence for c in classifications.values()) / len(
            classifications
        )
        avg_hed_relevance = sum(
            c.hed_relevance for c in classifications.values()
        ) / len(classifications)

        return {
            "semantic_types": dict(semantic_types),
            "data_types": dict(data_types),
            "average_confidence": avg_confidence,
            "average_hed_relevance": avg_hed_relevance,
            "high_confidence_columns": [
                name
                for name, c in classifications.items()
                if c.confidence >= self.classification_threshold
            ],
            "hed_relevant_columns": [
                name for name, c in classifications.items() if c.hed_relevance > 0.5
            ],
        }

    async def _cleanup_implementation(self) -> None:
        """Clean up column classification stage resources."""
        # No specific cleanup needed
        pass


# Register the stage
register_stage("column_classification", ColumnClassificationStage)
