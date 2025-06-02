"""Column classification stage for the HED sidecar generation pipeline.

This stage processes LLM-classified columns, distinguishing between skip_cols
and value_cols, and prepares them for HED annotation.
"""

from typing import Dict, Any, List
import pandas as pd

from ..core import PipelineStage, PipelineContext


class ColumnClassificationStage(PipelineStage):
    """Stage for processing and classifying columns for HED annotation.

    This stage:
    1. Processes skip_cols and value_cols from input parameters
    2. Validates column classifications against loaded data
    3. Provides fallback classification using heuristics
    4. Prepares column metadata for HED mapping stage
    """

    async def validate_prerequisites(self, context: PipelineContext) -> bool:
        """Validate that prerequisites are met."""
        # Check that data ingestion stage completed successfully
        df = context.processed_data.get("dataframe")
        if df is None:
            context.add_error("No dataframe found from data ingestion stage", self.name)
            return False

        if df.empty:
            context.add_error("Dataframe is empty", self.name)
            return False

        return True

    async def execute(self, context: PipelineContext) -> bool:
        """Execute column classification."""
        try:
            df = context.processed_data["dataframe"]

            # Get column classifications from input
            skip_cols = context.input_data.get("skip_cols", ["onset", "duration"])
            value_cols = context.input_data.get("value_cols", [])

            # Validate and process column classifications
            processed_classification = await self._process_column_classification(
                df, skip_cols, value_cols, context
            )

            if processed_classification is None:
                return False

            # Store results in context
            context.processed_data["column_classification"] = processed_classification
            context.set_stage_result(
                self.name,
                {
                    "skip_columns": processed_classification["skip_columns"],
                    "value_columns": processed_classification["value_columns"],
                    "unclassified_columns": processed_classification[
                        "unclassified_columns"
                    ],
                    "classification_method": processed_classification["method"],
                    "total_columns": len(df.columns),
                },
            )

            self.logger.info(
                f"Classified columns: {len(processed_classification['skip_columns'])} skip, "
                f"{len(processed_classification['value_columns'])} value, "
                f"{len(processed_classification['unclassified_columns'])} unclassified"
            )

            return True

        except Exception as e:
            context.add_error(f"Column classification failed: {str(e)}", self.name)
            self.logger.error(f"Column classification error: {e}", exc_info=True)
            return False

    async def _process_column_classification(
        self,
        df: pd.DataFrame,
        skip_cols: List[str],
        value_cols: List[str],
        context: PipelineContext,
    ) -> Dict[str, Any]:
        """Process and validate column classifications."""

        available_columns = set(df.columns)

        # Validate skip_cols
        valid_skip_cols = []
        for col in skip_cols:
            if col in available_columns:
                valid_skip_cols.append(col)
            else:
                context.add_warning(f"Skip column '{col}' not found in data", self.name)

        # Validate value_cols
        valid_value_cols = []
        for col in value_cols:
            if col in available_columns:
                valid_value_cols.append(col)
            else:
                context.add_warning(
                    f"Value column '{col}' not found in data", self.name
                )

        # Check for overlap between skip and value columns
        overlap = set(valid_skip_cols) & set(valid_value_cols)
        if overlap:
            context.add_warning(
                f"Columns appear in both skip and value lists: {list(overlap)}",
                self.name,
            )
            # Remove from value_cols (skip takes precedence)
            valid_value_cols = [col for col in valid_value_cols if col not in overlap]

        # Identify unclassified columns
        classified_columns = set(valid_skip_cols) | set(valid_value_cols)
        unclassified_columns = list(available_columns - classified_columns)

        # Apply heuristic classification for unclassified columns if enabled
        classification_method = "user_provided"
        if self.config.get("fallback_to_heuristics", True) and unclassified_columns:
            heuristic_results = await self._apply_heuristic_classification(
                df, unclassified_columns, context
            )

            valid_skip_cols.extend(heuristic_results["skip"])
            valid_value_cols.extend(heuristic_results["value"])
            unclassified_columns = heuristic_results["unclassified"]

            if heuristic_results["skip"] or heuristic_results["value"]:
                classification_method = "user_provided_with_heuristics"
                self.logger.info(
                    f"Heuristic classification added {len(heuristic_results['skip'])} skip "
                    f"and {len(heuristic_results['value'])} value columns"
                )

        # Generate column analysis for value columns
        column_analysis = await self._analyze_value_columns(
            df, valid_value_cols, context
        )

        return {
            "skip_columns": valid_skip_cols,
            "value_columns": valid_value_cols,
            "unclassified_columns": unclassified_columns,
            "method": classification_method,
            "column_analysis": column_analysis,
        }

    async def _apply_heuristic_classification(
        self,
        df: pd.DataFrame,
        unclassified_columns: List[str],
        context: PipelineContext,
    ) -> Dict[str, List[str]]:
        """Apply heuristic rules to classify unclassified columns."""

        heuristic_skip = []
        heuristic_value = []
        remaining_unclassified = []

        # Common BIDS skip patterns
        skip_patterns = {
            "onset",
            "duration",
            "sample",
            "stim_file",
            "response_time",
            "trial",
            "trial_type",
            "onset_time",
            "offset_time",
        }

        # Skip patterns (case-insensitive partial matches)
        skip_keywords = {
            "onset",
            "duration",
            "time",
            "sample",
            "file",
            "rt",
            "response",
        }

        for col in unclassified_columns:
            col_lower = col.lower()

            # Direct match with known skip patterns
            if col_lower in skip_patterns:
                heuristic_skip.append(col)
                continue

            # Partial keyword match for skip columns
            if any(keyword in col_lower for keyword in skip_keywords):
                heuristic_skip.append(col)
                continue

            # Check column content for classification hints
            classification = await self._analyze_column_for_classification(df, col)

            if classification == "skip":
                heuristic_skip.append(col)
            elif classification == "value":
                heuristic_value.append(col)
            else:
                remaining_unclassified.append(col)

        return {
            "skip": heuristic_skip,
            "value": heuristic_value,
            "unclassified": remaining_unclassified,
        }

    async def _analyze_column_for_classification(
        self, df: pd.DataFrame, column: str
    ) -> str:
        """Analyze column content to suggest classification."""

        col_data = df[column].dropna()
        if len(col_data) == 0:
            return "unknown"

        # Check data types and patterns
        dtype = str(col_data.dtype)
        unique_count = col_data.nunique()
        total_count = len(col_data)

        # High cardinality numerical columns might be timestamps or IDs (skip)
        if dtype in ["float64", "int64"] and unique_count > total_count * 0.8:
            return "skip"

        # Columns with file paths or URLs (skip)
        if col_data.dtype == "object":
            sample_values = col_data.head(10).astype(str)
            if any(
                "/" in str(val) or "\\" in str(val) or "." in str(val)
                for val in sample_values
            ):
                return "skip"

        # Low cardinality categorical columns are good for HED annotation (value)
        if unique_count <= 20 and col_data.dtype == "object":
            return "value"

        # Numerical columns with reasonable cardinality might be conditions/levels
        if dtype in ["int64"] and 2 <= unique_count <= 10:
            return "value"

        return "unknown"

    async def _analyze_value_columns(
        self, df: pd.DataFrame, value_columns: List[str], context: PipelineContext
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze value columns to prepare metadata for HED mapping."""

        analysis = {}
        sample_size = self.config.get("sample_size", 100)

        for col in value_columns:
            col_data = df[col].dropna()

            if len(col_data) == 0:
                context.add_warning(
                    f"Value column '{col}' contains no non-null data", self.name
                )
                continue

            # Basic statistics
            col_analysis = {
                "dtype": str(col_data.dtype),
                "total_count": len(col_data),
                "unique_count": col_data.nunique(),
                "null_count": df[col].isnull().sum(),
                "unique_values": col_data.unique().tolist()[:20],  # Limit to first 20
            }

            # Sample values for LLM processing
            if len(col_data) > sample_size:
                sample_values = col_data.sample(n=sample_size, random_state=42).tolist()
            else:
                sample_values = col_data.tolist()

            col_analysis["sample_values"] = sample_values

            # Add value frequency distribution for categorical columns
            if col_data.dtype == "object" and col_data.nunique() <= 50:
                value_counts = col_data.value_counts()
                col_analysis["value_distribution"] = value_counts.to_dict()

            analysis[col] = col_analysis

        return analysis

    async def cleanup(self, context: PipelineContext) -> None:
        """Cleanup resources after column classification."""
        # No specific cleanup needed for this stage
        pass
