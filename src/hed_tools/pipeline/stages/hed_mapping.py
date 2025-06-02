"""HED mapping stage for the HED sidecar generation pipeline.

This stage generates HED annotations using TabularSummary integration and creates
mappings from column values to HED tags.
"""

from typing import Dict, Any, List
import pandas as pd
import logging

from ..core import PipelineStage, PipelineContext

logger = logging.getLogger(__name__)


class HEDMappingStage(PipelineStage):
    """Stage for generating HED annotations and mappings.

    This stage:
    1. Integrates with existing TabularSummary wrapper
    2. Generates HED tag suggestions for value columns
    3. Creates intelligent mappings based on column content
    4. Prepares structured HED mappings for sidecar generation
    """

    async def validate_prerequisites(self, context: PipelineContext) -> bool:
        """Validate that prerequisites are met."""
        # Check that column classification stage completed successfully
        classification = context.processed_data.get("column_classification")
        if classification is None:
            context.add_error(
                "No column classification found from previous stage", self.name
            )
            return False

        # Check that data ingestion provided necessary data
        dataframe = context.input_data.get("dataframe")
        if dataframe is None:
            context.add_error("No dataframe found from data ingestion stage", self.name)
            return False

        return True

    async def execute(self, context: PipelineContext) -> bool:
        """Execute HED mapping."""
        try:
            # Get data from previous stages
            classification = context.processed_data["column_classification"]
            dataframe = context.input_data["dataframe"]
            schema_version = context.input_data.get("hed_version", "8.3.0")

            # Determine mapping strategy
            mapping_strategy = self.config.get("mapping_strategy", "intelligent")

            # Generate HED mappings based on strategy
            hed_mappings = {}

            if mapping_strategy == "basic":
                hed_mappings = await self._generate_basic_mappings(
                    classification, dataframe, schema_version, context
                )
            elif mapping_strategy == "intelligent":
                hed_mappings = await self._generate_intelligent_mappings(
                    classification, dataframe, schema_version, context
                )
            elif mapping_strategy == "comprehensive":
                hed_mappings = await self._generate_comprehensive_mappings(
                    classification, dataframe, schema_version, context
                )
            else:
                context.add_warning(
                    f"Unknown mapping strategy '{mapping_strategy}', using intelligent",
                    self.name,
                )
                hed_mappings = await self._generate_intelligent_mappings(
                    classification, dataframe, schema_version, context
                )

            # Store results in context
            context.processed_data["hed_mappings"] = hed_mappings
            context.processed_data["schema_version"] = schema_version

            context.set_stage_result(
                self.name,
                {
                    "mapping_strategy": mapping_strategy,
                    "mappings_generated": len(hed_mappings),
                    "value_columns_processed": len(
                        classification.get("value_columns", [])
                    ),
                    "skip_columns_processed": len(
                        classification.get("skip_columns", [])
                    ),
                    "schema_version": schema_version,
                },
            )

            self.logger.info(
                f"Generated HED mappings for {len(hed_mappings)} columns "
                f"using {mapping_strategy} strategy"
            )

            return True

        except Exception as e:
            context.add_error(f"HED mapping failed: {str(e)}", self.name)
            self.logger.error(f"HED mapping error: {e}", exc_info=True)
            return False

    async def _generate_basic_mappings(
        self,
        classification: Dict[str, Any],
        dataframe: pd.DataFrame,
        schema_version: str,
        context: PipelineContext,
    ) -> Dict[str, Dict[str, Any]]:
        """Generate basic HED mappings using simple heuristics."""

        mappings = {}

        # Process skip columns with minimal HED mapping
        skip_columns = classification.get("skip_columns", [])
        for column in skip_columns:
            if column in dataframe.columns:
                mappings[column] = {
                    "description": f"Skip column: {column}",
                    "HED": self._infer_basic_hed_tag(column),
                    "mapping_type": "basic_skip",
                }

        # Process value columns with basic categorical mapping
        value_columns = classification.get("value_columns", [])
        for column in value_columns:
            if column in dataframe.columns:
                column_mapping = await self._create_basic_value_mapping(
                    column, dataframe, context
                )
                mappings[column] = column_mapping

        return mappings

    async def _generate_intelligent_mappings(
        self,
        classification: Dict[str, Any],
        dataframe: pd.DataFrame,
        schema_version: str,
        context: PipelineContext,
    ) -> Dict[str, Dict[str, Any]]:
        """Generate intelligent HED mappings using TabularSummary integration."""

        mappings = {}

        # Try to use existing TabularSummary integration
        try:
            # Import the existing TabularSummary wrapper
            from ...hed_integration.tabular_summary import TabularSummaryWrapper

            # Initialize wrapper with schema version
            tabular_summary = TabularSummaryWrapper()

            # Process value columns with TabularSummary
            value_columns = classification.get("value_columns", [])
            if value_columns:
                value_mappings = await self._use_tabular_summary(
                    tabular_summary, dataframe, value_columns, schema_version, context
                )
                mappings.update(value_mappings)

            # Process skip columns with enhanced heuristics
            skip_columns = classification.get("skip_columns", [])
            for column in skip_columns:
                if column in dataframe.columns:
                    mappings[column] = await self._create_enhanced_skip_mapping(
                        column, dataframe, context
                    )

        except ImportError as e:
            context.add_warning(
                f"TabularSummary integration unavailable: {e}. Using basic mapping.",
                self.name,
            )
            # Fallback to basic mapping
            return await self._generate_basic_mappings(
                classification, dataframe, schema_version, context
            )
        except Exception as e:
            context.add_error(
                f"TabularSummary integration failed: {e}. Using fallback mapping.",
                self.name,
            )
            # Fallback to basic mapping
            return await self._generate_basic_mappings(
                classification, dataframe, schema_version, context
            )

        return mappings

    async def _generate_comprehensive_mappings(
        self,
        classification: Dict[str, Any],
        dataframe: pd.DataFrame,
        schema_version: str,
        context: PipelineContext,
    ) -> Dict[str, Dict[str, Any]]:
        """Generate comprehensive HED mappings with full analysis."""

        # Start with intelligent mappings
        mappings = await self._generate_intelligent_mappings(
            classification, dataframe, schema_version, context
        )

        # Enhance with additional analysis
        for column_name, mapping in mappings.items():
            if column_name in dataframe.columns:
                # Add detailed statistical analysis
                enhanced_mapping = await self._enhance_mapping_with_stats(
                    column_name, dataframe, mapping, context
                )

                # Add cross-column relationship analysis
                enhanced_mapping = await self._add_relationship_analysis(
                    column_name, dataframe, enhanced_mapping, context
                )

                mappings[column_name] = enhanced_mapping

        return mappings

    async def _use_tabular_summary(
        self,
        tabular_summary: Any,
        dataframe: pd.DataFrame,
        value_columns: List[str],
        schema_version: str,
        context: PipelineContext,
    ) -> Dict[str, Dict[str, Any]]:
        """Use TabularSummary to generate HED mappings for value columns."""

        mappings = {}

        # Filter dataframe to only value columns
        value_data = dataframe[value_columns]

        try:
            # Use the existing async wrapper to get TabularSummary results
            summary_result = await tabular_summary.generate_summary_async(
                value_data,
                hed_schema_version=schema_version,
                include_context=True,
                # Add other parameters as needed from context
            )

            # Extract HED mappings from TabularSummary result
            if summary_result and "column_summary" in summary_result:
                column_summaries = summary_result["column_summary"]

                for column in value_columns:
                    if column in column_summaries:
                        column_summary = column_summaries[column]

                        mapping = {
                            "description": column_summary.get(
                                "description", f"Value column: {column}"
                            ),
                            "HED": column_summary.get("hed_tags", ""),
                            "mapping_type": "tabular_summary",
                            "confidence": column_summary.get("confidence", 0.0),
                        }

                        # Add value-level mappings if available
                        if "value_mappings" in column_summary:
                            mapping["value_mappings"] = column_summary["value_mappings"]

                        mappings[column] = mapping

            # Add context metadata
            context.metadata["tabular_summary_used"] = True
            context.metadata["tabular_summary_version"] = getattr(
                tabular_summary, "version", "unknown"
            )

        except Exception as e:
            context.add_warning(
                f"TabularSummary processing failed for value columns: {e}", self.name
            )
            # Fallback to basic mapping for value columns
            for column in value_columns:
                if column in dataframe.columns:
                    mappings[column] = await self._create_basic_value_mapping(
                        column, dataframe, context
                    )

        return mappings

    async def _create_basic_value_mapping(
        self, column: str, dataframe: pd.DataFrame, context: PipelineContext
    ) -> Dict[str, Any]:
        """Create basic HED mapping for a value column."""

        column_data = dataframe[column]
        unique_values = column_data.dropna().unique()

        mapping = {
            "description": f"Value column: {column}",
            "HED": self._infer_basic_hed_tag(column),
            "mapping_type": "basic_value",
        }

        # Add value mappings for categorical columns
        if len(unique_values) <= 20:  # Threshold for categorical
            value_mappings = {}
            for value in unique_values:
                value_mappings[str(value)] = {
                    "HED": self._infer_value_hed_tag(column, value),
                    "description": f"Value '{value}' in column '{column}'",
                }
            mapping["value_mappings"] = value_mappings

        return mapping

    async def _create_enhanced_skip_mapping(
        self, column: str, dataframe: pd.DataFrame, context: PipelineContext
    ) -> Dict[str, Any]:
        """Create enhanced HED mapping for a skip column."""

        mapping = {
            "description": f"Skip column: {column} (likely metadata or identifier)",
            "HED": self._infer_skip_hed_tag(column),
            "mapping_type": "enhanced_skip",
        }

        # Add column analysis
        column_data = dataframe[column]
        mapping["metadata"] = {
            "unique_count": column_data.nunique(),
            "null_count": column_data.isnull().sum(),
            "dtype": str(column_data.dtype),
        }

        return mapping

    async def _enhance_mapping_with_stats(
        self,
        column_name: str,
        dataframe: pd.DataFrame,
        mapping: Dict[str, Any],
        context: PipelineContext,
    ) -> Dict[str, Any]:
        """Enhance mapping with statistical analysis."""

        column_data = dataframe[column_name]

        # Add comprehensive statistics
        stats = {
            "count": len(column_data),
            "unique_count": column_data.nunique(),
            "null_count": column_data.isnull().sum(),
            "dtype": str(column_data.dtype),
        }

        # Add numeric statistics if applicable
        if pd.api.types.is_numeric_dtype(column_data):
            stats.update(
                {
                    "mean": float(column_data.mean())
                    if not column_data.isnull().all()
                    else None,
                    "std": float(column_data.std())
                    if not column_data.isnull().all()
                    else None,
                    "min": float(column_data.min())
                    if not column_data.isnull().all()
                    else None,
                    "max": float(column_data.max())
                    if not column_data.isnull().all()
                    else None,
                }
            )

        mapping["statistics"] = stats
        return mapping

    async def _add_relationship_analysis(
        self,
        column_name: str,
        dataframe: pd.DataFrame,
        mapping: Dict[str, Any],
        context: PipelineContext,
    ) -> Dict[str, Any]:
        """Add cross-column relationship analysis."""

        # Simple correlation analysis for numeric columns
        if pd.api.types.is_numeric_dtype(dataframe[column_name]):
            numeric_columns = dataframe.select_dtypes(include=[float, int]).columns
            correlations = {}

            for other_col in numeric_columns:
                if other_col != column_name:
                    try:
                        corr = dataframe[column_name].corr(dataframe[other_col])
                        if abs(corr) > 0.5:  # Threshold for significant correlation
                            correlations[other_col] = float(corr)
                    except Exception:
                        continue

            if correlations:
                mapping["relationships"] = {"correlations": correlations}

        return mapping

    def _infer_basic_hed_tag(self, column_name: str) -> str:
        """Infer basic HED tag from column name."""

        col_lower = column_name.lower()

        # Time-related columns
        if any(
            keyword in col_lower
            for keyword in ["time", "timestamp", "duration", "latency"]
        ):
            return "Temporal-value"

        # Response-related columns
        elif any(
            keyword in col_lower for keyword in ["response", "rt", "reaction", "answer"]
        ):
            return "Response-time"

        # Stimulus-related columns
        elif any(
            keyword in col_lower for keyword in ["stimulus", "stim", "cue", "trial"]
        ):
            return "Stimulus"

        # Condition/event columns
        elif any(
            keyword in col_lower for keyword in ["condition", "event", "phase", "block"]
        ):
            return "Condition-variable"

        # Identifier columns
        elif any(
            keyword in col_lower
            for keyword in ["id", "subject", "participant", "session"]
        ):
            return "Label"

        # Default
        else:
            return "Data-value"

    def _infer_skip_hed_tag(self, column_name: str) -> str:
        """Infer HED tag for skip columns (typically metadata)."""

        col_lower = column_name.lower()

        if any(keyword in col_lower for keyword in ["id", "subject", "participant"]):
            return "Label/Participant-identifier"
        elif any(keyword in col_lower for keyword in ["session", "run", "block"]):
            return "Label/Session-identifier"
        elif any(keyword in col_lower for keyword in ["trial", "sequence"]):
            return "Label/Trial-identifier"
        else:
            return "Label"

    def _infer_value_hed_tag(self, column_name: str, value: Any) -> str:
        """Infer HED tag for specific values in a column."""

        col_lower = column_name.lower()
        val_str = str(value).lower()

        # Response values
        if any(keyword in col_lower for keyword in ["response", "answer", "choice"]):
            if val_str in ["correct", "1", "true", "yes"]:
                return "Response/Correct"
            elif val_str in ["incorrect", "0", "false", "no"]:
                return "Response/Incorrect"
            else:
                return f"Response/{value}"

        # Condition values
        elif any(keyword in col_lower for keyword in ["condition", "trial_type"]):
            return f"Condition-variable/{value}"

        # Generic value mapping
        else:
            return f"Data-value/{value}"

    async def cleanup(self, context: PipelineContext) -> None:
        """Cleanup resources after HED mapping."""
        # No specific cleanup needed for this stage
        pass
