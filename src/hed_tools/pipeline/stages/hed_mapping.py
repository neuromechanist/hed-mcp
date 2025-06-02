"""HED mapping stage for the HED sidecar generation pipeline.

This stage generates HED annotations using TabularSummary integration and creates
mappings from column values to HED tags.
"""

from typing import Dict, Any, List

from ..core import PipelineStage, PipelineContext


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

        value_columns = classification.get("value_columns", [])
        if not value_columns:
            context.add_warning(
                "No value columns to process for HED mapping", self.name
            )
            # This is not an error - we can still generate a sidecar with skip columns only

        return True

    async def execute(self, context: PipelineContext) -> bool:
        """Execute HED mapping generation."""
        try:
            classification = context.processed_data["column_classification"]
            value_columns = classification["value_columns"]
            column_analysis = classification["column_analysis"]

            # Get HED schema version from input
            schema_version = context.input_data.get("schema_version", "8.3.0")

            # Generate HED mappings for value columns
            hed_mappings = await self._generate_hed_mappings(
                value_columns, column_analysis, schema_version, context
            )

            # Store results in context
            context.processed_data["hed_mappings"] = hed_mappings
            context.processed_data["schema_version"] = schema_version
            context.set_stage_result(
                self.name,
                {
                    "mapped_columns": list(hed_mappings.keys()),
                    "total_mappings": sum(
                        len(mappings.get("value_mappings", {}))
                        for mappings in hed_mappings.values()
                    ),
                    "schema_version": schema_version,
                    "mapping_strategy": self.config.get(
                        "mapping_strategy", "intelligent"
                    ),
                },
            )

            self.logger.info(
                f"Generated HED mappings for {len(hed_mappings)} columns using schema {schema_version}"
            )

            return True

        except Exception as e:
            context.add_error(f"HED mapping failed: {str(e)}", self.name)
            self.logger.error(f"HED mapping error: {e}", exc_info=True)
            return False

    async def _generate_hed_mappings(
        self,
        value_columns: List[str],
        column_analysis: Dict[str, Dict[str, Any]],
        schema_version: str,
        context: PipelineContext,
    ) -> Dict[str, Dict[str, Any]]:
        """Generate HED mappings for value columns."""

        mappings = {}
        mapping_strategy = self.config.get("mapping_strategy", "intelligent")
        auto_suggest_tags = self.config.get("auto_suggest_tags", True)

        for column in value_columns:
            if column not in column_analysis:
                context.add_warning(
                    f"No analysis data found for column '{column}'", self.name
                )
                continue

            col_data = column_analysis[column]

            # Generate column-level mapping
            column_mapping = await self._generate_column_mapping(
                column, col_data, schema_version, mapping_strategy, context
            )

            # Generate value-specific mappings
            if auto_suggest_tags:
                value_mappings = await self._generate_value_mappings(
                    column, col_data, schema_version, context
                )
                column_mapping["value_mappings"] = value_mappings

            # Add metadata
            column_mapping["metadata"] = {
                "column_name": column,
                "data_type": col_data["dtype"],
                "unique_count": col_data["unique_count"],
                "total_count": col_data["total_count"],
                "mapping_strategy": mapping_strategy,
                "schema_version": schema_version,
            }

            mappings[column] = column_mapping

        return mappings

    async def _generate_column_mapping(
        self,
        column: str,
        col_data: Dict[str, Any],
        schema_version: str,
        mapping_strategy: str,
        context: PipelineContext,
    ) -> Dict[str, Any]:
        """Generate HED mapping for a specific column."""

        # Placeholder implementation - will be replaced with TabularSummary integration
        # in subsequent subtasks

        column_mapping = {
            "HED": {},
            "description": f"Event markers for column '{column}'",
            "levels": {},
        }

        if mapping_strategy == "basic":
            # Basic mapping with minimal HED tags
            column_mapping["HED"] = self._generate_basic_hed_mapping(column, col_data)
        elif mapping_strategy == "intelligent":
            # Intelligent mapping using heuristics and patterns
            column_mapping["HED"] = await self._generate_intelligent_hed_mapping(
                column, col_data, schema_version, context
            )
        elif mapping_strategy == "comprehensive":
            # Comprehensive mapping with full TabularSummary integration
            column_mapping["HED"] = await self._generate_comprehensive_hed_mapping(
                column, col_data, schema_version, context
            )

        return column_mapping

    def _generate_basic_hed_mapping(
        self, column: str, col_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate basic HED mapping using simple heuristics."""

        # Basic mapping based on column name patterns
        column_lower = column.lower()

        if "condition" in column_lower:
            return {"n/a": "Condition-variable/Experimental-condition"}
        elif "response" in column_lower or "answer" in column_lower:
            return {"n/a": "Response/Motor-response"}
        elif "stimulus" in column_lower or "stim" in column_lower:
            return {"n/a": "Stimulus/Stimulus-presentation"}
        elif "task" in column_lower:
            return {"n/a": "Task/Task-execution"}
        else:
            return {"n/a": "Event/Event-marker"}

    async def _generate_intelligent_hed_mapping(
        self,
        column: str,
        col_data: Dict[str, Any],
        schema_version: str,
        context: PipelineContext,
    ) -> Dict[str, Any]:
        """Generate intelligent HED mapping using advanced heuristics."""

        # Placeholder for intelligent mapping logic
        # This will be enhanced with TabularSummary integration

        hed_mapping = {}
        unique_values = col_data.get("unique_values", [])

        # Analyze column content for intelligent mapping
        for value in unique_values:
            if isinstance(value, str):
                value_str = value.lower()

                # Map common experimental conditions
                if any(
                    keyword in value_str
                    for keyword in ["target", "standard", "deviant"]
                ):
                    hed_mapping[value] = "Stimulus/Stimulus-type/Auditory-stimulus"
                elif any(keyword in value_str for keyword in ["left", "right"]):
                    hed_mapping[value] = "Response/Motor-response/Hand-response"
                elif any(
                    keyword in value_str
                    for keyword in ["correct", "incorrect", "error"]
                ):
                    hed_mapping[value] = "Response/Response-accuracy"
                else:
                    hed_mapping[value] = "Event/Event-marker"
            else:
                # Numerical values
                hed_mapping[str(value)] = "Event/Event-marker"

        return hed_mapping

    async def _generate_comprehensive_hed_mapping(
        self,
        column: str,
        col_data: Dict[str, Any],
        schema_version: str,
        context: PipelineContext,
    ) -> Dict[str, Any]:
        """Generate comprehensive HED mapping using TabularSummary integration."""

        # Placeholder for TabularSummary integration
        # This will be implemented in the next subtask

        # For now, fall back to intelligent mapping
        return await self._generate_intelligent_hed_mapping(
            column, col_data, schema_version, context
        )

    async def _generate_value_mappings(
        self,
        column: str,
        col_data: Dict[str, Any],
        schema_version: str,
        context: PipelineContext,
    ) -> Dict[str, Any]:
        """Generate HED mappings for specific values in the column."""

        value_mappings = {}
        unique_values = col_data.get("unique_values", [])

        # Generate descriptions for each unique value
        for value in unique_values:
            # Placeholder implementation
            value_mappings[str(value)] = {
                "HED": f"Event/Event-marker/{value}",
                "description": f"Event marker for {column} value: {value}",
            }

        return value_mappings

    async def cleanup(self, context: PipelineContext) -> None:
        """Cleanup resources after HED mapping."""
        # No specific cleanup needed for this stage
        pass
