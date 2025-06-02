"""Sidecar generation stage for the HED sidecar generation pipeline.

This stage creates the final HED sidecar JSON template from processed HED mappings
and column classifications, following BIDS sidecar format standards.
"""

import json
from typing import Dict, Any, Optional
from datetime import datetime

from ..core import PipelineStage, PipelineContext


class SidecarGenerationStage(PipelineStage):
    """Stage for generating the final HED sidecar JSON template.

    This stage:
    1. Takes HED mappings from the HED mapping stage
    2. Combines them with column classifications
    3. Generates a properly formatted BIDS sidecar JSON
    4. Adds metadata and validation information
    5. Optionally saves the sidecar to file
    """

    async def validate_prerequisites(self, context: PipelineContext) -> bool:
        """Validate that prerequisites are met."""
        # Check that HED mapping stage completed successfully
        hed_mappings = context.processed_data.get("hed_mappings")
        if hed_mappings is None:
            context.add_error("No HED mappings found from previous stage", self.name)
            return False

        classification = context.processed_data.get("column_classification")
        if classification is None:
            context.add_error(
                "No column classification found from previous stage", self.name
            )
            return False

        return True

    async def execute(self, context: PipelineContext) -> bool:
        """Execute sidecar generation."""
        try:
            # Get processed data from previous stages
            hed_mappings = context.processed_data["hed_mappings"]
            classification = context.processed_data["column_classification"]
            schema_version = context.processed_data.get("schema_version", "8.3.0")

            # Generate the sidecar content
            sidecar_content = await self._generate_sidecar_content(
                hed_mappings, classification, schema_version, context
            )

            # Add metadata if configured
            if self.config.get("include_metadata", True):
                sidecar_content = await self._add_metadata(sidecar_content, context)

            # Format according to specified format
            output_format = self.config.get("template_format", "json")
            formatted_content = await self._format_output(
                sidecar_content, output_format, context
            )

            # Store results in context
            context.processed_data["sidecar_content"] = sidecar_content
            context.processed_data["formatted_sidecar"] = formatted_content
            context.set_stage_result(
                self.name,
                {
                    "sidecar_generated": True,
                    "format": output_format,
                    "column_count": len(sidecar_content.get("columns", {})),
                    "has_hed_mappings": len(hed_mappings) > 0,
                    "schema_version": schema_version,
                },
            )

            self.logger.info(
                f"Generated sidecar with {len(sidecar_content.get('columns', {}))} columns "
                f"using HED schema {schema_version}"
            )

            return True

        except Exception as e:
            context.add_error(f"Sidecar generation failed: {str(e)}", self.name)
            self.logger.error(f"Sidecar generation error: {e}", exc_info=True)
            return False

    async def _generate_sidecar_content(
        self,
        hed_mappings: Dict[str, Dict[str, Any]],
        classification: Dict[str, Any],
        schema_version: str,
        context: PipelineContext,
    ) -> Dict[str, Any]:
        """Generate the core sidecar content structure."""

        sidecar = {
            "HEDVersion": schema_version,
            "columns": {},
        }

        # Add task name if available
        task_name = context.input_data.get("task_name")
        if task_name:
            sidecar["TaskName"] = task_name

        # Process all columns (both skip and value columns)
        all_columns = classification.get("skip_columns", []) + classification.get(
            "value_columns", []
        )

        for column in all_columns:
            column_def = await self._generate_column_definition(
                column, hed_mappings.get(column), classification, context
            )
            if column_def:
                sidecar["columns"][column] = column_def

        # Sort columns if configured
        if self.config.get("sort_columns", True):
            sidecar["columns"] = dict(sorted(sidecar["columns"].items()))

        return sidecar

    async def _generate_column_definition(
        self,
        column: str,
        hed_mapping: Optional[Dict[str, Any]],
        classification: Dict[str, Any],
        context: PipelineContext,
    ) -> Optional[Dict[str, Any]]:
        """Generate definition for a single column."""

        column_def = {}

        # Add description
        if hed_mapping and "description" in hed_mapping:
            column_def["Description"] = hed_mapping["description"]
        else:
            # Generate basic description
            column_def["Description"] = f"Data values for column '{column}'"

        # Add HED mapping if available
        if hed_mapping and "HED" in hed_mapping:
            hed_data = hed_mapping["HED"]
            if hed_data:
                column_def["HED"] = hed_data

        # Add levels for value columns if configured
        if (
            column in classification.get("value_columns", [])
            and self.config.get("generate_value_mapping", True)
            and hed_mapping
            and "value_mappings" in hed_mapping
        ):
            levels = {}
            value_mappings = hed_mapping["value_mappings"]

            for value, mapping_info in value_mappings.items():
                if isinstance(mapping_info, dict) and "HED" in mapping_info:
                    levels[value] = {
                        "HED": mapping_info["HED"],
                    }
                    if "description" in mapping_info:
                        levels[value]["Description"] = mapping_info["description"]

            if levels:
                column_def["Levels"] = levels

        # Add column metadata if configured
        if self.config.get("include_column_metadata", True):
            column_metadata = await self._get_column_metadata(
                column, classification, context
            )
            if column_metadata:
                column_def.update(column_metadata)

        return column_def if column_def else None

    async def _get_column_metadata(
        self, column: str, classification: Dict[str, Any], context: PipelineContext
    ) -> Dict[str, Any]:
        """Get metadata for a specific column."""

        metadata = {}
        column_analysis = classification.get("column_analysis", {})

        if column in column_analysis:
            col_data = column_analysis[column]

            # Add data type information
            if "dtype" in col_data:
                metadata["DataType"] = col_data["dtype"]

            # Add unique count for categorical columns
            if col_data.get("unique_count", 0) <= 20:
                metadata["UniqueValues"] = col_data.get("unique_count", 0)

        return metadata

    async def _add_metadata(
        self, sidecar_content: Dict[str, Any], context: PipelineContext
    ) -> Dict[str, Any]:
        """Add generation metadata to the sidecar."""

        metadata = {
            "GeneratedBy": {
                "Name": "HED-MCP",
                "Version": "0.1.0",
                "CodeURL": "https://github.com/hed-standard/hed-mcp",
            },
            "GeneratedAt": datetime.utcnow().isoformat() + "Z",
            "ProcessingSettings": {
                "HEDVersion": sidecar_content.get("HEDVersion", "8.3.0"),
                "MappingStrategy": context.metadata.get(
                    "mapping_strategy", "intelligent"
                ),
                "ValidationEnabled": context.metadata.get("validation_enabled", True),
            },
        }

        # Add processing statistics
        performance_data = context.metadata.get("performance", {})
        if performance_data:
            metadata["ProcessingStatistics"] = {
                "TotalDuration": performance_data.get("total_duration", 0),
                "StageTimings": performance_data.get("stage_timings", {}),
            }

        sidecar_content["_HED_MCP_Metadata"] = metadata
        return sidecar_content

    async def _format_output(
        self,
        sidecar_content: Dict[str, Any],
        output_format: str,
        context: PipelineContext,
    ) -> str:
        """Format the sidecar content according to the specified format."""

        try:
            if output_format.lower() == "json":
                return json.dumps(sidecar_content, indent=2, ensure_ascii=False)
            elif output_format.lower() in ["yaml", "yml"]:
                import yaml

                return yaml.dump(
                    sidecar_content, default_flow_style=False, allow_unicode=True
                )
            else:
                context.add_warning(
                    f"Unsupported output format '{output_format}', using JSON",
                    self.name,
                )
                return json.dumps(sidecar_content, indent=2, ensure_ascii=False)

        except Exception as e:
            context.add_error(f"Failed to format output: {str(e)}", self.name)
            # Fallback to JSON
            return json.dumps(sidecar_content, indent=2, ensure_ascii=False)

    async def cleanup(self, context: PipelineContext) -> None:
        """Cleanup resources after sidecar generation."""
        # No specific cleanup needed for this stage
        pass
