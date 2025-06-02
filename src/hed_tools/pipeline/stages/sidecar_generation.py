"""Sidecar generation stage for HED sidecar generation pipeline.

This stage handles:
- Converting HED mappings to BIDS-compatible sidecar format
- JSON formatting and validation
- Metadata integration and documentation
- Output file generation and validation
"""

import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from . import (
    PipelineStage,
    StageInput,
    StageOutput,
    create_stage_output,
    register_stage,
)

logger = logging.getLogger(__name__)


@dataclass
class SidecarMetadata:
    """Metadata for the generated sidecar."""

    generated_by: str
    generation_time: str
    hed_version: str
    pipeline_version: str
    total_columns: int
    mapped_columns: int
    confidence_stats: Dict[str, float]
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class SidecarGenerationStage(PipelineStage):
    """Stage for generating BIDS-compatible HED sidecar files.

    This stage takes HED mappings from previous stages and formats them
    into a proper BIDS sidecar JSON structure with appropriate metadata.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("sidecar_generation", config)

        # Configuration with defaults
        self.output_format = self.get_config_value("output_format", "json")
        self.include_metadata = self.get_config_value("include_metadata", True)
        self.include_confidence = self.get_config_value("include_confidence", False)
        self.include_warnings = self.get_config_value("include_warnings", False)
        self.min_confidence = self.get_config_value("min_confidence", 0.6)
        self.pretty_print = self.get_config_value("pretty_print", True)
        self.validate_output = self.get_config_value("validate_output", True)

        # BIDS compliance settings
        self.bids_compliance = self.get_config_value("bids_compliance", True)
        self.include_descriptions = self.get_config_value("include_descriptions", True)

    async def _initialize_implementation(self) -> None:
        """Initialize the sidecar generation stage."""
        self.logger.info(
            f"Initializing sidecar generation: format={self.output_format}, "
            f"bids_compliance={self.bids_compliance}"
        )

    async def _execute_implementation(self, stage_input: StageInput) -> StageOutput:
        """Execute sidecar generation from HED mappings."""

        # Get HED mappings from previous stage
        hed_mappings = stage_input.context.get("hed_mappings", {})

        if not hed_mappings:
            return create_stage_output(
                data=None,
                metadata=stage_input.metadata,
                context=stage_input.context,
                errors=["No HED mappings found from previous stage"],
            )

        try:
            # Generate the sidecar structure
            sidecar_content = await self._generate_sidecar_content(
                hed_mappings, stage_input
            )

            # Add metadata if enabled
            if self.include_metadata:
                metadata = self._create_sidecar_metadata(hed_mappings, stage_input)
                sidecar_content["_metadata"] = asdict(metadata)

            # Validate the output if enabled
            validation_warnings = []
            if self.validate_output:
                validation_warnings = await self._validate_sidecar(sidecar_content)

            # Format output
            if self.output_format == "json":
                formatted_output = self._format_json_output(sidecar_content)
            else:
                formatted_output = sidecar_content

            # Update metadata
            generation_metadata = {
                "sidecar_columns": len(sidecar_content)
                - (1 if self.include_metadata else 0),
                "output_format": self.output_format,
                "bids_compliant": self.bids_compliance,
                "validation_warnings": len(validation_warnings),
                "generation_time": datetime.now().isoformat(),
            }
            stage_input.metadata.update(generation_metadata)

            # Update context for potential next stages
            stage_input.context.update(
                {
                    "sidecar_content": sidecar_content,
                    "formatted_output": formatted_output,
                    "generation_metadata": generation_metadata,
                }
            )

            self.logger.info(
                f"Generated sidecar with {generation_metadata['sidecar_columns']} columns, "
                f"{len(validation_warnings)} validation warnings"
            )

            return create_stage_output(
                data=formatted_output,
                metadata=stage_input.metadata,
                context=stage_input.context,
                warnings=validation_warnings,
            )

        except Exception as e:
            self.logger.error(f"Sidecar generation failed: {e}")
            return create_stage_output(
                data=None,
                metadata=stage_input.metadata,
                context=stage_input.context,
                errors=[f"Sidecar generation failed: {str(e)}"],
            )

    async def _generate_sidecar_content(
        self, hed_mappings: Dict[str, Any], stage_input: StageInput
    ) -> Dict[str, Any]:
        """Generate the main sidecar content from HED mappings."""
        sidecar = {}

        for column_name, column_mapping in hed_mappings.items():
            column_entry = await self._create_column_entry(column_mapping)

            if column_entry:
                sidecar[column_name] = column_entry

        return sidecar

    async def _create_column_entry(
        self, column_mapping: Any
    ) -> Optional[Dict[str, Any]]:
        """Create a sidecar entry for a single column."""

        # Filter value mappings by confidence if required
        filtered_mappings = {}
        for value, mapping in column_mapping.value_mappings.items():
            if mapping.confidence >= self.min_confidence:
                filtered_mappings[value] = mapping

        if not filtered_mappings:
            return None

        # Create the column entry structure
        column_entry = {}

        # Add HED annotations for each value
        hed_dict = {}
        for value, mapping in filtered_mappings.items():
            if mapping.hed_tags:
                # Join HED tags with commas as per BIDS specification
                hed_string = ", ".join(mapping.hed_tags)
                hed_dict[value] = hed_string

        if hed_dict:
            column_entry["HED"] = hed_dict

        # Add column-level HED tags if present
        if column_mapping.column_level_tags:
            column_hed = ", ".join(column_mapping.column_level_tags)
            if "HED" not in column_entry:
                column_entry["HED"] = {}
            # Add column-level HED as a special entry
            column_entry["HED"]["_column_level"] = column_hed

        # Add descriptions if enabled
        if self.include_descriptions:
            descriptions = self._generate_value_descriptions(filtered_mappings)
            if descriptions:
                column_entry["Description"] = descriptions

        # Add confidence information if enabled (non-BIDS)
        if self.include_confidence:
            confidence_dict = {
                value: mapping.confidence
                for value, mapping in filtered_mappings.items()
            }
            column_entry["_confidence"] = confidence_dict

        # Add warnings if enabled (non-BIDS)
        if self.include_warnings:
            warnings_dict = {
                value: mapping.warnings
                for value, mapping in filtered_mappings.items()
                if mapping.warnings
            }
            if warnings_dict:
                column_entry["_warnings"] = warnings_dict

        return column_entry if column_entry else None

    def _generate_value_descriptions(
        self, value_mappings: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """Generate human-readable descriptions for values."""
        descriptions = {}

        for value, mapping in value_mappings.items():
            # Create description from HED tags
            if mapping.hed_tags:
                # Convert HED tags to human-readable description
                description_parts = []
                for tag in mapping.hed_tags:
                    # Convert hierarchical tags to readable format
                    readable_tag = tag.replace("/", " ").replace("-", " ").lower()
                    description_parts.append(readable_tag)

                description = "Represents " + ", ".join(description_parts)
                descriptions[value] = description

        return descriptions if descriptions else None

    def _create_sidecar_metadata(
        self, hed_mappings: Dict[str, Any], stage_input: StageInput
    ) -> SidecarMetadata:
        """Create metadata for the sidecar file."""

        # Calculate confidence statistics
        all_confidences = []
        total_warnings = []

        for column_mapping in hed_mappings.values():
            for value_mapping in column_mapping.value_mappings.values():
                all_confidences.append(value_mapping.confidence)
                total_warnings.extend(value_mapping.warnings)

        confidence_stats = {}
        if all_confidences:
            confidence_stats = {
                "mean": sum(all_confidences) / len(all_confidences),
                "min": min(all_confidences),
                "max": max(all_confidences),
                "above_threshold": sum(
                    1 for c in all_confidences if c >= self.min_confidence
                ),
            }

        # Get pipeline and HED version info
        hed_version = stage_input.context.get("hed_schema_version", "unknown")
        pipeline_version = stage_input.metadata.get("pipeline_version", "unknown")

        return SidecarMetadata(
            generated_by="HED-MCP Sidecar Pipeline",
            generation_time=datetime.now().isoformat(),
            hed_version=hed_version,
            pipeline_version=pipeline_version,
            total_columns=stage_input.metadata.get("total_columns", 0),
            mapped_columns=len(hed_mappings),
            confidence_stats=confidence_stats,
            warnings=list(set(total_warnings)),  # Remove duplicates
        )

    async def _validate_sidecar(self, sidecar_content: Dict[str, Any]) -> List[str]:
        """Validate the generated sidecar for common issues."""
        warnings = []

        # Check for empty sidecar
        if not sidecar_content or (
            len(sidecar_content) == 1 and "_metadata" in sidecar_content
        ):
            warnings.append("Generated sidecar contains no column mappings")
            return warnings

        # Validate each column entry
        for column_name, column_entry in sidecar_content.items():
            if column_name.startswith("_"):  # Skip metadata entries
                continue

            # Check for HED entry
            if "HED" not in column_entry:
                warnings.append(f"Column '{column_name}' has no HED annotations")
                continue

            hed_entry = column_entry["HED"]

            # Check for empty HED entries
            if not hed_entry:
                warnings.append(f"Column '{column_name}' has empty HED entry")
                continue

            # Validate HED strings
            for value, hed_string in hed_entry.items():
                if value.startswith("_"):  # Skip special entries
                    continue

                if not isinstance(hed_string, str) or not hed_string.strip():
                    warnings.append(
                        f"Column '{column_name}', value '{value}' has invalid HED string"
                    )

                # Check for basic HED format issues
                if not self._validate_hed_string_format(hed_string):
                    warnings.append(
                        f"Column '{column_name}', value '{value}' may have HED format issues"
                    )

        # BIDS compliance checks
        if self.bids_compliance:
            bids_warnings = self._validate_bids_compliance(sidecar_content)
            warnings.extend(bids_warnings)

        return warnings

    def _validate_hed_string_format(self, hed_string: str) -> bool:
        """Basic validation of HED string format."""
        if not hed_string or not hed_string.strip():
            return False

        # Check for basic format issues
        stripped = hed_string.strip()

        # Should not start or end with comma
        if stripped.startswith(",") or stripped.endswith(","):
            return False

        # Should not have double commas
        if ",," in stripped:
            return False

        # Tags should not be empty after splitting
        tags = [tag.strip() for tag in stripped.split(",")]
        if any(not tag for tag in tags):
            return False

        return True

    def _validate_bids_compliance(self, sidecar_content: Dict[str, Any]) -> List[str]:
        """Validate BIDS compliance of the sidecar."""
        warnings = []

        # Check for non-BIDS fields
        for column_name, column_entry in sidecar_content.items():
            if column_name.startswith("_"):  # Skip metadata
                continue

            for field_name in column_entry.keys():
                if field_name.startswith("_"):
                    warnings.append(
                        f"Column '{column_name}' contains non-BIDS field '{field_name}'"
                    )

        # Check for required BIDS structure
        for column_name, column_entry in sidecar_content.items():
            if column_name.startswith("_"):
                continue

            if "HED" in column_entry:
                hed_entry = column_entry["HED"]
                if "_column_level" in hed_entry:
                    warnings.append(
                        f"Column '{column_name}' uses non-standard '_column_level' entry"
                    )

        return warnings

    def _format_json_output(self, sidecar_content: Dict[str, Any]) -> str:
        """Format sidecar content as JSON string."""
        try:
            if self.pretty_print:
                return json.dumps(sidecar_content, indent=2, ensure_ascii=False)
            else:
                return json.dumps(sidecar_content, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"JSON formatting failed: {e}")
            raise

    async def save_to_file(self, formatted_output: str, output_path: Path) -> bool:
        """Save the generated sidecar to a file."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(formatted_output)

            self.logger.info(f"Sidecar saved to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save sidecar to {output_path}: {e}")
            return False

    async def _cleanup_implementation(self) -> None:
        """Clean up sidecar generation stage resources."""
        # No specific cleanup needed
        pass


# Register the stage
register_stage("sidecar_generation", SidecarGenerationStage)
