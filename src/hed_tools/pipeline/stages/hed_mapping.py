"""HED mapping stage for HED sidecar generation pipeline.

This stage handles:
- Mapping column values to HED annotations
- Integration with HED schema validation
- Automatic tag suggestion and completion
- Custom mapping rule application
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import pandas as pd

from . import (
    PipelineStage,
    StageInput,
    StageOutput,
    create_stage_output,
    register_stage,
)
from ...hed_integration.schema import SchemaHandler
from ...hed_integration.tabular_summary import TabularSummaryWrapper

logger = logging.getLogger(__name__)


@dataclass
class HEDMapping:
    """HED mapping result for a column value."""

    value: str
    hed_tags: List[str]
    confidence: float
    mapping_method: str
    validation_status: str
    suggestions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ColumnMapping:
    """Complete HED mapping for a column."""

    column_name: str
    value_mappings: Dict[str, HEDMapping]
    default_tags: List[str] = field(default_factory=list)
    column_level_tags: List[str] = field(default_factory=list)
    mapping_stats: Dict[str, Any] = field(default_factory=dict)


class HEDMappingStage(PipelineStage):
    """Stage for mapping column values to HED annotations.

    This stage uses the HED schema and existing integrations to create
    comprehensive HED annotations for classified columns.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("hed_mapping", config)

        # Configuration with defaults
        self.hed_version = self.get_config_value("hed_version", "8.3.0")
        self.validation_level = self.get_config_value("validation_level", "strict")
        self.auto_completion = self.get_config_value("auto_completion", True)
        self.min_confidence = self.get_config_value("min_confidence", 0.6)
        self.max_suggestions = self.get_config_value("max_suggestions", 5)

        # Initialize HED components
        self.schema_handler: Optional[SchemaHandler] = None
        self.tabular_summary: Optional[TabularSummaryWrapper] = None

        # Mapping rules and patterns
        self._standard_mappings = self._initialize_standard_mappings()
        self._semantic_patterns = self._initialize_semantic_patterns()

    def _initialize_standard_mappings(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize standard HED mappings for common patterns."""
        return {
            "temporal": {
                "onset": ["Event/Onset"],
                "duration": ["Event/Duration"],
                "offset": ["Event/Offset"],
                "start_time": ["Event/Onset"],
                "end_time": ["Event/Offset"],
                "response_time": ["Event/Duration", "Response-time"],
            },
            "stimulus": {
                "image": ["Sensory-event", "Visual-presentation"],
                "sound": ["Sensory-event", "Auditory-presentation"],
                "text": ["Sensory-event", "Visual-presentation", "Textual"],
                "video": ["Sensory-event", "Visual-presentation", "Dynamic"],
                "word": ["Sensory-event", "Visual-presentation", "Textual", "Word"],
            },
            "response": {
                "correct": ["Response", "Correct-action"],
                "incorrect": ["Response", "Incorrect-action"],
                "error": ["Response", "Incorrect-action"],
                "button_press": ["Response", "Button-press"],
                "key_press": ["Response", "Key-press"],
                "mouse_click": ["Response", "Mouse-button-press"],
            },
            "condition": {
                "baseline": ["Experimental-condition", "Baseline"],
                "control": ["Experimental-condition", "Control"],
                "experimental": ["Experimental-condition", "Experimental"],
                "target": ["Target"],
                "distractor": ["Distractor"],
            },
        }

    def _initialize_semantic_patterns(self) -> Dict[str, List[str]]:
        """Initialize semantic patterns for HED mapping."""
        return {
            "participant_id": ["Participant", "ID"],
            "session_id": ["Session", "ID"],
            "trial_number": ["Trial", "Number"],
            "block_number": ["Block", "Number"],
            "accuracy": ["Performance", "Accuracy"],
            "reaction_time": ["Response-time"],
            "stimulus_type": ["Stimulus-type"],
            "response_type": ["Response-type"],
        }

    async def _initialize_implementation(self) -> None:
        """Initialize the HED mapping stage."""
        self.logger.info(
            f"Initializing HED mapping: version={self.hed_version}, "
            f"validation={self.validation_level}"
        )

        try:
            # Initialize schema handler
            self.schema_handler = SchemaHandler()
            await self.schema_handler.initialize(version=self.hed_version)

            # Initialize tabular summary wrapper
            self.tabular_summary = TabularSummaryWrapper()

            self.logger.info("HED mapping stage initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize HED mapping stage: {e}")
            raise

    async def _execute_implementation(self, stage_input: StageInput) -> StageOutput:
        """Execute HED mapping for classified columns."""
        dataframe = stage_input.get_data()

        if not isinstance(dataframe, pd.DataFrame):
            return create_stage_output(
                data=None,
                metadata=stage_input.metadata,
                context=stage_input.context,
                errors=["Expected pandas DataFrame input for HED mapping"],
            )

        # Get column classifications from previous stage
        column_classifications = stage_input.context.get("column_classifications", {})

        if not column_classifications:
            return create_stage_output(
                data=None,
                metadata=stage_input.metadata,
                context=stage_input.context,
                errors=["No column classifications found from previous stage"],
            )

        try:
            # Create HED mappings for relevant columns
            hed_mappings = {}
            warnings = []
            errors = []

            hed_relevant_columns = [
                name
                for name, classification in column_classifications.items()
                if classification.hed_relevance > 0.5
            ]

            self.logger.info(
                f"Processing {len(hed_relevant_columns)} HED-relevant columns"
            )

            for column_name in hed_relevant_columns:
                try:
                    classification = column_classifications[column_name]
                    column_mapping = await self._create_column_mapping(
                        dataframe, column_name, classification
                    )
                    hed_mappings[column_name] = column_mapping

                    # Collect warnings from mapping
                    for value_mapping in column_mapping.value_mappings.values():
                        warnings.extend(value_mapping.warnings)

                except Exception as e:
                    error_msg = f"Failed to create HED mapping for column '{column_name}': {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)

            # Generate mapping summary
            mapping_summary = self._generate_mapping_summary(hed_mappings)

            # Update metadata
            hed_metadata = {
                "mapped_columns": len(hed_mappings),
                "total_value_mappings": sum(
                    len(mapping.value_mappings) for mapping in hed_mappings.values()
                ),
                "high_confidence_mappings": sum(
                    sum(
                        1
                        for vm in mapping.value_mappings.values()
                        if vm.confidence >= self.min_confidence
                    )
                    for mapping in hed_mappings.values()
                ),
                "hed_version": self.hed_version,
                "validation_level": self.validation_level,
                "mapping_methods": mapping_summary["mapping_methods"],
            }
            stage_input.metadata.update(hed_metadata)

            # Update context for next stages
            stage_input.context.update(
                {
                    "hed_mappings": hed_mappings,
                    "mapping_summary": mapping_summary,
                    "hed_schema_version": self.hed_version,
                }
            )

            self.logger.info(
                f"Created HED mappings for {len(hed_mappings)} columns: "
                f"{hed_metadata['total_value_mappings']} value mappings, "
                f"{hed_metadata['high_confidence_mappings']} high confidence"
            )

            return create_stage_output(
                data=dataframe,  # Pass through the original data
                metadata=stage_input.metadata,
                context=stage_input.context,
                warnings=warnings,
                errors=errors,
            )

        except Exception as e:
            self.logger.error(f"HED mapping failed: {e}")
            return create_stage_output(
                data=None,
                metadata=stage_input.metadata,
                context=stage_input.context,
                errors=[f"HED mapping failed: {str(e)}"],
            )

    async def _create_column_mapping(
        self, df: pd.DataFrame, column_name: str, classification: Any
    ) -> ColumnMapping:
        """Create HED mapping for a specific column."""

        column_data = df[column_name].dropna()
        unique_values = column_data.unique()

        # Create mappings for each unique value
        value_mappings = {}

        for value in unique_values:
            if pd.isna(value):
                continue

            value_str = str(value)
            hed_mapping = await self._map_value_to_hed(
                value_str, column_name, classification
            )
            value_mappings[value_str] = hed_mapping

        # Generate column-level tags based on semantic type
        column_level_tags = self._generate_column_level_tags(
            column_name, classification
        )

        # Calculate mapping statistics
        mapping_stats = self._calculate_mapping_stats(value_mappings)

        return ColumnMapping(
            column_name=column_name,
            value_mappings=value_mappings,
            column_level_tags=column_level_tags,
            mapping_stats=mapping_stats,
        )

    async def _map_value_to_hed(
        self, value: str, column_name: str, classification: Any
    ) -> HEDMapping:
        """Map a specific value to HED annotations."""

        hed_tags = []
        confidence = 0.5
        mapping_method = "unknown"
        warnings = []
        suggestions = []

        # Try standard mappings first
        semantic_type = classification.semantic_type
        if semantic_type in self._standard_mappings:
            value_lower = value.lower()
            for pattern, tags in self._standard_mappings[semantic_type].items():
                if pattern in value_lower:
                    hed_tags.extend(tags)
                    confidence = 0.9
                    mapping_method = "standard_pattern"
                    break

        # Try semantic pattern matching
        if not hed_tags:
            column_lower = column_name.lower()
            for pattern, tags in self._semantic_patterns.items():
                if pattern in column_lower:
                    hed_tags.extend(tags)
                    confidence = 0.7
                    mapping_method = "semantic_pattern"
                    break

        # Try HED schema-based suggestions
        if not hed_tags and self.schema_handler:
            try:
                schema_suggestions = await self._get_schema_suggestions(
                    value, column_name
                )
                if schema_suggestions:
                    hed_tags.extend(schema_suggestions[:2])  # Take top 2 suggestions
                    suggestions.extend(schema_suggestions[2 : self.max_suggestions])
                    confidence = 0.6
                    mapping_method = "schema_suggestion"
            except Exception as e:
                warnings.append(f"Schema suggestion failed: {str(e)}")

        # Generate generic tags if nothing specific found
        if not hed_tags:
            hed_tags = self._generate_generic_tags(value, classification)
            confidence = 0.3
            mapping_method = "generic"

        # Validate the HED tags
        validation_status = await self._validate_hed_tags(hed_tags)

        # Auto-complete tags if enabled
        if self.auto_completion and self.schema_handler:
            try:
                completed_tags = await self._auto_complete_tags(hed_tags)
                if completed_tags != hed_tags:
                    hed_tags = completed_tags
                    confidence = min(confidence + 0.1, 1.0)
            except Exception as e:
                warnings.append(f"Auto-completion failed: {str(e)}")

        return HEDMapping(
            value=value,
            hed_tags=hed_tags,
            confidence=confidence,
            mapping_method=mapping_method,
            validation_status=validation_status,
            suggestions=suggestions,
            warnings=warnings,
        )

    async def _get_schema_suggestions(self, value: str, column_name: str) -> List[str]:
        """Get HED tag suggestions from the schema."""
        suggestions = []

        try:
            # Use schema handler to find related tags
            if self.schema_handler and hasattr(self.schema_handler, "search_tags"):
                # Search for tags related to the value
                value_suggestions = await self.schema_handler.search_tags(value)
                suggestions.extend(value_suggestions[:3])

                # Search for tags related to the column name
                column_suggestions = await self.schema_handler.search_tags(column_name)
                suggestions.extend(column_suggestions[:2])

        except Exception as e:
            self.logger.debug(f"Schema suggestion error: {e}")

        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)

        return unique_suggestions

    def _generate_column_level_tags(
        self, column_name: str, classification: Any
    ) -> List[str]:
        """Generate HED tags that apply to the entire column."""
        tags = []

        semantic_type = classification.semantic_type

        # Add semantic type-based column tags
        semantic_tags = {
            "temporal": ["Temporal"],
            "stimulus": ["Stimulus"],
            "response": ["Response"],
            "categorical": ["Categorical"],
            "identifier": ["Identifier"],
            "physiological": ["Physiological"],
        }

        if semantic_type in semantic_tags:
            tags.extend(semantic_tags[semantic_type])

        # Add column name-based tags
        column_lower = column_name.lower()
        if "onset" in column_lower:
            tags.append("Event/Onset")
        elif "duration" in column_lower:
            tags.append("Event/Duration")
        elif "trial" in column_lower:
            tags.append("Trial")
        elif "block" in column_lower:
            tags.append("Block")

        return tags

    def _generate_generic_tags(self, value: str, classification: Any) -> List[str]:
        """Generate generic HED tags when specific mapping isn't found."""
        tags = []

        semantic_type = classification.semantic_type
        data_type = classification.data_type

        # Add basic semantic type tag
        if semantic_type == "stimulus":
            tags.append("Sensory-event")
        elif semantic_type == "response":
            tags.append("Response")
        elif semantic_type == "temporal":
            tags.append("Temporal")
        elif semantic_type == "categorical":
            tags.append("Categorical-value")

        # Add data type information
        if data_type in ["integer", "float"]:
            tags.append("Numerical-value")
        elif data_type == "text":
            tags.append("Textual")

        # Add value-specific tags
        value_lower = value.lower()
        if value_lower in ["yes", "true", "1"]:
            tags.append("Positive")
        elif value_lower in ["no", "false", "0"]:
            tags.append("Negative")

        return tags if tags else ["Data-value"]

    async def _validate_hed_tags(self, hed_tags: List[str]) -> str:
        """Validate HED tags against the schema."""
        if not hed_tags:
            return "empty"

        try:
            if self.schema_handler and hasattr(self.schema_handler, "validate_tags"):
                validation_result = await self.schema_handler.validate_tags(hed_tags)
                if validation_result.get("valid", False):
                    return "valid"
                else:
                    return "invalid"
            else:
                # Basic validation - check if tags contain valid characters
                for tag in hed_tags:
                    if not isinstance(tag, str) or not tag.strip():
                        return "invalid"
                return "unchecked"

        except Exception as e:
            self.logger.debug(f"Tag validation error: {e}")
            return "error"

    async def _auto_complete_tags(self, hed_tags: List[str]) -> List[str]:
        """Auto-complete partial HED tags."""
        completed_tags = []

        try:
            if self.schema_handler and hasattr(self.schema_handler, "complete_tags"):
                for tag in hed_tags:
                    completed_tag = await self.schema_handler.complete_tag(tag)
                    completed_tags.append(completed_tag or tag)
            else:
                completed_tags = hed_tags

        except Exception as e:
            self.logger.debug(f"Auto-completion error: {e}")
            completed_tags = hed_tags

        return completed_tags

    def _calculate_mapping_stats(
        self, value_mappings: Dict[str, HEDMapping]
    ) -> Dict[str, Any]:
        """Calculate statistics for column mappings."""
        if not value_mappings:
            return {}

        confidences = [mapping.confidence for mapping in value_mappings.values()]
        methods = [mapping.mapping_method for mapping in value_mappings.values()]

        return {
            "total_values": len(value_mappings),
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "high_confidence_count": sum(
                1 for c in confidences if c >= self.min_confidence
            ),
            "mapping_methods": {
                method: methods.count(method) for method in set(methods)
            },
            "validation_statuses": {
                status: sum(
                    1
                    for mapping in value_mappings.values()
                    if mapping.validation_status == status
                )
                for status in set(
                    mapping.validation_status for mapping in value_mappings.values()
                )
            },
        }

    def _generate_mapping_summary(
        self, hed_mappings: Dict[str, ColumnMapping]
    ) -> Dict[str, Any]:
        """Generate summary statistics for all HED mappings."""
        if not hed_mappings:
            return {}

        all_mappings = []
        for column_mapping in hed_mappings.values():
            all_mappings.extend(column_mapping.value_mappings.values())

        if not all_mappings:
            return {}

        mapping_methods = [mapping.mapping_method for mapping in all_mappings]
        confidences = [mapping.confidence for mapping in all_mappings]
        validation_statuses = [mapping.validation_status for mapping in all_mappings]

        return {
            "total_columns": len(hed_mappings),
            "total_value_mappings": len(all_mappings),
            "avg_confidence": sum(confidences) / len(confidences),
            "high_confidence_rate": sum(
                1 for c in confidences if c >= self.min_confidence
            )
            / len(confidences),
            "mapping_methods": {
                method: mapping_methods.count(method) for method in set(mapping_methods)
            },
            "validation_statuses": {
                status: validation_statuses.count(status)
                for status in set(validation_statuses)
            },
            "columns_with_warnings": sum(
                1
                for mapping in hed_mappings.values()
                if any(vm.warnings for vm in mapping.value_mappings.values())
            ),
        }

    async def _cleanup_implementation(self) -> None:
        """Clean up HED mapping stage resources."""
        if self.schema_handler:
            await self.schema_handler.cleanup()

        if self.tabular_summary:
            await self.tabular_summary.cleanup()


# Register the stage
register_stage("hed_mapping", HEDMappingStage)
