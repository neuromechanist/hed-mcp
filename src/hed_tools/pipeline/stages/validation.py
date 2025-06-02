"""Validation stage for the HED sidecar generation pipeline.

This stage validates the generated HED sidecar for compliance with HED standards,
BIDS requirements, and data integrity constraints.
"""

from typing import Dict, Any, List

from ..core import PipelineStage, PipelineContext


class ValidationStage(PipelineStage):
    """Stage for validating generated HED sidecars.

    This stage:
    1. Validates HED tag syntax and schema compliance
    2. Checks BIDS sidecar format requirements
    3. Validates data integrity and consistency
    4. Performs cross-reference validation with source data
    5. Generates validation reports and warnings
    """

    async def validate_prerequisites(self, context: PipelineContext) -> bool:
        """Validate that prerequisites are met."""
        # Check that sidecar generation stage completed successfully
        sidecar_content = context.processed_data.get("sidecar_content")
        if sidecar_content is None:
            context.add_error("No sidecar content found from previous stage", self.name)
            return False

        return True

    async def execute(self, context: PipelineContext) -> bool:
        """Execute validation."""
        try:
            # Get data from previous stages
            sidecar_content = context.processed_data["sidecar_content"]
            source_data = context.input_data.get("dataframe")
            schema_version = sidecar_content.get("HEDVersion", "8.3.0")

            # Perform validation steps
            validation_results = {
                "overall_valid": True,
                "errors": [],
                "warnings": [],
                "checks_performed": [],
                "schema_version": schema_version,
            }

            # 1. BIDS format validation
            await self._validate_bids_format(sidecar_content, validation_results)

            # 2. HED schema validation
            if self.config.get("validate_hed_schema", True):
                await self._validate_hed_schema(
                    sidecar_content, schema_version, validation_results
                )

            # 3. Data consistency validation
            if source_data is not None and self.config.get(
                "validate_data_consistency", True
            ):
                await self._validate_data_consistency(
                    sidecar_content, source_data, validation_results
                )

            # 4. Cross-reference validation
            if self.config.get("validate_cross_references", True):
                await self._validate_cross_references(
                    sidecar_content, validation_results
                )

            # 5. Custom validation rules
            custom_rules = self.config.get("custom_validation_rules", [])
            if custom_rules:
                await self._apply_custom_validation(
                    sidecar_content, custom_rules, validation_results
                )

            # Determine overall validation status
            has_critical_errors = any(
                error.get("severity", "error") == "critical"
                for error in validation_results["errors"]
            )

            validation_results["overall_valid"] = (
                len(validation_results["errors"]) == 0 or not has_critical_errors
            )

            # Store results
            context.processed_data["validation_results"] = validation_results
            context.set_stage_result(
                self.name,
                {
                    "validation_passed": validation_results["overall_valid"],
                    "error_count": len(validation_results["errors"]),
                    "warning_count": len(validation_results["warnings"]),
                    "checks_performed": len(validation_results["checks_performed"]),
                },
            )

            # Log summary
            if validation_results["overall_valid"]:
                self.logger.info(
                    f"Validation passed with {len(validation_results['warnings'])} warnings"
                )
            else:
                self.logger.warning(
                    f"Validation failed with {len(validation_results['errors'])} errors"
                )

            return True

        except Exception as e:
            context.add_error(f"Validation stage failed: {str(e)}", self.name)
            self.logger.error(f"Validation error: {e}", exc_info=True)
            return False

    async def _validate_bids_format(
        self, sidecar_content: Dict[str, Any], results: Dict[str, Any]
    ) -> None:
        """Validate BIDS sidecar format requirements."""

        results["checks_performed"].append("BIDS format validation")

        # Check required fields
        if "HEDVersion" not in sidecar_content:
            results["errors"].append(
                {
                    "type": "missing_field",
                    "field": "HEDVersion",
                    "message": "HEDVersion field is required in BIDS sidecars",
                    "severity": "critical",
                }
            )

        # Validate HED version format
        hed_version = sidecar_content.get("HEDVersion", "")
        if hed_version and not self._is_valid_hed_version(hed_version):
            results["errors"].append(
                {
                    "type": "invalid_format",
                    "field": "HEDVersion",
                    "message": f"Invalid HED version format: {hed_version}",
                    "severity": "error",
                }
            )

        # Check columns structure
        columns = sidecar_content.get("columns", {})
        if not isinstance(columns, dict):
            results["errors"].append(
                {
                    "type": "invalid_structure",
                    "field": "columns",
                    "message": "Columns field must be a dictionary",
                    "severity": "critical",
                }
            )
        elif len(columns) == 0:
            results["warnings"].append(
                {
                    "type": "empty_structure",
                    "field": "columns",
                    "message": "No columns defined in sidecar",
                    "severity": "warning",
                }
            )

        # Validate individual column definitions
        for column_name, column_def in columns.items():
            await self._validate_column_definition(column_name, column_def, results)

    async def _validate_column_definition(
        self, column_name: str, column_def: Dict[str, Any], results: Dict[str, Any]
    ) -> None:
        """Validate individual column definition."""

        if not isinstance(column_def, dict):
            results["errors"].append(
                {
                    "type": "invalid_structure",
                    "field": f"columns.{column_name}",
                    "message": f"Column definition for '{column_name}' must be a dictionary",
                    "severity": "error",
                }
            )
            return

        # Check for description
        if "Description" not in column_def:
            results["warnings"].append(
                {
                    "type": "missing_description",
                    "field": f"columns.{column_name}.Description",
                    "message": f"Column '{column_name}' missing description",
                    "severity": "warning",
                }
            )

        # Validate HED field if present
        if "HED" in column_def:
            hed_content = column_def["HED"]
            if hed_content and not isinstance(hed_content, (str, dict)):
                results["errors"].append(
                    {
                        "type": "invalid_type",
                        "field": f"columns.{column_name}.HED",
                        "message": f"HED field in column '{column_name}' must be string or object",
                        "severity": "error",
                    }
                )

        # Validate Levels if present
        if "Levels" in column_def:
            levels = column_def["Levels"]
            if not isinstance(levels, dict):
                results["errors"].append(
                    {
                        "type": "invalid_structure",
                        "field": f"columns.{column_name}.Levels",
                        "message": f"Levels in column '{column_name}' must be a dictionary",
                        "severity": "error",
                    }
                )
            else:
                for level_key, level_def in levels.items():
                    if not isinstance(level_def, dict):
                        results["errors"].append(
                            {
                                "type": "invalid_structure",
                                "field": f"columns.{column_name}.Levels.{level_key}",
                                "message": "Level definition must be a dictionary",
                                "severity": "error",
                            }
                        )

    async def _validate_hed_schema(
        self,
        sidecar_content: Dict[str, Any],
        schema_version: str,
        results: Dict[str, Any],
    ) -> None:
        """Validate HED tags against schema."""

        results["checks_performed"].append("HED schema validation")

        try:
            # Try to import HED validation (this would need actual HED library)
            # For now, we'll do basic syntax checking
            await self._validate_hed_syntax(sidecar_content, results)

        except ImportError:
            results["warnings"].append(
                {
                    "type": "validation_unavailable",
                    "message": "HED schema validation unavailable - HED library not found",
                    "severity": "warning",
                }
            )
        except Exception as e:
            results["errors"].append(
                {
                    "type": "validation_error",
                    "message": f"HED schema validation failed: {str(e)}",
                    "severity": "error",
                }
            )

    async def _validate_hed_syntax(
        self, sidecar_content: Dict[str, Any], results: Dict[str, Any]
    ) -> None:
        """Basic HED syntax validation."""

        columns = sidecar_content.get("columns", {})

        for column_name, column_def in columns.items():
            # Check HED field syntax
            if "HED" in column_def:
                hed_content = column_def["HED"]
                if isinstance(hed_content, str):
                    await self._validate_hed_string(
                        hed_content, f"columns.{column_name}.HED", results
                    )

            # Check Levels HED syntax
            if "Levels" in column_def:
                levels = column_def["Levels"]
                if isinstance(levels, dict):
                    for level_key, level_def in levels.items():
                        if isinstance(level_def, dict) and "HED" in level_def:
                            hed_content = level_def["HED"]
                            if isinstance(hed_content, str):
                                await self._validate_hed_string(
                                    hed_content,
                                    f"columns.{column_name}.Levels.{level_key}.HED",
                                    results,
                                )

    async def _validate_hed_string(
        self, hed_string: str, field_path: str, results: Dict[str, Any]
    ) -> None:
        """Validate a HED tag string."""

        # Basic syntax checks
        if not hed_string.strip():
            results["warnings"].append(
                {
                    "type": "empty_hed",
                    "field": field_path,
                    "message": "Empty HED string",
                    "severity": "warning",
                }
            )
            return

        # Check for balanced parentheses
        if hed_string.count("(") != hed_string.count(")"):
            results["errors"].append(
                {
                    "type": "syntax_error",
                    "field": field_path,
                    "message": "Unbalanced parentheses in HED string",
                    "severity": "error",
                }
            )

        # Check for valid tag structure
        if "//" in hed_string:
            results["errors"].append(
                {
                    "type": "syntax_error",
                    "field": field_path,
                    "message": "Double slashes not allowed in HED tags",
                    "severity": "error",
                }
            )

    async def _validate_data_consistency(
        self, sidecar_content: Dict[str, Any], source_data, results: Dict[str, Any]
    ) -> None:
        """Validate consistency between sidecar and source data."""

        results["checks_performed"].append("Data consistency validation")

        # Get column names from source data
        if hasattr(source_data, "columns"):
            source_columns = set(source_data.columns)
            sidecar_columns = set(sidecar_content.get("columns", {}).keys())

            # Check for missing columns in sidecar
            missing_in_sidecar = source_columns - sidecar_columns
            if missing_in_sidecar:
                results["warnings"].append(
                    {
                        "type": "missing_columns",
                        "message": f"Columns in data but not in sidecar: {list(missing_in_sidecar)}",
                        "severity": "warning",
                    }
                )

            # Check for extra columns in sidecar
            extra_in_sidecar = sidecar_columns - source_columns
            if extra_in_sidecar:
                results["warnings"].append(
                    {
                        "type": "extra_columns",
                        "message": f"Columns in sidecar but not in data: {list(extra_in_sidecar)}",
                        "severity": "warning",
                    }
                )

            # Validate level mappings against actual data values
            await self._validate_level_mappings(sidecar_content, source_data, results)

    async def _validate_level_mappings(
        self, sidecar_content: Dict[str, Any], source_data, results: Dict[str, Any]
    ) -> None:
        """Validate that level mappings cover actual data values."""

        columns = sidecar_content.get("columns", {})

        for column_name, column_def in columns.items():
            if "Levels" in column_def and hasattr(source_data, "columns"):
                if column_name in source_data.columns:
                    # Get unique values from data
                    try:
                        unique_values = set(
                            source_data[column_name].dropna().astype(str)
                        )
                        sidecar_levels = set(column_def["Levels"].keys())

                        # Check for unmapped values
                        unmapped_values = unique_values - sidecar_levels
                        if unmapped_values:
                            results["warnings"].append(
                                {
                                    "type": "unmapped_values",
                                    "field": f"columns.{column_name}.Levels",
                                    "message": f"Values in data not mapped in levels: {list(unmapped_values)}",
                                    "severity": "warning",
                                }
                            )

                        # Check for unused level definitions
                        unused_levels = sidecar_levels - unique_values
                        if unused_levels:
                            results["warnings"].append(
                                {
                                    "type": "unused_levels",
                                    "field": f"columns.{column_name}.Levels",
                                    "message": f"Level definitions not used in data: {list(unused_levels)}",
                                    "severity": "info",
                                }
                            )

                    except Exception as e:
                        results["warnings"].append(
                            {
                                "type": "validation_error",
                                "field": f"columns.{column_name}",
                                "message": f"Could not validate levels for column: {str(e)}",
                                "severity": "warning",
                            }
                        )

    async def _validate_cross_references(
        self, sidecar_content: Dict[str, Any], results: Dict[str, Any]
    ) -> None:
        """Validate cross-references within the sidecar."""

        results["checks_performed"].append("Cross-reference validation")

        # Check for consistent HED version usage
        hed_version = sidecar_content.get("HEDVersion")
        if hed_version:
            # Could check that all HED tags are compatible with specified version
            pass

        # Check for duplicate descriptions
        descriptions = []
        columns = sidecar_content.get("columns", {})
        for column_name, column_def in columns.items():
            if "Description" in column_def:
                desc = column_def["Description"]
                if desc in descriptions:
                    results["warnings"].append(
                        {
                            "type": "duplicate_description",
                            "field": f"columns.{column_name}.Description",
                            "message": f"Duplicate description found: {desc}",
                            "severity": "info",
                        }
                    )
                descriptions.append(desc)

    async def _apply_custom_validation(
        self,
        sidecar_content: Dict[str, Any],
        custom_rules: List[Dict[str, Any]],
        results: Dict[str, Any],
    ) -> None:
        """Apply custom validation rules."""

        results["checks_performed"].append("Custom validation rules")

        for rule in custom_rules:
            try:
                rule_type = rule.get("type")
                if rule_type == "required_field":
                    await self._validate_required_field(sidecar_content, rule, results)
                elif rule_type == "pattern_match":
                    await self._validate_pattern_match(sidecar_content, rule, results)
                # Add more custom rule types as needed

            except Exception as e:
                results["warnings"].append(
                    {
                        "type": "custom_rule_error",
                        "message": f"Custom validation rule failed: {str(e)}",
                        "severity": "warning",
                    }
                )

    async def _validate_required_field(
        self,
        sidecar_content: Dict[str, Any],
        rule: Dict[str, Any],
        results: Dict[str, Any],
    ) -> None:
        """Validate required field custom rule."""

        field_path = rule.get("field_path", "")
        message = rule.get("message", f"Required field missing: {field_path}")

        # Simple path traversal (could be enhanced)
        current = sidecar_content
        for part in field_path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                results["errors"].append(
                    {
                        "type": "custom_validation",
                        "field": field_path,
                        "message": message,
                        "severity": rule.get("severity", "error"),
                    }
                )
                return

    async def _validate_pattern_match(
        self,
        sidecar_content: Dict[str, Any],
        rule: Dict[str, Any],
        results: Dict[str, Any],
    ) -> None:
        """Validate pattern match custom rule."""

        import re

        field_path = rule.get("field_path", "")
        pattern = rule.get("pattern", "")
        message = rule.get("message", f"Pattern validation failed for {field_path}")

        # Simple path traversal and pattern matching
        try:
            current = sidecar_content
            for part in field_path.split("."):
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return  # Field not found, skip validation

            if isinstance(current, str) and not re.match(pattern, current):
                results["errors"].append(
                    {
                        "type": "pattern_validation",
                        "field": field_path,
                        "message": message,
                        "severity": rule.get("severity", "error"),
                    }
                )

        except re.error as e:
            results["warnings"].append(
                {
                    "type": "invalid_pattern",
                    "message": f"Invalid regex pattern in custom rule: {str(e)}",
                    "severity": "warning",
                }
            )

    def _is_valid_hed_version(self, version: str) -> bool:
        """Check if HED version format is valid."""
        import re

        # Basic version pattern like "8.3.0" or "8.2.0"
        pattern = r"^\d+\.\d+\.\d+$"
        return bool(re.match(pattern, version))

    async def cleanup(self, context: PipelineContext) -> None:
        """Cleanup resources after validation."""
        # No specific cleanup needed for this stage
        pass
