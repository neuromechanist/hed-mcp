"""Validation stage for the HED sidecar generation pipeline.

This stage validates the generated HED sidecar for compliance with HED standards,
BIDS requirements, and data integrity constraints.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Set

try:
    from hed.errors.exceptions import HedFileError, HedExceptions
    from hed.models.hed_string import HedString

    HED_AVAILABLE = True
except ImportError:
    HedFileError = Exception
    HedExceptions = Exception
    HedString = None
    HED_AVAILABLE = False

from ..core import PipelineStage, PipelineContext
from ...hed_integration.schema import SchemaHandler
from ...hed_integration.models import SchemaConfig, ValidationConfig

logger = logging.getLogger(__name__)


class HedColumnValidator:
    """Validates HED annotations for specific BIDS column types."""

    # BIDS column type requirements mapping
    COLUMN_TYPE_REQUIREMENTS = {
        "onset": {
            "required_tags": ["Event"],
            "numeric_validation": True,
            "description": "Event onset time",
        },
        "duration": {
            "numeric_validation": True,
            "positive_only": True,
            "description": "Event duration",
        },
        "trial_type": {
            "categorical_validation": True,
            "hed_required": True,
            "description": "Trial or event type identifier",
        },
        "response_time": {
            "numeric_validation": True,
            "unit_suggestion": "s",
            "positive_only": True,
            "description": "Response time measurement",
        },
        "stim_file": {
            "file_path_validation": True,
            "hed_optional": True,
            "description": "Stimulus file reference",
        },
        "accuracy": {
            "numeric_validation": True,
            "range_validation": (0, 1),
            "description": "Response accuracy",
        },
        "condition": {
            "categorical_validation": True,
            "hed_suggested": True,
            "description": "Experimental condition",
        },
    }

    def __init__(self, schema_handler: SchemaHandler):
        """Initialize with schema handler for tag validation."""
        self.schema_handler = schema_handler
        self.logger = logging.getLogger(f"{__name__}.HedColumnValidator")

    async def validate_column_annotations(
        self,
        column_name: str,
        column_def: Dict[str, Any],
        data_values: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Validate HED annotations for a specific column.

        Args:
            column_name: Name of the column
            column_def: Column definition from sidecar
            data_values: Optional set of actual values from data

        Returns:
            List of validation issues found
        """
        issues = []
        column_lower = column_name.lower()

        # Get column type requirements
        requirements = None
        for col_type, reqs in self.COLUMN_TYPE_REQUIREMENTS.items():
            if col_type in column_lower:
                requirements = reqs
                break

        if requirements:
            issues.extend(
                await self._validate_column_requirements(
                    column_name, column_def, requirements, data_values
                )
            )

        # Validate HED content regardless of column type
        if "HED" in column_def:
            issues.extend(
                await self._validate_hed_content(column_name, column_def["HED"])
            )

        # Validate Levels if present
        if "Levels" in column_def:
            issues.extend(
                await self._validate_levels_structure(
                    column_name, column_def["Levels"], data_values
                )
            )

        return issues

    async def _validate_column_requirements(
        self,
        column_name: str,
        column_def: Dict[str, Any],
        requirements: Dict[str, Any],
        data_values: Optional[Set[str]],
    ) -> List[Dict[str, Any]]:
        """Validate column against specific requirements."""
        issues = []

        # Check if HED is required
        if requirements.get("hed_required", False) and "HED" not in column_def:
            issues.append(
                {
                    "type": "missing_hed",
                    "field": f"columns.{column_name}.HED",
                    "message": f"HED annotation is required for {column_name} columns",
                    "severity": "error",
                    "suggestion": "Add HED annotation to describe the column content",
                }
            )

        # Validate required tags
        if "required_tags" in requirements and "HED" in column_def:
            hed_content = str(column_def["HED"])
            for required_tag in requirements["required_tags"]:
                if required_tag not in hed_content:
                    issues.append(
                        {
                            "type": "missing_required_tag",
                            "field": f"columns.{column_name}.HED",
                            "message": f'Required HED tag "{required_tag}" not found',
                            "severity": "error",
                            "suggestion": f'Include "{required_tag}" tag in HED annotation',
                        }
                    )

        # Suggest HED if not present but recommended
        if requirements.get("hed_suggested", False) and "HED" not in column_def:
            issues.append(
                {
                    "type": "missing_suggested_hed",
                    "field": f"columns.{column_name}.HED",
                    "message": f"HED annotation is recommended for {column_name} columns",
                    "severity": "warning",
                    "suggestion": "Consider adding HED annotation for better data description",
                }
            )

        return issues

    async def _validate_hed_content(
        self, column_name: str, hed_content: Any
    ) -> List[Dict[str, Any]]:
        """Validate HED content using schema handler."""
        issues = []

        if isinstance(hed_content, str):
            # Single HED string validation
            issues.extend(
                await self._validate_hed_string(
                    hed_content, f"columns.{column_name}.HED"
                )
            )
        elif isinstance(hed_content, dict):
            # Categorical HED validation
            for value, hed_string in hed_content.items():
                if isinstance(hed_string, str):
                    issues.extend(
                        await self._validate_hed_string(
                            hed_string, f"columns.{column_name}.HED.{value}"
                        )
                    )

        return issues

    async def _validate_hed_string(
        self, hed_string: str, field_path: str
    ) -> List[Dict[str, Any]]:
        """Validate individual HED string using schema."""
        issues = []

        if not HED_AVAILABLE:
            return issues

        try:
            # Use schema handler to validate
            if self.schema_handler.is_schema_loaded():
                schema = self.schema_handler.get_schema()
                if schema:
                    # Create HedString and validate
                    hed_obj = HedString(hed_string)

                    # Get validation issues from HED library
                    validation_issues = hed_obj.validate(schema)

                    for issue in validation_issues:
                        issues.append(
                            {
                                "type": "hed_validation_error",
                                "field": field_path,
                                "message": str(issue),
                                "severity": "error"
                                if "error" in str(issue).lower()
                                else "warning",
                                "schema_reference": self.schema_handler.get_schema_info().version,
                            }
                        )
        except Exception as e:
            self.logger.warning(f"HED validation failed for {field_path}: {e}")
            issues.append(
                {
                    "type": "validation_exception",
                    "field": field_path,
                    "message": f"HED validation error: {str(e)}",
                    "severity": "warning",
                }
            )

        return issues

    async def _validate_levels_structure(
        self, column_name: str, levels: Dict[str, Any], data_values: Optional[Set[str]]
    ) -> List[Dict[str, Any]]:
        """Validate Levels structure and HED content."""
        issues = []

        for level_key, level_def in levels.items():
            # Validate level definition structure
            if not isinstance(level_def, dict):
                issues.append(
                    {
                        "type": "invalid_level_structure",
                        "field": f"columns.{column_name}.Levels.{level_key}",
                        "message": "Level definition must be a dictionary",
                        "severity": "error",
                    }
                )
                continue

            # Validate HED in levels
            if "HED" in level_def:
                issues.extend(
                    await self._validate_hed_string(
                        str(level_def["HED"]),
                        f"columns.{column_name}.Levels.{level_key}.HED",
                    )
                )

        # Check coverage against actual data values
        if data_values:
            level_keys = set(levels.keys())
            uncovered_values = data_values - level_keys
            if uncovered_values:
                issues.append(
                    {
                        "type": "incomplete_level_coverage",
                        "field": f"columns.{column_name}.Levels",
                        "message": f"Data values not covered by levels: {list(uncovered_values)}",
                        "severity": "warning",
                        "suggestion": "Add level definitions for all data values",
                    }
                )

        return issues


class SchemaCompatibilityDetector:
    """Detects optimal HED schema version for sidecar content."""

    def __init__(self, schema_handler: SchemaHandler):
        """Initialize with schema handler."""
        self.schema_handler = schema_handler
        self.logger = logging.getLogger(f"{__name__}.SchemaCompatibilityDetector")

    async def detect_optimal_version(
        self, sidecar_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect optimal schema version for sidecar.

        Returns:
            Dictionary with compatibility analysis
        """
        analysis = {
            "recommended_version": None,
            "compatibility_scores": {},
            "migration_suggestions": [],
            "incompatible_tags": [],
        }

        # Extract HED tags from sidecar
        hed_tags = self._extract_all_hed_tags(sidecar_content)

        if not hed_tags:
            analysis["recommended_version"] = "8.3.0"  # Default latest
            return analysis

        # Test against available versions
        available_versions = self.schema_handler.get_loaded_schema_versions()
        if not available_versions:
            # Try to load common versions
            common_versions = ["8.3.0", "8.2.0", "8.1.0"]
            for version in common_versions:
                try:
                    await self.schema_handler.load_schema(version=version)
                    available_versions.append(version)
                except Exception:
                    continue

        # Score each version
        for version in available_versions:
            score = await self._calculate_compatibility_score(hed_tags, version)
            analysis["compatibility_scores"][version] = score

        # Determine recommendation
        if analysis["compatibility_scores"]:
            best_version = max(
                analysis["compatibility_scores"].items(), key=lambda x: x[1]
            )[0]
            analysis["recommended_version"] = best_version

        return analysis

    def _extract_all_hed_tags(self, sidecar_content: Dict[str, Any]) -> Set[str]:
        """Extract all HED tags from sidecar content."""
        tags = set()

        columns = sidecar_content.get("columns", {})
        for column_def in columns.values():
            # Direct HED strings
            if "HED" in column_def:
                hed_content = column_def["HED"]
                if isinstance(hed_content, str):
                    tags.update(self._parse_hed_string(hed_content))
                elif isinstance(hed_content, dict):
                    for hed_string in hed_content.values():
                        if isinstance(hed_string, str):
                            tags.update(self._parse_hed_string(hed_string))

            # HED in Levels
            if "Levels" in column_def:
                levels = column_def["Levels"]
                if isinstance(levels, dict):
                    for level_def in levels.values():
                        if isinstance(level_def, dict) and "HED" in level_def:
                            hed_string = level_def["HED"]
                            if isinstance(hed_string, str):
                                tags.update(self._parse_hed_string(hed_string))

        return tags

    def _parse_hed_string(self, hed_string: str) -> Set[str]:
        """Parse HED string to extract individual tags."""
        # Simple tag extraction (could be enhanced)
        import re

        # Remove parentheses and split by commas
        cleaned = re.sub(r"[(),]", " ", hed_string)
        tags = set()

        for potential_tag in cleaned.split():
            potential_tag = potential_tag.strip()
            if potential_tag and "/" in potential_tag:
                # Extract the main tag (before the value placeholder)
                base_tag = potential_tag.split("#")[0].strip()
                if base_tag:
                    tags.add(base_tag)

        return tags

    async def _calculate_compatibility_score(
        self, hed_tags: Set[str], version: str
    ) -> float:
        """Calculate compatibility score for a schema version."""
        if not hed_tags:
            return 1.0

        try:
            schema = self.schema_handler.get_schema_by_version(version)
            if not schema:
                return 0.0

            valid_tags = 0
            for tag in hed_tags:
                if self.schema_handler.validate_tag(tag, version):
                    valid_tags += 1

            return valid_tags / len(hed_tags)

        except Exception as e:
            self.logger.warning(f"Error calculating compatibility for {version}: {e}")
            return 0.0


class ValidationStage(PipelineStage):
    """Stage for validating generated HED sidecars.

    This stage:
    1. Validates HED tag syntax and schema compliance
    2. Checks BIDS sidecar format requirements
    3. Validates data integrity and consistency
    4. Performs cross-reference validation with source data
    5. Generates validation reports and warnings
    """

    def __init__(self, name: str = "Validation", config: Dict[str, Any] = None):
        """Initialize validation stage with enhanced HED integration."""
        super().__init__(name, config)

        # Initialize schema handler
        schema_config = SchemaConfig(
            version=self.config.get("hed_version", "8.3.0"),
            fallback_versions=self.config.get(
                "fallback_versions", ["8.3.0", "8.2.0", "8.1.0"]
            ),
        )
        self.schema_handler = SchemaHandler(schema_config)

        # Initialize specialized validators
        self.column_validator = HedColumnValidator(self.schema_handler)
        self.compatibility_detector = SchemaCompatibilityDetector(self.schema_handler)

        # Validation configuration
        self.validation_config = ValidationConfig(
            check_for_warnings=self.config.get("check_warnings", True),
            check_syntax=self.config.get("check_syntax", True),
            check_required_tags=self.config.get("check_required_tags", True),
            max_errors=self.config.get("max_errors", 100),
            timeout_seconds=self.config.get("timeout_seconds", 60),
        )

    async def validate_prerequisites(self, context: PipelineContext) -> bool:
        """Validate that prerequisites are met."""
        # Check that sidecar generation stage completed successfully
        sidecar_result = context.stage_results.get("SidecarGeneration")
        if not sidecar_result:
            context.add_error("No sidecar generation result found", self.name)
            return False

        sidecar_template = sidecar_result.get("sidecar_template")
        if not sidecar_template:
            context.add_error(
                "No sidecar template found from generation stage", self.name
            )
            return False

        return True

    async def execute(self, context: PipelineContext) -> bool:
        """Execute validation."""
        try:
            start_time = time.time()

            # Get sidecar template from generation stage
            sidecar_result = context.stage_results.get("SidecarGeneration", {})
            sidecar_template = sidecar_result.get("sidecar_template")

            # Parse template if it's JSON string
            if isinstance(sidecar_template, str):
                import json

                try:
                    sidecar_content = json.loads(sidecar_template)
                except json.JSONDecodeError as e:
                    context.add_error(
                        f"Invalid JSON in sidecar template: {e}", self.name
                    )
                    return False
            else:
                sidecar_content = sidecar_template

            # Get source data for consistency validation
            source_data = context.input_data.get("dataframe")
            schema_version = sidecar_content.get(
                "HEDVersion", self.config.get("hed_version", "8.3.0")
            )

            # Initialize validation results
            validation_results = {
                "overall_valid": True,
                "errors": [],
                "warnings": [],
                "info": [],
                "checks_performed": [],
                "schema_version": schema_version,
                "processing_time": 0.0,
                "statistics": {
                    "total_columns": len(sidecar_content.get("columns", {})),
                    "columns_with_hed": 0,
                    "columns_with_levels": 0,
                    "total_hed_strings": 0,
                },
            }

            # Load appropriate HED schema
            await self._ensure_schema_loaded(schema_version, validation_results)

            # 1. Schema compatibility analysis
            if self.config.get("analyze_compatibility", True):
                await self._analyze_schema_compatibility(
                    sidecar_content, validation_results
                )

            # 2. BIDS format validation
            await self._validate_bids_format(sidecar_content, validation_results)

            # 3. Enhanced HED schema validation
            if self.validation_config.check_syntax:
                await self._validate_hed_schema_enhanced(
                    sidecar_content, validation_results
                )

            # 4. Column-specific validation
            if self.config.get("validate_column_types", True):
                await self._validate_column_types(
                    sidecar_content, source_data, validation_results
                )

            # 5. Data consistency validation
            if source_data is not None and self.config.get(
                "validate_data_consistency", True
            ):
                await self._validate_data_consistency(
                    sidecar_content, source_data, validation_results
                )

            # 6. Cross-reference validation
            if self.config.get("validate_cross_references", True):
                await self._validate_cross_references(
                    sidecar_content, validation_results
                )

            # 7. Custom validation rules
            custom_rules = self.config.get("custom_validation_rules", [])
            if custom_rules:
                await self._apply_custom_validation(
                    sidecar_content, custom_rules, validation_results
                )

            # Finalize results
            validation_results["processing_time"] = time.time() - start_time

            # Determine overall validation status
            critical_errors = [
                e
                for e in validation_results["errors"]
                if e.get("severity") == "critical"
            ]
            validation_results["overall_valid"] = len(critical_errors) == 0

            # Store results
            context.set_stage_result(
                self.name,
                {
                    "validation_passed": validation_results["overall_valid"],
                    "error_count": len(validation_results["errors"]),
                    "warning_count": len(validation_results["warnings"]),
                    "info_count": len(validation_results["info"]),
                    "checks_performed": len(validation_results["checks_performed"]),
                    "processing_time": validation_results["processing_time"],
                    "schema_version": validation_results["schema_version"],
                    "validation_details": validation_results,
                },
            )

            # Log summary
            if validation_results["overall_valid"]:
                self.logger.info(
                    f"Validation passed with {len(validation_results['errors'])} errors, "
                    f"{len(validation_results['warnings'])} warnings in "
                    f"{validation_results['processing_time']:.2f}s"
                )
            else:
                self.logger.warning(
                    f"Validation failed with {len(critical_errors)} critical errors, "
                    f"{len(validation_results['errors'])} total errors"
                )

            return True

        except Exception as e:
            context.add_error(f"Validation stage failed: {str(e)}", self.name)
            self.logger.error(f"Validation error: {e}", exc_info=True)
            return False

    async def _ensure_schema_loaded(
        self, schema_version: str, results: Dict[str, Any]
    ) -> None:
        """Ensure appropriate HED schema is loaded."""
        try:
            if not self.schema_handler.is_schema_loaded():
                load_result = await self.schema_handler.load_schema(
                    version=schema_version
                )
                if not load_result.success:
                    # Try fallback versions
                    for (
                        fallback_version
                    ) in self.schema_handler.config.fallback_versions:
                        if fallback_version != schema_version:
                            fallback_result = await self.schema_handler.load_schema(
                                version=fallback_version
                            )
                            if fallback_result.success:
                                results["warnings"].append(
                                    {
                                        "type": "schema_fallback",
                                        "message": f"Using fallback schema version {fallback_version} instead of {schema_version}",
                                        "severity": "warning",
                                    }
                                )
                                results["schema_version"] = fallback_version
                                return

                    # If all fallbacks fail
                    results["errors"].append(
                        {
                            "type": "schema_load_failed",
                            "message": f"Failed to load HED schema version {schema_version}",
                            "severity": "critical",
                        }
                    )

        except Exception as e:
            results["errors"].append(
                {
                    "type": "schema_error",
                    "message": f"Error loading HED schema: {str(e)}",
                    "severity": "error",
                }
            )

    async def _analyze_schema_compatibility(
        self, sidecar_content: Dict[str, Any], results: Dict[str, Any]
    ) -> None:
        """Analyze schema compatibility and suggest optimal version."""
        results["checks_performed"].append("Schema compatibility analysis")

        try:
            compatibility = await self.compatibility_detector.detect_optimal_version(
                sidecar_content
            )

            current_version = results["schema_version"]
            recommended_version = compatibility["recommended_version"]

            if recommended_version and recommended_version != current_version:
                compatibility_score = compatibility["compatibility_scores"].get(
                    current_version, 0.0
                )
                recommended_score = compatibility["compatibility_scores"].get(
                    recommended_version, 0.0
                )

                if recommended_score > compatibility_score:
                    results["info"].append(
                        {
                            "type": "schema_recommendation",
                            "message": f"Consider using HED schema version {recommended_version} for better compatibility",
                            "severity": "info",
                            "compatibility_analysis": compatibility,
                        }
                    )

            # Store compatibility info in metadata
            results["compatibility_analysis"] = compatibility

        except Exception as e:
            self.logger.warning(f"Schema compatibility analysis failed: {e}")

    async def _validate_hed_schema_enhanced(
        self,
        sidecar_content: Dict[str, Any],
        results: Dict[str, Any],
    ) -> None:
        """Enhanced HED schema validation using the schema handler."""
        results["checks_performed"].append("Enhanced HED schema validation")

        if not HED_AVAILABLE:
            results["warnings"].append(
                {
                    "type": "validation_unavailable",
                    "message": "HED library not available - schema validation limited to syntax checking",
                    "severity": "warning",
                }
            )
            await self._validate_hed_syntax(sidecar_content, results)
            return

        if not self.schema_handler.is_schema_loaded():
            results["warnings"].append(
                {
                    "type": "schema_not_loaded",
                    "message": "No HED schema loaded - validation limited to syntax checking",
                    "severity": "warning",
                }
            )
            await self._validate_hed_syntax(sidecar_content, results)
            return

        try:
            # Validate all HED content using schema
            columns = sidecar_content.get("columns", {})

            for column_name, column_def in columns.items():
                column_issues = await self.column_validator.validate_column_annotations(
                    column_name, column_def
                )

                # Categorize issues
                for issue in column_issues:
                    severity = issue.get("severity", "warning")
                    if severity == "error":
                        results["errors"].append(issue)
                    elif severity == "warning":
                        results["warnings"].append(issue)
                    else:
                        results["info"].append(issue)

                # Update statistics
                if "HED" in column_def:
                    results["statistics"]["columns_with_hed"] += 1
                    if isinstance(column_def["HED"], str):
                        results["statistics"]["total_hed_strings"] += 1
                    elif isinstance(column_def["HED"], dict):
                        results["statistics"]["total_hed_strings"] += len(
                            column_def["HED"]
                        )

                if "Levels" in column_def:
                    results["statistics"]["columns_with_levels"] += 1

        except Exception as e:
            results["errors"].append(
                {
                    "type": "validation_error",
                    "message": f"Enhanced HED schema validation failed: {str(e)}",
                    "severity": "error",
                }
            )

    async def _validate_column_types(
        self, sidecar_content: Dict[str, Any], source_data: Any, results: Dict[str, Any]
    ) -> None:
        """Validate column types and their HED annotations."""
        results["checks_performed"].append("Column type validation")

        columns = sidecar_content.get("columns", {})

        # Get actual data values if available
        data_values_map = {}
        if source_data is not None and hasattr(source_data, "columns"):
            for column_name in columns.keys():
                if column_name in source_data.columns:
                    try:
                        unique_values = set(
                            source_data[column_name].dropna().astype(str)
                        )
                        data_values_map[column_name] = unique_values
                    except Exception as e:
                        self.logger.warning(
                            f"Could not extract values for {column_name}: {e}"
                        )

        # Validate each column
        for column_name, column_def in columns.items():
            data_values = data_values_map.get(column_name)

            column_issues = await self.column_validator.validate_column_annotations(
                column_name, column_def, data_values
            )

            # Add issues to results
            for issue in column_issues:
                severity = issue.get("severity", "warning")
                if severity == "error":
                    results["errors"].append(issue)
                elif severity == "warning":
                    results["warnings"].append(issue)
                else:
                    results["info"].append(issue)

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
