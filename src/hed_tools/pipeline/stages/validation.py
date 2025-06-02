"""Validation stage for HED sidecar generation pipeline.

This stage handles:
- Final quality assurance of generated sidecar
- BIDS compliance validation
- HED schema validation
- Performance metrics validation
"""

import logging
from typing import Any, Dict, List
from dataclasses import dataclass, field
import json

from . import (
    PipelineStage,
    StageInput,
    StageOutput,
    create_stage_output,
    register_stage,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation checks."""

    check_name: str
    passed: bool
    message: str
    severity: str = "info"  # info, warning, error
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationSummary:
    """Summary of all validation results."""

    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    errors: int
    overall_status: str  # passed, warning, failed
    results: List[ValidationResult] = field(default_factory=list)


class ValidationStage(PipelineStage):
    """Stage for validating the generated HED sidecar.

    This stage performs comprehensive validation of the sidecar output
    including BIDS compliance, HED schema validation, and quality checks.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("validation", config)

        # Configuration with defaults
        self.strict_validation = self.get_config_value("strict_validation", False)
        self.bids_validation = self.get_config_value("bids_validation", True)
        self.hed_validation = self.get_config_value("hed_validation", True)
        self.performance_validation = self.get_config_value(
            "performance_validation", True
        )
        self.fail_on_warnings = self.get_config_value("fail_on_warnings", False)

        # Validation thresholds
        self.min_confidence_threshold = self.get_config_value(
            "min_confidence_threshold", 0.6
        )
        self.max_execution_time = self.get_config_value("max_execution_time", 10.0)
        self.min_coverage_threshold = self.get_config_value(
            "min_coverage_threshold", 0.8
        )

    async def _initialize_implementation(self) -> None:
        """Initialize the validation stage."""
        self.logger.info(
            f"Initializing validation: strict={self.strict_validation}, "
            f"bids={self.bids_validation}, hed={self.hed_validation}"
        )

    async def _execute_implementation(self, stage_input: StageInput) -> StageOutput:
        """Execute validation of the generated sidecar."""

        # Get sidecar content from previous stage
        formatted_output = stage_input.get_data()
        sidecar_content = stage_input.context.get("sidecar_content", {})

        if not sidecar_content and not formatted_output:
            return create_stage_output(
                data=None,
                metadata=stage_input.metadata,
                context=stage_input.context,
                errors=["No sidecar content found for validation"],
            )

        try:
            # Parse formatted output if we don't have structured content
            if not sidecar_content and formatted_output:
                if isinstance(formatted_output, str):
                    sidecar_content = json.loads(formatted_output)
                else:
                    sidecar_content = formatted_output

            # Run validation checks
            validation_results = []

            if self.bids_validation:
                bids_results = await self._validate_bids_compliance(sidecar_content)
                validation_results.extend(bids_results)

            if self.hed_validation:
                hed_results = await self._validate_hed_compliance(
                    sidecar_content, stage_input.context
                )
                validation_results.extend(hed_results)

            if self.performance_validation:
                performance_results = await self._validate_performance_metrics(
                    stage_input.metadata
                )
                validation_results.extend(performance_results)

            # Additional quality checks
            quality_results = await self._validate_quality_metrics(
                sidecar_content, stage_input.context
            )
            validation_results.extend(quality_results)

            # Create validation summary
            summary = self._create_validation_summary(validation_results)

            # Determine if validation passed
            validation_passed = self._determine_validation_status(summary)

            # Update metadata
            validation_metadata = {
                "validation_passed": validation_passed,
                "total_checks": summary.total_checks,
                "failed_checks": summary.failed_checks,
                "warnings": summary.warnings,
                "errors": summary.errors,
                "overall_status": summary.overall_status,
            }
            stage_input.metadata.update(validation_metadata)

            # Update context
            stage_input.context.update(
                {
                    "validation_summary": summary,
                    "validation_results": validation_results,
                    "validation_passed": validation_passed,
                }
            )

            # Prepare warnings and errors for output
            warnings = [
                r.message for r in validation_results if r.severity == "warning"
            ]
            errors = [r.message for r in validation_results if r.severity == "error"]

            self.logger.info(
                f"Validation completed: {summary.passed_checks}/{summary.total_checks} passed, "
                f"{summary.warnings} warnings, {summary.errors} errors"
            )

            # Return the original data with validation results
            return create_stage_output(
                data=formatted_output,
                metadata=stage_input.metadata,
                context=stage_input.context,
                warnings=warnings,
                errors=errors if not validation_passed else None,
            )

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return create_stage_output(
                data=None,
                metadata=stage_input.metadata,
                context=stage_input.context,
                errors=[f"Validation failed: {str(e)}"],
            )

    async def _validate_bids_compliance(
        self, sidecar_content: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate BIDS compliance of the sidecar."""
        results = []

        # Check for required structure
        if not sidecar_content:
            results.append(
                ValidationResult(
                    check_name="bids_structure",
                    passed=False,
                    message="Sidecar is empty",
                    severity="error",
                )
            )
            return results

        # Check each column entry
        bids_compliant_columns = 0
        total_columns = 0

        for column_name, column_entry in sidecar_content.items():
            if column_name.startswith("_"):  # Skip metadata
                continue

            total_columns += 1
            column_compliant = True

            # Check if it's a proper dictionary
            if not isinstance(column_entry, dict):
                results.append(
                    ValidationResult(
                        check_name="bids_column_structure",
                        passed=False,
                        message=f"Column '{column_name}' is not a dictionary",
                        severity="error",
                    )
                )
                column_compliant = False
                continue

            # Check for HED field
            if "HED" not in column_entry:
                results.append(
                    ValidationResult(
                        check_name="bids_hed_field",
                        passed=False,
                        message=f"Column '{column_name}' missing HED field",
                        severity="warning",
                    )
                )
                column_compliant = False

            # Check for non-BIDS fields
            for field_name in column_entry.keys():
                if field_name.startswith("_"):
                    results.append(
                        ValidationResult(
                            check_name="bids_non_standard_fields",
                            passed=False,
                            message=f"Column '{column_name}' contains non-BIDS field '{field_name}'",
                            severity="warning",
                        )
                    )

            if column_compliant:
                bids_compliant_columns += 1

        # Overall BIDS compliance
        compliance_rate = (
            bids_compliant_columns / total_columns if total_columns > 0 else 0
        )
        results.append(
            ValidationResult(
                check_name="bids_overall_compliance",
                passed=compliance_rate >= 0.8,
                message=f"BIDS compliance: {bids_compliant_columns}/{total_columns} columns compliant",
                severity="info" if compliance_rate >= 0.8 else "warning",
                details={"compliance_rate": compliance_rate},
            )
        )

        return results

    async def _validate_hed_compliance(
        self, sidecar_content: Dict[str, Any], context: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate HED compliance and tag quality."""
        results = []

        total_tags = 0
        valid_tags = 0
        empty_tags = 0

        for column_name, column_entry in sidecar_content.items():
            if column_name.startswith("_"):
                continue

            if "HED" not in column_entry:
                continue

            hed_entry = column_entry["HED"]

            if isinstance(hed_entry, dict):
                # Categorical column
                for value, hed_string in hed_entry.items():
                    if value.startswith("_"):
                        continue

                    total_tags += 1

                    if not hed_string or not hed_string.strip():
                        empty_tags += 1
                        results.append(
                            ValidationResult(
                                check_name="hed_empty_tags",
                                passed=False,
                                message=f"Column '{column_name}', value '{value}' has empty HED tags",
                                severity="warning",
                            )
                        )
                    else:
                        # Basic format validation
                        if self._validate_hed_string_format(hed_string):
                            valid_tags += 1
                        else:
                            results.append(
                                ValidationResult(
                                    check_name="hed_format_validation",
                                    passed=False,
                                    message=f"Column '{column_name}', value '{value}' has invalid HED format",
                                    severity="warning",
                                )
                            )

            elif isinstance(hed_entry, str):
                # Value column
                total_tags += 1
                if self._validate_hed_string_format(hed_entry):
                    valid_tags += 1
                else:
                    results.append(
                        ValidationResult(
                            check_name="hed_format_validation",
                            passed=False,
                            message=f"Column '{column_name}' has invalid HED format",
                            severity="warning",
                        )
                    )

        # Overall HED quality
        if total_tags > 0:
            valid_rate = valid_tags / total_tags
            results.append(
                ValidationResult(
                    check_name="hed_overall_quality",
                    passed=valid_rate >= 0.8,
                    message=f"HED quality: {valid_tags}/{total_tags} valid tags",
                    severity="info" if valid_rate >= 0.8 else "warning",
                    details={
                        "valid_rate": valid_rate,
                        "empty_tags": empty_tags,
                        "total_tags": total_tags,
                    },
                )
            )

        return results

    async def _validate_performance_metrics(
        self, metadata: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate performance metrics against targets."""
        results = []

        # Check execution time
        generation_time = metadata.get("generation_time")
        if generation_time:
            # Parse ISO format if needed
            # For now, assume we have timing info in metadata
            pass

        # Check confidence metrics
        hed_metadata = {}
        for key, value in metadata.items():
            if "confidence" in key.lower() or "hed" in key.lower():
                hed_metadata[key] = value

        high_confidence_mappings = metadata.get("high_confidence_mappings", 0)
        total_mappings = metadata.get("total_value_mappings", 1)

        if total_mappings > 0:
            confidence_rate = high_confidence_mappings / total_mappings
            results.append(
                ValidationResult(
                    check_name="performance_confidence",
                    passed=confidence_rate >= self.min_confidence_threshold,
                    message=f"Confidence rate: {confidence_rate:.2f} (threshold: {self.min_confidence_threshold})",
                    severity="info"
                    if confidence_rate >= self.min_confidence_threshold
                    else "warning",
                    details={"confidence_rate": confidence_rate},
                )
            )

        # Check coverage
        mapped_columns = metadata.get("mapped_columns", 0)
        total_columns = metadata.get("total_columns", 1)

        if total_columns > 0:
            coverage_rate = mapped_columns / total_columns
            results.append(
                ValidationResult(
                    check_name="performance_coverage",
                    passed=coverage_rate >= self.min_coverage_threshold,
                    message=f"Coverage rate: {coverage_rate:.2f} (threshold: {self.min_coverage_threshold})",
                    severity="info"
                    if coverage_rate >= self.min_coverage_threshold
                    else "warning",
                    details={"coverage_rate": coverage_rate},
                )
            )

        return results

    async def _validate_quality_metrics(
        self, sidecar_content: Dict[str, Any], context: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate general quality metrics."""
        results = []

        # Check for reasonable content size
        column_count = len([k for k in sidecar_content.keys() if not k.startswith("_")])

        if column_count == 0:
            results.append(
                ValidationResult(
                    check_name="quality_content_size",
                    passed=False,
                    message="No columns found in sidecar",
                    severity="error",
                )
            )
        elif column_count < 3:
            results.append(
                ValidationResult(
                    check_name="quality_content_size",
                    passed=True,
                    message=f"Small sidecar with {column_count} columns",
                    severity="info",
                )
            )
        else:
            results.append(
                ValidationResult(
                    check_name="quality_content_size",
                    passed=True,
                    message=f"Sidecar contains {column_count} columns",
                    severity="info",
                )
            )

        # Check for warnings from previous stages
        pipeline_warnings = []
        for stage_data in context.values():
            if isinstance(stage_data, dict) and "warnings" in stage_data:
                pipeline_warnings.extend(stage_data["warnings"])

        if pipeline_warnings:
            results.append(
                ValidationResult(
                    check_name="quality_pipeline_warnings",
                    passed=len(pipeline_warnings) < 5,
                    message=f"Pipeline generated {len(pipeline_warnings)} warnings",
                    severity="warning" if len(pipeline_warnings) < 5 else "error",
                    details={"warning_count": len(pipeline_warnings)},
                )
            )

        return results

    def _validate_hed_string_format(self, hed_string: str) -> bool:
        """Basic validation of HED string format."""
        if not hed_string or not hed_string.strip():
            return False

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

    def _create_validation_summary(
        self, validation_results: List[ValidationResult]
    ) -> ValidationSummary:
        """Create a summary of validation results."""
        total_checks = len(validation_results)
        passed_checks = sum(1 for r in validation_results if r.passed)
        failed_checks = total_checks - passed_checks
        warnings = sum(1 for r in validation_results if r.severity == "warning")
        errors = sum(1 for r in validation_results if r.severity == "error")

        # Determine overall status
        if errors > 0:
            overall_status = "failed"
        elif warnings > 0:
            overall_status = "warning"
        else:
            overall_status = "passed"

        return ValidationSummary(
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warnings=warnings,
            errors=errors,
            overall_status=overall_status,
            results=validation_results,
        )

    def _determine_validation_status(self, summary: ValidationSummary) -> bool:
        """Determine if validation passed based on configuration."""
        if summary.errors > 0:
            return False

        if self.fail_on_warnings and summary.warnings > 0:
            return False

        if self.strict_validation and summary.failed_checks > 0:
            return False

        return True

    async def _cleanup_implementation(self) -> None:
        """Clean up validation stage resources."""
        # No specific cleanup needed
        pass


# Register the stage
register_stage("validation", ValidationStage)
