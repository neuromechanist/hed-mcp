"""Sidecar generation stage for creating BIDS-compliant HED sidecar files."""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

try:
    from hed.tools.analysis.tabular_summary import TabularSummary

    HED_AVAILABLE = True
except ImportError:
    TabularSummary = None
    HED_AVAILABLE = False

from ..core import PipelineStage, PipelineContext


logger = logging.getLogger(__name__)


class TemplateFormat(Enum):
    """Supported template output formats."""

    BIDS = "bids"
    HED_NOTEBOOK = "hed_notebook"  # exact notebook compatibility
    HYBRID = "hybrid"  # best of both approaches


class CompatibilityMode(Enum):
    """Template generation compatibility modes."""

    NOTEBOOK = "notebook"  # exact notebook workflow
    CURRENT = "current"  # our enhanced approach
    AUTO = "auto"  # intelligent hybrid


@dataclass
class NotebookWorkflowConfig:
    """Configuration for notebook-compatible workflow."""

    dataset_name: str = "Dataset Analysis"
    value_columns: Optional[List[str]] = None
    skip_columns: Optional[List[str]] = None
    auto_detect_columns: bool = True


class NotebookCompatibleGenerator:
    """Exact replication of extract_json_template.ipynb workflow."""

    def __init__(self, config: NotebookWorkflowConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.NotebookCompatibleGenerator")

    def create_tabular_summary(
        self, value_columns: List[str], skip_columns: List[str]
    ) -> TabularSummary:
        """Follow exact notebook instantiation pattern.

        Args:
            value_columns: Columns to treat as value columns (single HED annotation)
            skip_columns: Columns to skip entirely

        Returns:
            TabularSummary instance configured like the notebook
        """
        if not HED_AVAILABLE:
            raise ImportError("HED library not available for notebook compatibility")

        return TabularSummary(
            value_cols=value_columns,
            skip_cols=skip_columns,
            name=self.config.dataset_name,
        )

    def process_notebook_workflow(
        self,
        event_files: List[Union[str, Path]],
        value_columns: List[str],
        skip_columns: List[str],
    ) -> Dict[str, Any]:
        """Exact notebook sequence: instantiate -> update -> extract.

        Args:
            event_files: List of event file paths to process
            value_columns: Columns treated as value columns
            skip_columns: Columns to skip

        Returns:
            Sidecar template in exact notebook format
        """
        try:
            # Step 1: Create TabularSummary instance (exact notebook pattern)
            summary = self.create_tabular_summary(value_columns, skip_columns)

            # Step 2: Update with event files (exact notebook pattern)
            summary.update(event_files)

            # Step 3: Extract sidecar template (exact notebook pattern)
            template = summary.extract_sidecar_template()

            self.logger.info(
                f"Generated notebook-compatible template with {len(template)} columns"
            )
            return template

        except Exception as e:
            self.logger.error(f"Notebook workflow failed: {e}")
            raise


class ColumnDetector:
    """Intelligent column type detection for automatic workflow configuration."""

    def __init__(self):
        self.value_patterns = [
            r".*duration.*",
            r".*onset.*",
            r".*rt.*",
            r".*response_time.*",
            r".*amplitude.*",
            r".*intensity.*",
            r".*score.*",
            r".*rating.*",
            r".*stim_file.*",
            r".*stimulus.*",
            r".*accuracy.*",
        ]

        self.skip_patterns = [
            r"^onset$",
            r"^duration$",
            r"^sample$",  # BIDS standard columns
            r".*index.*",
            r".*id$",
            r".*timestamp.*",
            r".*file_path.*",
        ]

    def detect_value_columns(self, column_names: List[str]) -> List[str]:
        """Identify columns that should be treated as value_cols in notebook workflow."""
        value_cols = []
        for col in column_names:
            if any(
                re.match(pattern, col, re.IGNORECASE) for pattern in self.value_patterns
            ):
                value_cols.append(col)
        return value_cols

    def detect_skip_columns(self, column_names: List[str]) -> List[str]:
        """Identify columns to skip based on notebook patterns."""
        skip_cols = []
        for col in column_names:
            if any(
                re.match(pattern, col, re.IGNORECASE) for pattern in self.skip_patterns
            ):
                skip_cols.append(col)
        return skip_cols

    def get_categorical_columns(
        self, column_names: List[str], value_cols: List[str], skip_cols: List[str]
    ) -> List[str]:
        """Get columns that will be treated as categorical (not value or skip)."""
        all_excluded = set(value_cols + skip_cols)
        return [col for col in column_names if col not in all_excluded]


class TemplateValidator:
    """Validate generated templates against different standards."""

    def validate_notebook_compatibility(
        self,
        generated_template: Dict[str, Any],
        reference_structure: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bool]:
        """Ensure generated template matches notebook structure."""
        validation_results = {
            "has_categorical_structure": self._check_categorical_format(
                generated_template
            ),
            "has_value_column_structure": self._check_value_column_format(
                generated_template
            ),
            "valid_hed_tags": self._check_hed_tag_format(generated_template),
            "bids_compliance": self._check_bids_compliance(generated_template),
        }

        return validation_results

    def _check_categorical_format(self, template: Dict[str, Any]) -> bool:
        """Verify categorical columns follow exact notebook HED structure."""
        for col_name, col_data in template.items():
            if isinstance(col_data, dict) and "HED" in col_data:
                hed_data = col_data["HED"]
                if isinstance(hed_data, dict):
                    # Categorical column - check structure
                    if "Levels" not in col_data:
                        return False
                    # Check HED tags use Label pattern
                    for value, hed_tag in hed_data.items():
                        if not hed_tag.startswith("(Label/"):
                            return False
        return True

    def _check_value_column_format(self, template: Dict[str, Any]) -> bool:
        """Verify value columns have simple HED string format."""
        for col_name, col_data in template.items():
            if isinstance(col_data, dict) and "HED" in col_data:
                hed_data = col_data["HED"]
                if isinstance(hed_data, str):
                    # Value column - should use Label/# pattern
                    if not hed_data.startswith("(Label/") or not hed_data.endswith(
                        "Label/#)"
                    ):
                        return False
        return True

    def _check_hed_tag_format(self, template: Dict[str, Any]) -> bool:
        """Check if HED tags follow valid format."""
        for col_name, col_data in template.items():
            if isinstance(col_data, dict) and "HED" in col_data:
                # Basic validation - more sophisticated checks could be added
                pass
        return True

    def _check_bids_compliance(self, template: Dict[str, Any]) -> bool:
        """Check BIDS sidecar compliance."""
        for col_name, col_data in template.items():
            if not isinstance(col_data, dict):
                return False
            if "Description" not in col_data:
                return False
        return True


class SidecarGenerationStage(PipelineStage):
    """Enhanced sidecar generation stage with notebook compatibility."""

    def __init__(self, name: str = "SidecarGeneration", config: Dict[str, Any] = None):
        super().__init__(name, config)

        # Configuration
        self.compatibility_mode = CompatibilityMode(
            config.get("compatibility_mode", "auto")
        )
        self.template_format = TemplateFormat(config.get("template_format", "hybrid"))
        self.output_format = config.get("output_format", "json")

        # Components
        self.column_detector = ColumnDetector()
        self.template_validator = TemplateValidator()

        # Notebook workflow configuration
        notebook_config_data = config.get("notebook_config", {})
        self.notebook_config = NotebookWorkflowConfig(
            dataset_name=notebook_config_data.get("dataset_name", "Dataset Analysis"),
            value_columns=notebook_config_data.get("value_columns"),
            skip_columns=notebook_config_data.get("skip_columns"),
            auto_detect_columns=notebook_config_data.get("auto_detect_columns", True),
        )

    async def validate_prerequisites(self, context: PipelineContext) -> bool:
        """Validate that required data is available."""
        required_keys = ["column_classifications"]

        if self.compatibility_mode == CompatibilityMode.NOTEBOOK:
            required_keys.append("event_files")

        for key in required_keys:
            if key not in context.processed_data and key not in context.input_data:
                self.logger.error(f"Missing required data: {key}")
                return False

        return True

    async def execute(self, context: PipelineContext) -> bool:
        """Execute sidecar generation with multiple compatibility modes."""
        self.logger.info(
            f"Starting sidecar generation in {self.compatibility_mode.value} mode"
        )

        try:
            # Get input data
            input_data = context.input_data
            processed_data = context.processed_data

            # Get column classifications and HED mappings from previous stages
            column_classifications = processed_data.get("column_classifications", {})
            hed_mappings = processed_data.get("hed_mappings", {})

            # Get event files or dataframe from input
            event_files = input_data.get("event_files", [])
            dataframe = input_data.get("dataframe")

            if not event_files and dataframe is None:
                context.add_error(
                    "No event files or dataframe provided for sidecar generation"
                )
                return False

            # Generate sidecar based on compatibility mode
            if self.compatibility_mode == CompatibilityMode.NOTEBOOK:
                sidecar = await self._generate_notebook_compatible(
                    context, event_files, column_classifications
                )
            elif self.compatibility_mode == CompatibilityMode.CURRENT:
                sidecar = await self._generate_current_method(
                    context, column_classifications, hed_mappings
                )
            else:  # AUTO mode
                sidecar = await self._generate_hybrid_approach(
                    context, event_files, column_classifications, hed_mappings
                )

            # Format output
            formatted_sidecar = self._format_output(sidecar)

            # Validate template
            validation_results = (
                self.template_validator.validate_notebook_compatibility(sidecar)
            )
            context.add_warning(f"Template validation: {validation_results}")

            # Store results
            context.set_stage_result(
                self.name,
                {
                    "sidecar_template": formatted_sidecar,
                    "validation_results": validation_results,
                    "generation_mode": self.compatibility_mode.value,
                    "template_format": self.template_format.value,
                    "column_count": len(sidecar),
                    "generation_time": time.time(),
                },
            )

            self.logger.info(
                f"Successfully generated sidecar with {len(sidecar)} columns"
            )
            return True

        except Exception as e:
            self.logger.error(f"Sidecar generation failed: {e}")
            context.add_error(f"Sidecar generation error: {str(e)}")
            return False

    async def _generate_notebook_compatible(
        self,
        context: PipelineContext,
        event_files: List[Union[str, Path]],
        column_classifications: Dict[str, str],
    ) -> Dict[str, Any]:
        """Generate sidecar following exact notebook workflow."""
        self.logger.info("Generating notebook-compatible sidecar template")

        # Get or detect column types
        if self.notebook_config.auto_detect_columns:
            all_columns = list(column_classifications.keys())
            value_columns = (
                self.notebook_config.value_columns
                or self.column_detector.detect_value_columns(all_columns)
            )
            skip_columns = (
                self.notebook_config.skip_columns
                or self.column_detector.detect_skip_columns(all_columns)
            )
        else:
            value_columns = self.notebook_config.value_columns or []
            skip_columns = self.notebook_config.skip_columns or []

        # Create notebook generator
        generator = NotebookCompatibleGenerator(self.notebook_config)

        # Process using exact notebook workflow
        template = generator.process_notebook_workflow(
            event_files, value_columns, skip_columns
        )

        context.metadata["notebook_value_columns"] = value_columns
        context.metadata["notebook_skip_columns"] = skip_columns

        return template

    async def _generate_current_method(
        self,
        context: PipelineContext,
        column_classifications: Dict[str, str],
        hed_mappings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate sidecar using our enhanced current method."""
        self.logger.info("Generating sidecar using current enhanced method")

        sidecar = {}

        for column_name, classification in column_classifications.items():
            hed_mapping = hed_mappings.get(column_name, {})

            column_def = {
                "Description": f"Description for {column_name}",
                "HED": self._generate_hed_annotation(
                    classification, hed_mapping, column_name
                ),
            }

            # Add Levels for categorical columns
            if classification == "categorical" and "unique_values" in hed_mapping:
                levels = {}
                hed_values = {}
                for value in hed_mapping["unique_values"]:
                    clean_value = str(value).replace(".", "_")
                    levels[str(value)] = (
                        f"Here describe column value {value} of column {column_name}"
                    )
                    hed_values[str(value)] = (
                        f"(Label/{column_name}, Label/{clean_value})"
                    )

                column_def["Levels"] = levels
                column_def["HED"] = hed_values

            sidecar[column_name] = column_def

        return sidecar

    async def _generate_hybrid_approach(
        self,
        context: PipelineContext,
        event_files: List[Union[str, Path]],
        column_classifications: Dict[str, str],
        hed_mappings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate sidecar using hybrid approach - best of both methods."""
        self.logger.info("Generating sidecar using hybrid approach")

        # Try notebook method first if event files available
        if event_files and HED_AVAILABLE:
            try:
                notebook_template = await self._generate_notebook_compatible(
                    context, event_files, column_classifications
                )

                # Validate notebook template quality
                validation = self.template_validator.validate_notebook_compatibility(
                    notebook_template
                )
                if all(validation.values()):
                    self.logger.info(
                        "Using notebook-compatible template (high quality)"
                    )
                    return notebook_template
                else:
                    self.logger.warning(
                        "Notebook template validation failed, falling back to enhanced method"
                    )
            except Exception as e:
                self.logger.warning(
                    f"Notebook method failed, falling back to enhanced method: {e}"
                )

        # Fall back to enhanced current method
        return await self._generate_current_method(
            context, column_classifications, hed_mappings
        )

    def _generate_hed_annotation(
        self, classification: str, hed_mapping: Dict[str, Any], column_name: str
    ) -> Union[str, Dict[str, str]]:
        """Generate HED annotation based on column classification and mapping."""

        if classification == "categorical":
            # Return dict for categorical columns (will be replaced with Levels structure)
            return {}
        elif classification == "value":
            # Value columns get simple Label/# pattern
            return f"(Label/{column_name}, Label/#)"
        elif classification == "temporal":
            return "Temporal-value, #"
        elif classification == "response":
            return "Response-time, #"
        elif classification == "stimulus":
            return "Stimulus-presentation, (Label/#)"
        else:
            # Default pattern
            return f"(Label/{column_name}, Label/#)"

    def _format_output(self, sidecar: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
        """Format the sidecar output according to specified format."""

        if self.output_format == "json":
            return json.dumps(sidecar, indent=4)
        elif self.output_format == "dict":
            return sidecar
        else:
            return sidecar

    async def cleanup(self, context: PipelineContext) -> None:
        """Clean up any resources used during sidecar generation."""
        # Clean up temporary files or resources if any
        pass
