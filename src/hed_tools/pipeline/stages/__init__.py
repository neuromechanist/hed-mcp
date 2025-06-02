"""Pipeline stages for HED sidecar generation.

This package contains the individual pipeline stages that handle different
aspects of the sidecar generation process:

- DataIngestionStage: Load and validate input files
- ColumnClassificationStage: Process LLM-classified columns
- HEDMappingStage: Generate HED annotations using TabularSummary
- SidecarGenerationStage: Create final JSON template
- ValidationStage: Validate against HED schema
"""

from .data_ingestion import DataIngestionStage
from .column_classification import ColumnClassificationStage
from .hed_mapping import HEDMappingStage
from .sidecar_generation import SidecarGenerationStage
from .validation import ValidationStage

__all__ = [
    "DataIngestionStage",
    "ColumnClassificationStage",
    "HEDMappingStage",
    "SidecarGenerationStage",
    "ValidationStage",
]
