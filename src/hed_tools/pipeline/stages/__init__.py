"""Pipeline stages for HED sidecar generation.

This module contains all pipeline stages for the modular HED sidecar generation
process, each implementing specific processing steps in the overall workflow.
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
