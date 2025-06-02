"""HED Integration package for hierarchical event descriptor processing.

This package provides comprehensive tools for working with HED (Hierarchical Event Descriptors)
in the context of BIDS datasets and neuroimaging research. It includes schema management,
tabular data processing, validation utilities, and sidecar generation capabilities.

Key Components:
- HEDWrapper: Main interface for HED operations
- SchemaHandler: HED schema loading and management
- TabularSummaryWrapper: Advanced tabular data processing with async support
- HEDValidator: Comprehensive validation for HED annotations
- SidecarGenerator: Automated sidecar template generation
- BatchValidator: Efficient batch processing of multiple files
- BIDSValidator: BIDS-specific validation and compliance checking

Usage Examples:
    Basic HED validation:
    >>> from hed_tools.hed_integration import HEDValidator
    >>> validator = await create_hed_validator()
    >>> result = await validator.validate_string("(Label/condition-1)")

    Sidecar generation:
    >>> from hed_tools.hed_integration import SidecarGenerator
    >>> generator = await create_sidecar_generator()
    >>> template = await generator.generate_sidecar_template("events.tsv")

    Batch validation:
    >>> from hed_tools.hed_integration import BatchValidator
    >>> batch = await create_batch_validator()
    >>> async for result in batch.validate_directory(Path("data")):
    ...     print(f"File: {result['file']}, Valid: {result['validation_result'].is_valid}")
"""

__version__ = "0.1.0"

# Core wrapper and main interface
from .hed_wrapper import HEDWrapper

# Schema management
from .schema import (
    SchemaHandler,
    SchemaManagerFacade,
    HEDSchemaError,
    load_hed_schema,
    validate_hed_tag_simple,
    get_schema_version_info,
)

# Tabular data processing
from .tabular_summary import (
    TabularSummaryWrapper,
    create_tabular_summary_wrapper,
    DataValidator,
    MemoryManager,
    CacheManager,
    ValidationError as TabularValidationError,
    DataFormat,
)

# Validation and sidecar utilities
from .validation import (
    HEDValidator,
    SidecarGenerator,
    BatchValidator,
    BIDSValidator,
    ValidationError,
    # Factory functions
    create_hed_validator,
    create_sidecar_generator,
    create_batch_validator,
    create_bids_validator,
)

# Data models
from .models import (
    # Configuration models
    HEDWrapperConfig,
    SchemaConfig,
    TabularSummaryConfig,
    # Data models
    EventsData,
    ColumnInfo,
    SchemaInfo,
    # Result models
    OperationResult,
    ValidationResult,
    SidecarTemplate,
)

# Convenience imports for common workflows
__all__ = [
    # Main wrapper
    "HEDWrapper",
    # Schema management
    "SchemaHandler",
    "SchemaManagerFacade",
    "HEDSchemaError",
    "load_hed_schema",
    "validate_hed_tag_simple",
    "get_schema_version_info",
    # Tabular processing
    "TabularSummaryWrapper",
    "create_tabular_summary_wrapper",
    "DataValidator",
    "MemoryManager",
    "CacheManager",
    "DataFormat",
    # Validation and sidecar generation
    "HEDValidator",
    "SidecarGenerator",
    "BatchValidator",
    "BIDSValidator",
    "ValidationError",
    "TabularValidationError",
    # Factory functions
    "create_hed_validator",
    "create_sidecar_generator",
    "create_batch_validator",
    "create_bids_validator",
    # Configuration models
    "HEDWrapperConfig",
    "SchemaConfig",
    "TabularSummaryConfig",
    # Data models
    "EventsData",
    "ColumnInfo",
    "SchemaInfo",
    # Result models
    "OperationResult",
    "ValidationResult",
    "SidecarTemplate",
]


# Module-level convenience functions
async def quick_validate_string(hed_string: str, schema_version: str = "8.3.0") -> bool:
    """Quick validation of a HED string.

    Args:
        hed_string: HED annotation string to validate
        schema_version: HED schema version to use (default: 8.3.0)

    Returns:
        True if valid, False otherwise
    """
    validator = await create_hed_validator(schema_version)
    result = await validator.validate_string(hed_string)
    return result.is_valid


async def quick_generate_sidecar(
    events_file: str, output_file: str = None, schema_version: str = "8.3.0"
) -> str:
    """Quick sidecar generation from events file.

    Args:
        events_file: Path to events TSV file
        output_file: Optional output path (defaults to same name with .json)
        schema_version: HED schema version to use (default: 8.3.0)

    Returns:
        Path to generated sidecar file
    """
    generator = await create_sidecar_generator(schema_version)
    template = await generator.generate_sidecar_template(events_file)

    if output_file is None:
        from pathlib import Path

        events_path = Path(events_file)
        output_file = events_path.with_suffix(".json")

    save_result = await generator.save_sidecar(template, output_file)
    if not save_result.success:
        raise RuntimeError(f"Failed to save sidecar: {save_result.error}")

    return str(output_file)


async def quick_validate_bids_dataset(dataset_path: str) -> dict:
    """Quick validation of a BIDS dataset.

    Args:
        dataset_path: Path to BIDS dataset root

    Returns:
        Validation summary dictionary
    """
    from pathlib import Path

    validator = await create_bids_validator()
    result = await validator.validate_bids_dataset(Path(dataset_path))

    return {
        "valid": result["valid"],
        "files_validated": result["files_validated"],
        "error_count": len(result["errors"]),
        "warning_count": len(result["warnings"]),
        "summary": result["summary"],
    }
