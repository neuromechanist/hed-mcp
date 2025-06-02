"""HED Tools Integration package.

This package provides integration tools for HED (Hierarchical Event Descriptor)
through an MCP (Model Context Protocol) server interface.

The package includes:
- FastMCP server for HED operations
- BIDS events file column analysis
- HED schema management and validation
- Automated sidecar generation using TabularSummary
- File handling utilities for BIDS and HED workflows
"""

__version__ = "0.1.0"

# Core components - import with graceful fallbacks
try:
    from .server.server import HEDServer, create_server
except ImportError:
    HEDServer = None
    create_server = None

try:
    from .hed_integration.hed_wrapper import HEDWrapper, create_hed_wrapper
except ImportError:
    HEDWrapper = None
    create_hed_wrapper = None

try:
    from .tools.column_analyzer import ColumnAnalyzer, create_column_analyzer

    # Create alias for backward compatibility
    BIDSColumnAnalyzer = ColumnAnalyzer
except ImportError:
    BIDSColumnAnalyzer = None
    ColumnAnalyzer = None
    create_column_analyzer = None

try:
    from .utils.file_utils import FileHandler, create_file_handler
except ImportError:
    FileHandler = None
    create_file_handler = None

# Convenience imports and factory functions
__all__ = [
    # Version info
    "__version__",
    # Core classes
    "HEDServer",
    "HEDWrapper",
    "BIDSColumnAnalyzer",
    "ColumnAnalyzer",
    "FileHandler",
    # Factory functions
    "create_server",
    "create_hed_wrapper",
    "create_column_analyzer",
    "create_file_handler",
    # High-level convenience functions
    "create_integration_suite",
    "get_package_info",
    "validate_installation",
]


def create_integration_suite(schema_version: str = "latest") -> dict:
    """Create a complete suite of HED integration tools.

    Args:
        schema_version: HED schema version to use for the wrapper

    Returns:
        Dictionary containing all initialized components
    """
    suite = {}

    if HEDServer is not None:
        suite["server"] = create_server()

    if HEDWrapper is not None:
        suite["hed_wrapper"] = create_hed_wrapper(schema_version)

    if ColumnAnalyzer is not None:
        suite["column_analyzer"] = create_column_analyzer()

    if FileHandler is not None:
        suite["file_handler"] = create_file_handler()

    return suite


def get_package_info() -> dict:
    """Get comprehensive package information and status.

    Returns:
        Dictionary with package metadata and component availability
    """
    return {
        "name": "hed-tools",
        "version": __version__,
        "description": "Integration tools for HED through MCP server interface",
        "components": {
            "server": HEDServer is not None,
            "hed_wrapper": HEDWrapper is not None,
            "column_analyzer": ColumnAnalyzer is not None,
            "file_handler": FileHandler is not None,
        },
        "dependencies": {
            "hed": _check_dependency("hed"),
            "fastmcp": _check_dependency("fastmcp"),
            "pandas": _check_dependency("pandas"),
            "numpy": _check_dependency("numpy"),
            "aiofiles": _check_dependency("aiofiles"),
        },
    }


def validate_installation() -> dict:
    """Validate the installation and component availability.

    Returns:
        Validation report with status and recommendations
    """
    info = get_package_info()
    validation = {"valid": True, "errors": [], "warnings": [], "recommendations": []}

    # Check core components
    missing_components = [
        name for name, available in info["components"].items() if not available
    ]

    if missing_components:
        validation["errors"].extend(
            [
                f"Component '{comp}' not available - check dependencies"
                for comp in missing_components
            ]
        )
        validation["valid"] = False

    # Check dependencies
    missing_deps = [
        name for name, available in info["dependencies"].items() if not available
    ]

    if missing_deps:
        validation["warnings"].extend(
            [
                f"Dependency '{dep}' not available - some features may not work"
                for dep in missing_deps
            ]
        )

    # Generate recommendations
    if "hed" not in info["dependencies"] or not info["dependencies"]["hed"]:
        validation["recommendations"].append(
            "Install 'hedtools' package for full HED functionality: pip install hedtools"
        )

    if "fastmcp" not in info["dependencies"] or not info["dependencies"]["fastmcp"]:
        validation["recommendations"].append(
            "Install 'fastmcp' package for MCP server functionality: pip install fastmcp"
        )

    if not missing_components and not missing_deps:
        validation["recommendations"].append(
            "Installation is complete and ready to use!"
        )

    return validation


def _check_dependency(module_name: str) -> bool:
    """Check if a dependency module is available.

    Args:
        module_name: Name of the module to check

    Returns:
        True if module is available, False otherwise
    """
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


# Module-level convenience for common operations
def quick_analyze_events(file_path, output_path=None):
    """Quick analysis of a BIDS events file.

    Args:
        file_path: Path to BIDS events file
        output_path: Optional path to save analysis results

    Returns:
        Analysis results dictionary
    """
    if ColumnAnalyzer is None:
        raise ImportError("ColumnAnalyzer not available - check dependencies")

    import asyncio
    from pathlib import Path

    async def _analyze():
        analyzer = create_column_analyzer()
        return await analyzer.analyze_events_file(Path(file_path))

    return asyncio.run(_analyze())
