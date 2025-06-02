"""HED MCP Server implementation.

This module provides the main MCP server implementation for HED (Hierarchical Event Descriptors)
tools integration, including column analysis and sidecar generation capabilities.
"""

import anyio
import logging
from typing import Any, Dict, List

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HEDServer:
    """MCP server for HED Tools integration.

    Provides tools for:
    - Analyzing BIDS event file columns for HED annotation
    - Generating HED sidecar templates
    - Validating HED annotations
    """

    def __init__(self):
        """Initialize the HED MCP Server."""
        self.server = Server("hed-tools")
        logger.info("HED MCP Server initialized")

        # Try to initialize HED components but handle graceful degradation
        self.validator = None
        self.schema_manager = None
        self.hed_wrapper = None
        self.column_analyzer = None

        try:
            # Import HED components conditionally
            from ..hed_integration.validation import HEDValidator
            from ..hed_integration.schema import SchemaManagerFacade
            from ..hed_integration.hed_wrapper import HEDWrapper, HEDWrapperConfig
            from ..tools.column_analyzer import ColumnAnalyzer

            # Initialize HED components with proper config
            config = HEDWrapperConfig(hed_schema="8.3.0")
            self.hed_wrapper = HEDWrapper(config=config)
            self.validator = HEDValidator()
            self.schema_manager = SchemaManagerFacade()
            self.column_analyzer = ColumnAnalyzer()

            logger.info("HED components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize HED components: {e}")
            logger.warning("Server will run with limited functionality")

        # Register tools and resources
        self._register_tools()
        self._register_resources()

    def _register_tools(self):
        """Register MCP tools for HED operations."""

        @self.server.list_tools()
        async def list_tools() -> List[types.Tool]:
            """List available HED tools."""
            return [
                types.Tool(
                    name="validate_hed",
                    description="Validate HED annotation strings against a schema",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "annotation": {
                                "type": "string",
                                "description": "HED annotation string to validate",
                            },
                            "schema_version": {
                                "type": "string",
                                "description": "HED schema version (e.g., '8.3.0')",
                                "default": "8.3.0",
                            },
                        },
                        "required": ["annotation"],
                    },
                ),
                types.Tool(
                    name="analyze_event_columns",
                    description="Analyze BIDS event file columns for HED annotation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the BIDS event file (.tsv)",
                            },
                            "max_unique_values": {
                                "type": "integer",
                                "description": "Maximum unique values to show per column",
                                "default": 20,
                            },
                        },
                        "required": ["file_path"],
                    },
                ),
                types.Tool(
                    name="generate_hed_sidecar",
                    description="Generate HED sidecar template from analyzed columns",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the BIDS event file (.tsv)",
                            },
                            "skip_cols": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Columns to skip (e.g., timing columns)",
                                "default": ["onset", "duration"],
                            },
                            "value_cols": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Columns containing categorical values for HED annotation",
                            },
                            "schema_version": {
                                "type": "string",
                                "description": "HED schema version",
                                "default": "8.3.0",
                            },
                        },
                        "required": ["file_path", "value_cols"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(
            name: str, arguments: Dict[str, Any]
        ) -> List[types.TextContent]:
            """Handle tool calls."""
            try:
                if name == "validate_hed":
                    return await self._validate_hed(arguments)
                elif name == "analyze_event_columns":
                    return await self._analyze_event_columns(arguments)
                elif name == "generate_hed_sidecar":
                    return await self._generate_hed_sidecar(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Tool call error for {name}: {e}")
                raise ValueError(f"Tool execution failed: {str(e)}")

    def _register_resources(self):
        """Register MCP resources for HED schemas and metadata."""

        @self.server.list_resources()
        async def list_resources():
            """List available HED resources."""
            return [
                types.Resource(
                    uri="hed://schemas",
                    name="HED Schemas",
                    description="Available HED schema versions and metadata",
                )
            ]

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read HED resource content."""
            if uri == "hed://schemas":
                return await self._get_schema_info()
            else:
                raise ValueError(f"Unknown resource: {uri}")

    async def _validate_hed(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Validate HED annotation."""
        annotation = arguments.get("annotation", "")
        schema_version = arguments.get("schema_version", "8.3.0")

        if not annotation:
            raise ValueError("No annotation provided")

        if not self.validator:
            return [
                types.TextContent(
                    type="text",
                    text="HED validation not available - HED components not initialized",
                )
            ]

        try:
            result = await self.validator.validate_annotation(
                annotation, schema_version=schema_version
            )

            if result.get("is_valid", False):
                response = f"✅ Valid HED annotation!\n\nAnnotation: {annotation}\nSchema: {schema_version}"
            else:
                issues = result.get("issues", [])
                response = (
                    f"❌ Invalid HED annotation\n\nAnnotation: {annotation}\n"
                    f"Schema: {schema_version}\n\nIssues:\n"
                )
                for issue in issues:
                    response += f"- {issue}\n"

            return [types.TextContent(type="text", text=response)]

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return [types.TextContent(type="text", text=f"Validation failed: {str(e)}")]

    async def _analyze_event_columns(
        self, arguments: Dict[str, Any]
    ) -> List[types.TextContent]:
        """Analyze columns in a BIDS events file."""
        if not self.column_analyzer:
            return [
                types.TextContent(
                    type="text",
                    text="Column analysis not available - components not initialized",
                )
            ]

        try:
            file_path = arguments.get("file_path")
            max_unique_values = arguments.get("max_unique_values", 10)

            if not file_path:
                raise ValueError("file_path parameter is required")

            # Use the correct method name
            result = await self.column_analyzer.analyze_events_file(file_path)

            # Format the response
            response = f"""📊 Column Analysis Results

File: {file_path}
Analysis completed successfully.

Columns analyzed: {len(result.get('columns', {}))}
BIDS compliance: {'✅ Compliant' if result.get('bids_compliance', {}).get('is_compliant') else '❌ Issues found'}

Column Details:
{self._format_column_analysis(result.get('columns', {}), max_unique_values)}

Summary:
{result.get('summary', {}).get('description', 'No summary available')}

Recommendations:
{chr(10).join('• ' + rec for rec in result.get('recommendations', []))}
"""

            return [types.TextContent(type="text", text=response)]

        except Exception as e:
            logger.error(f"Column analysis error: {e}")
            return [
                types.TextContent(type="text", text=f"Column analysis failed: {str(e)}")
            ]

    async def _generate_hed_sidecar(
        self, arguments: Dict[str, Any]
    ) -> List[types.TextContent]:
        """Generate HED sidecar template."""
        file_path = arguments.get("file_path", "")
        skip_cols = arguments.get("skip_cols", ["onset", "duration"])
        value_cols = arguments.get("value_cols", [])
        schema_version = arguments.get("schema_version", "8.3.0")

        if not file_path:
            raise ValueError("No file_path provided")

        if not value_cols:
            raise ValueError("No value_cols provided")

        try:
            # Generate basic sidecar template
            sidecar = await self._create_basic_sidecar_template(
                file_path, skip_cols, value_cols, schema_version
            )

            response = "📄 Generated HED Sidecar Template\n\n"
            response += f"File: {file_path}\n"
            response += f"Schema: {schema_version}\n"
            response += f"Skip columns: {skip_cols}\n"
            response += f"Value columns: {value_cols}\n\n"
            response += "Sidecar template:\n```json\n"

            import json

            response += json.dumps(sidecar, indent=2)
            response += "\n```"

            return [types.TextContent(type="text", text=response)]

        except Exception as e:
            logger.error(f"Sidecar generation error: {e}")
            return [
                types.TextContent(
                    type="text", text=f"Sidecar generation failed: {str(e)}"
                )
            ]

    async def _create_basic_sidecar_template(
        self,
        file_path: str,
        skip_cols: List[str],
        value_cols: List[str],
        schema_version: str,
    ) -> Dict[str, Any]:
        """Create a basic HED sidecar template."""
        import pandas as pd

        # Read the file to get actual values
        try:
            df = pd.read_csv(file_path, sep="\t")
        except Exception as e:
            raise ValueError(f"Could not read file {file_path}: {e}")

        sidecar = {}

        # Process each value column
        for col in value_cols:
            if col not in df.columns:
                continue

            unique_values = df[col].dropna().unique()

            sidecar[col] = {
                "LevelsAndValues": {},
                "Description": f"Event type information for {col}",
                "HED": {},
            }

            # Add entry for each unique value
            for value in unique_values:
                value_str = str(value)
                sidecar[col]["LevelsAndValues"][value_str] = (
                    f"Description for {value_str}"
                )
                # Start with basic sensory event tag
                sidecar[col]["HED"][value_str] = "Sensory-event"

        return sidecar

    async def _get_schema_info(self) -> str:
        """Get information about available HED schemas."""
        if not self.schema_manager:
            schema_info = {
                "status": "limited",
                "message": "Schema manager not initialized",
                "available_schemas": ["8.3.0", "8.2.0", "8.1.0"],
                "default_schema": "8.3.0",
            }
        else:
            try:
                # Get schema info from the manager
                schema_info = {
                    "status": "available",
                    "message": "Schema manager initialized",
                    "current_schema": getattr(
                        self.schema_manager, "schema_version", "8.3.0"
                    ),
                    "available_schemas": ["8.3.0", "8.2.0", "8.1.0"],
                    "default_schema": "8.3.0",
                }
            except Exception as e:
                logger.error(f"Schema info error: {e}")
                schema_info = {
                    "status": "error",
                    "message": f"Error getting schema info: {str(e)}",
                    "available_schemas": ["8.3.0"],
                    "default_schema": "8.3.0",
                }

        import json

        return json.dumps(schema_info, indent=2)

    def _format_column_analysis(
        self, columns: Dict[str, Any], max_unique_values: int
    ) -> str:
        """Format column analysis results for display."""
        if not columns:
            return "No columns found"

        output = []
        for col_name, col_info in columns.items():
            col_type = col_info.get("type", "unknown")
            unique_count = col_info.get("unique_count", 0)

            line = f"• {col_name} ({col_type}): {unique_count} unique values"

            # Add sample values if available
            sample_values = col_info.get("sample_values", [])
            if sample_values:
                samples = str(sample_values[:3])[1:-1]  # Remove brackets
                line += f" - samples: {samples}"

            # Add HED suggestion if available
            bids_match = col_info.get("bids_match")
            if bids_match and bids_match.get("is_hed_candidate"):
                priority = bids_match.get("hed_priority", "medium")
                line += f" - HED candidate ({priority} priority)"

            output.append(line)

        return "\n".join(output)

    async def run(self):
        """Run the MCP server with stdio transport."""
        logger.info("Starting HED MCP Server...")

        async with stdio_server() as streams:
            await self.server.run(
                streams[0], streams[1], self.server.create_initialization_options()
            )


def main():
    """Main entry point for the server."""
    server = HEDServer()

    async def arun():
        await server.run()

    anyio.run(arun)


if __name__ == "__main__":
    main()
