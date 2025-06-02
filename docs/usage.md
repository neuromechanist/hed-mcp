# Usage Guide

This guide covers how to use the HED MCP Server for BIDS events analysis and HED sidecar generation.

## Table of Contents

- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Basic Operations](#basic-operations)
- [MCP Integration](#mcp-integration)
- [API Reference](#api-reference)
- [Common Workflows](#common-workflows)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Basic Package Usage

```python
import hedtools_integration
from pathlib import Path

# Create integration suite
suite = hedtools_integration.create_integration_suite()

# Analyze a BIDS events file
analyzer = suite['column_analyzer']
results = await analyzer.analyze_events_file(Path("events.tsv"))

print(f"Found {len(results['hed_candidates'])} columns suitable for HED annotation")
```

### MCP Server Usage

```python
from hedtools_integration import create_server

# Create and start MCP server
server = create_server()
await server.start("stdio")  # For use with Claude/AI clients
```

### Command Line Usage

```bash
# Test package functionality
uv run python -c "
import hedtools_integration
info = hedtools_integration.get_package_info()
print(f'HED MCP Server v{info[\"version\"]} ready!')
"

# Run the MCP server
uv run python -m hedtools_integration.server
```

## Core Components

The HED MCP Server consists of four main components:

### 1. HEDWrapper - HED Library Integration

Provides a simplified interface to HED Python tools:

```python
from hedtools_integration import create_hed_wrapper

# Create wrapper with specific schema version
wrapper = create_hed_wrapper(schema_version="8.2.0")

# Load schema
await wrapper.load_schema()

# Validate events
validation = await wrapper.validate_events(events_df, sidecar)

# Generate sidecar template
sidecar = await wrapper.generate_sidecar_template(events_df)
```

### 2. BIDSColumnAnalyzer - Column Analysis

Analyzes BIDS events files and classifies columns:

```python
from hedtools_integration import create_column_analyzer

analyzer = create_column_analyzer()

# Analyze events file
analysis = await analyzer.analyze_events_file(Path("events.tsv"))

# Get HED candidates
hed_candidates = analysis['hed_candidates']
for candidate in hed_candidates:
    print(f"Column: {candidate['column']}")
    print(f"Type: {candidate['type']}")
    print(f"Priority: {candidate['priority']}")
```

### 3. FileHandler - File Operations

Handles various file formats used in BIDS and HED workflows:

```python
from hedtools_integration import create_file_handler

handler = create_file_handler()

# Load events file
events_df = await handler.load_events_file(Path("events.tsv"))

# Save JSON sidecar
await handler.save_json_file(sidecar_data, Path("events.json"))

# Validate BIDS structure
validation = await handler.validate_bids_events_structure(Path("events.tsv"))
```

### 4. HEDServer - MCP Server

Provides MCP tools for AI integration:

```python
from hedtools_integration import create_server

server = create_server()

# Get server capabilities
info = server.get_server_info()
print(f"Available tools: {info['capabilities']['tools']}")

# Start server
await server.start("stdio")
```

## Basic Operations

### Analyzing Events Files

**Step 1: Load and validate the events file**

```python
import pandas as pd
from pathlib import Path
from hedtools_integration import create_file_handler

handler = create_file_handler()

# Load events file
events_path = Path("sub-01_task-faces_events.tsv")
events_df = await handler.load_events_file(events_path)

# Validate BIDS compliance
validation = await handler.validate_bids_events_structure(events_path)
if not validation['valid']:
    print("BIDS validation errors:", validation['errors'])
```

**Step 2: Analyze columns for HED suitability**

```python
from hedtools_integration import create_column_analyzer

analyzer = create_column_analyzer()

# Analyze the events file
analysis = await analyzer.analyze_events_file(events_path)

# Review results
print(f"Total columns: {analysis['file_info']['total_columns']}")
print(f"HED candidates: {len(analysis['hed_candidates'])}")

# Show HED candidates
for candidate in analysis['hed_candidates']:
    print(f"\n{candidate['column']} ({candidate['priority']} priority)")
    print(f"  Type: {candidate['type']}")
    print(f"  Unique values: {candidate['unique_values'][:5]}...")
```

**Step 3: Generate HED suggestions**

```python
# Get suggestions for specific columns
for candidate in analysis['hed_candidates']:
    column_name = candidate['column']
    column_data = events_df[column_name]
    
    suggestions = await analyzer.suggest_hed_annotations(column_data, column_name)
    
    print(f"\nHED suggestions for '{column_name}':")
    for suggestion in suggestions[:3]:  # Show first 3 suggestions
        print(f"  {suggestion['value']} -> {suggestion['suggested_hed']}")
```

### Working with HED Schemas

**Loading different schema versions:**

```python
from hedtools_integration import create_hed_wrapper

# Use latest schema
wrapper = create_hed_wrapper()
await wrapper.load_schema()

# Use specific version
wrapper = create_hed_wrapper(schema_version="8.1.0")
await wrapper.load_schema("8.1.0")

# Use custom schema file
wrapper = create_hed_wrapper()
await wrapper.load_schema(custom_path=Path("custom_schema.xml"))

# Check schema info
schema_info = wrapper.get_schema_info()
print(f"Schema loaded: {schema_info['loaded']}")
print(f"Version: {schema_info['version']}")
```

**Schema operations:**

```python
# Get available schemas
available = wrapper.get_available_schemas()
for schema in available:
    print(f"{schema['version']}: {schema['description']}")

# Parse HED strings
hed_string = "Event/Category/Experimental-stimulus, Sensory-event/Visual/Rendering-type/Screen"
parsed = wrapper.parse_hed_string(hed_string)
print(f"Valid: {parsed['valid']}")
print(f"Tags found: {len(parsed['tags'])}")
```

### Generating HED Sidecars

**Basic sidecar generation:**

```python
from hedtools_integration import create_hed_wrapper

wrapper = create_hed_wrapper()
await wrapper.load_schema()

# Generate sidecar template
columns_to_process = ['trial_type', 'response', 'condition']
sidecar = await wrapper.generate_sidecar_template(events_df, columns_to_process)

# Save sidecar
from hedtools_integration import create_file_handler
handler = create_file_handler()
await handler.save_json_file(sidecar, Path("events.json"))
```

**Advanced sidecar customization:**

```python
# Customize sidecar based on analysis
analysis = await analyzer.analyze_events_file(events_path)
high_priority_columns = [
    c['column'] for c in analysis['hed_candidates'] 
    if c['priority'] == 'high'
]

# Generate focused sidecar
sidecar = await wrapper.generate_sidecar_template(
    events_df, 
    high_priority_columns
)

# Add custom descriptions
for column in high_priority_columns:
    if column in sidecar:
        sidecar[column]['Description'] = f"Custom description for {column}"

await handler.save_json_file(sidecar, Path("custom_events.json"))
```

## MCP Integration

The HED MCP Server provides tools for AI-powered HED annotation through the Model Context Protocol.

### Setting up MCP Server

**For Claude Desktop integration:**

```json
{
  "mcpServers": {
    "hed-mcp": {
      "command": "uv",
      "args": ["run", "python", "-m", "hedtools_integration.server"],
      "cwd": "/path/to/hed-mcp"
    }
  }
}
```

**Programmatic server usage:**

```python
from hedtools_integration import create_server
import asyncio

async def run_server():
    server = create_server()
    
    # Setup and start server
    server.setup()
    await server.start("stdio")

# Run server
asyncio.run(run_server())
```

### Available MCP Tools

Once implemented, the server will provide these tools:

#### 1. analyze_event_columns

Analyzes BIDS events file columns:

```
Tool: analyze_event_columns
Parameters:
  - file_path: Path to BIDS events file
  - output_format: "summary" | "detailed" | "hed_ready"

Returns:
  - Column analysis with HED suitability
  - Unique value summaries
  - BIDS compliance status
  - Recommendations for HED annotation
```

#### 2. generate_hed_sidecar

Generates HED sidecar templates:

```
Tool: generate_hed_sidecar
Parameters:
  - events_file: Path to BIDS events file
  - schema_version: HED schema version (default: "latest")
  - columns: Specific columns to process (optional)
  - output_path: Where to save the sidecar (optional)

Returns:
  - Generated HED sidecar JSON
  - Validation status
  - Performance metrics
```

### Available MCP Resources

#### hed_schemas

Lists available HED schemas:

```
Resource: hed_schemas
Returns:
  - Available schema versions
  - Schema metadata and descriptions
  - Download URLs and file sizes
```

## API Reference

### Package-Level Functions

```python
# Factory functions
create_server() -> HEDServer
create_hed_wrapper(schema_version: str = "latest") -> HEDWrapper
create_column_analyzer() -> BIDSColumnAnalyzer
create_file_handler() -> FileHandler
create_integration_suite(schema_version: str = "latest") -> dict

# Utility functions
get_package_info() -> dict
validate_installation() -> dict
quick_analyze_events(file_path, output_path=None) -> dict
```

### Core Classes

#### HEDWrapper

```python
class HEDWrapper:
    async def load_schema(version: str = None, custom_path: Path = None) -> bool
    async def validate_events(events_data: Union[pd.DataFrame, dict], sidecar: dict = None) -> dict
    async def generate_sidecar_template(events_df: pd.DataFrame, columns: List[str] = None) -> dict
    def get_available_schemas() -> List[dict]
    def parse_hed_string(hed_string: str) -> dict
    def get_schema_info() -> dict
```

#### BIDSColumnAnalyzer

```python
class BIDSColumnAnalyzer:
    async def analyze_events_file(file_path: Path) -> dict
    async def suggest_hed_annotations(column_data: pd.Series, column_name: str = None) -> List[dict]
    def get_summary() -> dict
```

#### FileHandler

```python
class FileHandler:
    @staticmethod
    async def load_events_file(file_path: Path) -> Optional[pd.DataFrame]
    @staticmethod
    async def save_events_file(df: pd.DataFrame, file_path: Path, format: str = 'tsv') -> bool
    @staticmethod
    async def load_json_file(file_path: Path) -> Optional[dict]
    @staticmethod
    async def save_json_file(data: dict, file_path: Path, indent: int = 2) -> bool
    @staticmethod
    async def validate_bids_events_structure(file_path: Path) -> dict
```

#### HEDServer

```python
class HEDServer:
    def setup() -> None
    async def start(transport: str = "stdio") -> None
    async def stop() -> None
    def get_server_info() -> dict
```

## Common Workflows

### Workflow 1: Basic Events Analysis

```python
async def analyze_bids_events(events_path: str):
    """Analyze BIDS events file for HED annotation readiness."""
    from hedtools_integration import create_column_analyzer
    from pathlib import Path
    
    analyzer = create_column_analyzer()
    path = Path(events_path)
    
    # Analyze the file
    analysis = await analyzer.analyze_events_file(path)
    
    # Report findings
    print(f"File: {analysis['file_path']}")
    print(f"Rows: {analysis['file_info']['total_rows']}")
    print(f"Columns: {analysis['file_info']['total_columns']}")
    print(f"BIDS valid: {analysis['bids_compliance']['valid']}")
    print(f"HED candidates: {len(analysis['hed_candidates'])}")
    
    # Show recommendations
    for rec in analysis['recommendations']:
        print(f"• {rec}")
    
    return analysis

# Usage
analysis = await analyze_bids_events("sub-01_task-faces_events.tsv")
```

### Workflow 2: Complete Sidecar Generation

```python
async def generate_complete_sidecar(events_path: str, output_path: str = None):
    """Generate a complete HED sidecar from events file."""
    from hedtools_integration import create_integration_suite
    from pathlib import Path
    
    # Create all components
    suite = create_integration_suite()
    analyzer = suite['column_analyzer']
    wrapper = suite['hed_wrapper']
    handler = suite['file_handler']
    
    events_path = Path(events_path)
    output_path = Path(output_path or events_path.with_suffix('.json'))
    
    # Step 1: Analyze columns
    print("🔍 Analyzing events file...")
    analysis = await analyzer.analyze_events_file(events_path)
    
    if not analysis['bids_compliance']['valid']:
        print("❌ BIDS validation failed:")
        for error in analysis['bids_compliance']['errors']:
            print(f"   {error}")
        return False
    
    # Step 2: Load events data
    print("📂 Loading events data...")
    events_df = await handler.load_events_file(events_path)
    
    # Step 3: Load HED schema
    print("📋 Loading HED schema...")
    await wrapper.load_schema()
    
    # Step 4: Generate sidecar
    print("🏗️  Generating HED sidecar...")
    hed_columns = [c['column'] for c in analysis['hed_candidates']]
    sidecar = await wrapper.generate_sidecar_template(events_df, hed_columns)
    
    # Step 5: Save sidecar
    print(f"💾 Saving sidecar to {output_path}...")
    success = await handler.save_json_file(sidecar, output_path)
    
    if success:
        print(f"✅ Complete! Generated sidecar with {len(hed_columns)} HED columns.")
        return sidecar
    else:
        print("❌ Failed to save sidecar.")
        return None

# Usage
sidecar = await generate_complete_sidecar(
    "sub-01_task-faces_events.tsv",
    "sub-01_task-faces_events.json"
)
```

### Workflow 3: Batch Processing

```python
async def process_multiple_events_files(file_paths: List[str]):
    """Process multiple events files in batch."""
    from hedtools_integration import create_integration_suite
    import asyncio
    
    suite = create_integration_suite()
    
    async def process_single_file(file_path):
        try:
            analysis = await suite['column_analyzer'].analyze_events_file(Path(file_path))
            return {
                'file': file_path,
                'success': True,
                'hed_candidates': len(analysis['hed_candidates']),
                'bids_valid': analysis['bids_compliance']['valid']
            }
        except Exception as e:
            return {
                'file': file_path,
                'success': False,
                'error': str(e)
            }
    
    # Process files concurrently
    results = await asyncio.gather(*[
        process_single_file(path) for path in file_paths
    ])
    
    # Summary report
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"📊 Batch Processing Results:")
    print(f"   ✅ Successful: {len(successful)}")
    print(f"   ❌ Failed: {len(failed)}")
    print(f"   📋 Total HED candidates: {sum(r['hed_candidates'] for r in successful)}")
    
    return results

# Usage
file_list = [
    "sub-01_task-faces_events.tsv",
    "sub-01_task-scenes_events.tsv",
    "sub-02_task-faces_events.tsv"
]
batch_results = await process_multiple_events_files(file_list)
```

## Performance Tips

### 1. Efficient File Loading

```python
# Load large files efficiently
async def load_large_events_file(file_path: Path, chunk_size: int = 10000):
    """Load large events files in chunks."""
    chunks = []
    for chunk in pd.read_csv(file_path, sep='\t', chunksize=chunk_size):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)
```

### 2. Optimize Column Analysis

```python
# Analyze specific columns only
target_columns = ['trial_type', 'condition', 'response']
filtered_df = events_df[['onset', 'duration'] + target_columns]
analysis = await analyzer.analyze_events_file(filtered_df)
```

### 3. Cache Schema Loading

```python
# Reuse wrapper instances
class HEDProcessingManager:
    def __init__(self):
        self.wrapper = None
        
    async def get_wrapper(self, schema_version="latest"):
        if self.wrapper is None:
            self.wrapper = create_hed_wrapper(schema_version)
            await self.wrapper.load_schema()
        return self.wrapper

manager = HEDProcessingManager()
wrapper = await manager.get_wrapper()
```

### 4. Parallel Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def parallel_analysis(file_paths: List[Path]):
    """Analyze multiple files in parallel."""
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        tasks = [
            loop.run_in_executor(executor, analyze_single_file, path)
            for path in file_paths
        ]
        
        results = await asyncio.gather(*tasks)
    
    return results
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

```python
# Check component availability
import hedtools_integration
info = hedtools_integration.get_package_info()

if not info['components']['hed_wrapper']:
    print("HED wrapper not available - install hedtools")
    
if not info['components']['server']:
    print("Server not available - install fastmcp")
```

#### 2. File Loading Issues

```python
# Robust file loading with error handling
async def safe_load_events(file_path: Path):
    try:
        handler = create_file_handler()
        
        # Validate file first
        if not handler.validate_file_format(file_path, ['.tsv', '.csv']):
            return None
            
        # Load with fallback encoding
        try:
            df = await handler.load_events_file(file_path)
        except UnicodeDecodeError:
            # Try different encoding
            df = pd.read_csv(file_path, sep='\t', encoding='latin-1')
            
        return df
        
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None
```

#### 3. Schema Loading Problems

```python
# Check schema availability
wrapper = create_hed_wrapper()
available_schemas = wrapper.get_available_schemas()

if not available_schemas:
    print("No HED schemas available - check hedtools installation")
else:
    print(f"Available schemas: {[s['version'] for s in available_schemas]}")
```

#### 4. Memory Issues with Large Files

```python
# Process large files efficiently
async def analyze_large_file(file_path: Path, max_rows: int = 50000):
    """Analyze large files by sampling."""
    df = await handler.load_events_file(file_path)
    
    if len(df) > max_rows:
        # Sample the data
        sample_df = df.sample(n=max_rows, random_state=42)
        print(f"Analyzing sample of {max_rows} rows from {len(df)} total rows")
        df = sample_df
    
    return await analyzer.analyze_events_file(df)
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hedtools_integration')

# Check detailed component status
validation = hedtools_integration.validate_installation()
for component, status in validation.items():
    logger.debug(f"{component}: {status}")
```

### Getting Help

If you encounter issues:

1. **Check the logs**: Enable debug logging to see detailed error messages
2. **Validate installation**: Use `validate_installation()` to check component status
3. **Test with minimal examples**: Start with simple test cases
4. **Review file formats**: Ensure BIDS compliance for events files
5. **Check dependencies**: Verify hedtools and other dependencies are properly installed

For additional support:
- Review the [Installation Guide](installation.md)
- Check [HED Integration documentation](hed_integration.md)
- Open an issue on [GitHub](https://github.com/hed-standard/hed-mcp/issues) 