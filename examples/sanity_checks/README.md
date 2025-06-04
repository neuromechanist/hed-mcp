# Sanity Check Scripts

This directory contains sanity check scripts that demonstrate and verify the functionality of the HED MCP server. These scripts are not automated unit tests but rather manual verification tools and usage examples.

## Purpose

These scripts serve multiple purposes:
- **Verification**: Check that core functionality works correctly after installation
- **Examples**: Demonstrate how to use various HED tools and functions
- **Debugging**: Help diagnose issues during development
- **Integration testing**: Verify end-to-end workflows work as expected

## Scripts Overview

### Core Functionality Tests
- `test_hed_import.py` - Verify HED library imports work correctly
- `test_wrapper.py` - Test the HED wrapper functionality
- `test_server.py` - Basic server functionality verification
- `test_tools.py` - Test individual HED tools

### Feature-Specific Tests
- `test_sidecar_generation.py` - Verify HED sidecar generation works
- `test_pipeline.py` - Test complete validation pipeline
- `test_debug.py` - Debug-specific functionality checks

### Production Tests
- `test_production_server.py` - Full production server testing
- `test_production_simple.py` - Simplified production tests
- `test_real_files_corrected.py` - Test with real data files

### Advanced Tests
- `test_mcp_inspector.py` - MCP protocol inspection and debugging

## Running the Scripts

These scripts should be run from the project root directory:

```bash
# Install the package first
pip install -e .

# Run individual sanity checks
python examples/sanity_checks/test_hed_import.py
python examples/sanity_checks/test_sidecar_generation.py

# Or run multiple scripts
for script in examples/sanity_checks/test_*.py; do
    echo "Running $script..."
    python "$script"
done
```

## Requirements

- The HED MCP package must be installed (`pip install -e .`)
- Required dependencies must be available (see `pyproject.toml`)
- Some scripts may require test data files in the `examples/` directory

## Output

Most scripts will create output in the current directory or a `test_output/` directory. Review these outputs to verify functionality.

## Note

These are **not** automated tests run by pytest. For proper unit tests, see the `tests/` directory. These scripts are for manual verification and demonstration purposes.
