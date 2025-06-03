# Known Issues and Limitations

## MCP Server Hanging on macOS (Python 3.12)

### Issue Description

**Status**: Known upstream issue
**Affected Platforms**: macOS with Python 3.12+
**GitHub Issue**: [MCP Python SDK #547](https://github.com/modelcontextprotocol/python-sdk/issues/547)

The HED MCP server hangs indefinitely during startup on macOS when using Python 3.12, specifically after the "Using selector: KqueueSelector" log message appears. This is not a bug in our implementation but rather a known issue with the MCP Python SDK's interaction with asyncio's KqueueSelector on macOS.

### Symptoms

- Server starts successfully and loads all components
- All dependencies validate correctly
- Console commands (`--help`, `--version`, `--check-deps`) work properly
- Server reaches "ready for connections" state
- Server hangs when attempting MCP protocol communication
- Process becomes unresponsive and requires force termination

### Affected Functionality

- ✅ Console script functionality (working)
- ✅ Server initialization (working)
- ✅ Tool registration (working)
- ✅ Dependency validation (working)
- ❌ MCP protocol communication (blocked)
- ❌ Client-server interaction (blocked)
- ❌ Integration testing with Claude Desktop (blocked)

### Workarounds

#### Option 1: Use Linux Environment
```bash
# Deploy on Linux (recommended for production)
docker run -it python:3.12-slim bash
# Install and run the server normally
```

#### Option 2: Use Python 3.11 on macOS
```bash
# Install Python 3.11 using pyenv
pyenv install 3.11.9
pyenv local 3.11.9

# Recreate virtual environment
rm -rf .venv
uv sync
```

#### Option 3: Mock Testing (Development Only)
```python
# Use direct function calls for testing
from hed_tools.server.server import app

# Test tools directly without MCP protocol
result = await app._validate_hed_string({
    "hed_string": "Red, Blue",
    "schema_version": "8.3.0"
})
```

### Status Updates

- **2025-06-03**: Issue confirmed and documented
- **Monitor**: [MCP Python SDK Issues](https://github.com/modelcontextprotocol/python-sdk/issues)
- **Expected Fix**: Awaiting upstream resolution

### Production Recommendations

1. **Deploy on Linux**: Use Docker with `python:3.11-slim` or `python:3.12-slim` on Linux
2. **CI/CD**: Run tests on Ubuntu runners in GitHub Actions
3. **Development**: Use Linux VM or container for MCP testing
4. **Monitoring**: Implement health checks that detect hang conditions

### Alternative Testing Approaches

For development on macOS, you can verify functionality using:

```bash
# Test console functionality
hed-mcp-server --version
hed-mcp-server --check-deps

# Test component loading
python -c "from hed_tools.server.server import app; print('✅ Server imports successfully')"

# Test HED integration directly
python -c "from hed_tools.hed_integration.hed_wrapper import HedWrapper; print('✅ HED integration available')"
```

### Related Issues

- [MCP Server Inconsistent Exception Handling #396](https://github.com/modelcontextprotocol/python-sdk/issues/396)
- Multiple reports of similar hanging behavior across different MCP implementations

---

## Other Limitations

### Resource Requirements

- **Memory**: Minimum 512MB RAM for HED schema caching
- **Storage**: ~50MB for HED schema files
- **Network**: Internet access required for initial schema downloads

### Supported File Formats

- ✅ TSV files (tab-separated values)
- ✅ CSV files (comma-separated values)
- ✅ Excel files (.xlsx, .xls)
- ❌ Binary formats (not supported)
- ❌ Compressed files (must be extracted first)

### HED Schema Compatibility

- **Supported**: HED schema versions 8.0.0 - 8.3.0
- **Recommended**: HED 8.3.0 (latest stable)
- **Legacy**: Limited support for HED 7.x (manual testing required)

---

## Reporting Issues

If you encounter issues not listed here:

1. Check if it's a known upstream MCP issue
2. Verify your Python version and platform
3. Test with the minimal reproduction case
4. Open an issue with full environment details

For urgent production issues, consider the Linux deployment workaround.
