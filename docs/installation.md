# Installation Guide

This guide provides detailed instructions for installing and setting up the HED MCP Server for development and production use.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Development Installation](#development-installation)
- [Production Installation](#production-installation)
- [Docker Installation](#docker-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Updating](#updating)

## Prerequisites

### System Requirements

- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB recommended for large datasets)
- **Storage**: 500MB for the package and dependencies

### Required Tools

#### uv (Recommended)

[uv](https://docs.astral.sh/uv/) is the recommended package manager for this project, offering fast dependency resolution and reproducible builds.

**Installation:**

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Via pip (cross-platform)
pip install uv

# Via homebrew (macOS)
brew install uv

# Via scoop (Windows)
scoop install uv
```

**Verify installation:**
```bash
uv --version
```

#### Alternative: pip + venv

If you prefer traditional Python tooling:

```bash
python -m pip install --upgrade pip
python -m venv --help  # Verify venv is available
```

### Optional Dependencies

#### Git

Required for development installation:
- **macOS**: `brew install git` or Xcode Command Line Tools
- **Ubuntu/Debian**: `sudo apt-get install git`
- **Windows**: Download from [git-scm.com](https://git-scm.com/)

## Development Installation

Development installation provides full access to source code, tests, and development tools.

### Method 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/hed-standard/hed-mcp.git
cd hed-mcp

# Set up development environment
uv sync --dev

# Activate the environment (optional with uv)
source .venv/bin/activate

# Verify installation
uv run python tests/test_imports.py
```

### Method 2: Using pip + venv

```bash
# Clone the repository
git clone https://github.com/hed-standard/hed-mcp.git
cd hed-mcp

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install development dependencies
pip install -e .[dev]

# Verify installation
python tests/test_imports.py
```

### Development Dependencies

The development installation includes:

- **Core dependencies**: hedtools, mcp, pandas, numpy
- **Development tools**: pytest, black, isort, mypy, ruff
- **Documentation**: sphinx, mkdocs
- **Additional utilities**: pre-commit, jupyter

## Production Installation

For production use or integration into other projects.

### Method 1: From PyPI (When Available)

```bash
# Install from PyPI
pip install hed-mcp-server

# Or with uv
uv add hed-mcp-server
```

### Method 2: From Source

```bash
# Clone and install
git clone https://github.com/hed-standard/hed-mcp.git
cd hed-mcp

# Install without development dependencies
pip install .

# Or with uv
uv sync --no-dev
```

### Method 3: Direct from GitHub

```bash
# Install latest main branch
pip install git+https://github.com/hed-standard/hed-mcp.git

# Install specific version
pip install git+https://github.com/hed-standard/hed-mcp.git@v0.1.0
```

## Docker Installation

Docker provides a containerized installation option for consistent deployment.

### Dockerfile

```dockerfile
FROM python:3.11-slim

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN uv sync --no-dev

# Expose port (if using HTTP transport)
EXPOSE 8000

# Run the server
CMD ["uv", "run", "python", "-m", "hed_tools.server"]
```

### Building and Running

```bash
# Build the image
docker build -t hed-mcp-server .

# Run the container
docker run -p 8000:8000 hed-mcp-server

# With volume mounting for data
docker run -v /path/to/data:/app/data hed-mcp-server
```

### Docker Compose

```yaml
version: '3.8'
services:
  hed-mcp-server:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - LOG_LEVEL=info
```

## Verification

### Basic Functionality Test

```bash
# Test package imports
python -c "import hed_tools; print('✅ Package imported successfully')"

# Run import tests
uv run python tests/test_imports.py

# Check package info
python -c "
import hed_tools
info = hed_tools.get_package_info()
print(f'Package: {info[\"name\"]} v{info[\"version\"]}')
print(f'Components available: {sum(info[\"components\"].values())}/4')
"
```

### Dependency Verification

```bash
# Check all dependencies
python -c "
import hed_tools
validation = hed_tools.validate_installation()
print('Installation Status:', '✅ Valid' if validation['valid'] else '❌ Issues')
for error in validation['errors']:
    print(f'Error: {error}')
for warning in validation['warnings']:
    print(f'Warning: {warning}')
"
```

### Server Test

```bash
# Test server creation
python -c "
from hed_tools import create_server
server = create_server()
info = server.get_server_info()
print(f'Server: {info[\"name\"]} ready: {info[\"ready\"]}')
"
```

## Troubleshooting

### Common Issues

#### 1. Python Version Compatibility

**Error**: `This package requires Python 3.10 or higher`

**Solution**:
```bash
# Check Python version
python --version

# Install Python 3.10+ using pyenv (recommended)
curl https://pyenv.run | bash
pyenv install 3.11.7
pyenv global 3.11.7

# Or use conda
conda create -n hed-mcp python=3.11
conda activate hed-mcp
```

#### 2. uv Installation Issues

**Error**: `uv: command not found`

**Solution**:
```bash
# Ensure uv is in PATH
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Or reinstall uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 3. HED Library Import Errors

**Error**: `ImportError: No module named 'hed'`

**Solution**:
```bash
# Install hedtools manually
pip install hedtools>=0.5.0

# Or ensure development installation includes all dependencies
uv sync --dev
```

#### 4. FastMCP Import Issues

**Error**: `ImportError: No module named 'fastmcp'`

**Note**: FastMCP may not be available in early development. The package includes graceful fallbacks.

**Solution**:
```bash
# The package will run in stub mode
# Check installation status
python -c "
import hed_tools
validation = hed_tools.validate_installation()
print(validation['recommendations'])
"
```

#### 5. Permission Errors

**Error**: `Permission denied: '/usr/local/lib/python3.x/site-packages'`

**Solution**:
```bash
# Use virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Or install with --user flag
pip install --user hed-mcp-server
```

#### 6. Network/Proxy Issues

**Error**: Connection timeouts during installation

**Solution**:
```bash
# Configure pip for proxy
pip install --proxy http://proxy.company.com:port package-name

# Or use alternative index
pip install -i https://pypi.org/simple/ package-name

# With uv
uv sync --index-url https://pypi.org/simple/
```

### Platform-Specific Issues

#### Windows

- Use PowerShell or Command Prompt as Administrator if needed
- Ensure Windows Subsystem for Linux (WSL) is available for better compatibility
- Install Microsoft Visual C++ Build Tools if compilation errors occur

#### macOS

- Install Xcode Command Line Tools: `xcode-select --install`
- Use Homebrew for system dependencies: `brew install git python`
- Check for Apple Silicon compatibility

#### Linux

- Install development headers: `sudo apt-get install python3-dev build-essential`
- Ensure virtual environment support: `sudo apt-get install python3-venv`

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Enable debug logging with `export LOG_LEVEL=debug`
2. **Search existing issues**: [GitHub Issues](https://github.com/hed-standard/hed-mcp/issues)
3. **Create minimal reproduction**: Isolate the problem
4. **Open an issue**: Include system info, error messages, and steps to reproduce

### System Information for Bug Reports

```bash
# Collect system information
python -c "
import sys, platform, subprocess
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.architecture()}')
try:
    import hed_tools
    print(f'Package: {hed_tools.__version__}')
    print(f'Installation: {hed_tools.validate_installation()}')
except Exception as e:
    print(f'Package Error: {e}')
"
```

## Updating

### Development Environment

```bash
# Update to latest main branch
git pull origin main

# Update dependencies
uv sync --dev --upgrade

# Update specific package
uv add package-name@latest
```

### Production Environment

```bash
# Update from PyPI
pip install --upgrade hed-mcp-server

# Or with uv
uv sync --upgrade
```

### Checking for Updates

```bash
# Check for outdated packages
pip list --outdated

# With uv
uv tree --outdated
```

## Next Steps

After successful installation:

1. **Read the [Usage Guide](usage.md)** for basic operations
2. **Review [HED Integration](hed_integration.md)** for HED-specific features
3. **Check [Examples](../examples/)** for practical usage scenarios
4. **Set up your IDE** with the development environment

## Feedback

Help us improve this installation guide:
- Report installation issues on [GitHub](https://github.com/hed-standard/hed-mcp/issues)
- Suggest improvements via pull requests
- Share your installation experience with the community
