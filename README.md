# HED Tools Integration

Integration tools for HED (Hierarchical Event Descriptor) through an MCP (Model Context Protocol) server interface.

## Overview

This package provides a comprehensive integration between HED tools and modern development workflows through an MCP server. It enables analysis of BIDS event data, automatic HED sidecar generation, and seamless integration with AI development environments.

## Features

- **HED Integration**: Wrapper for HED Python tools with simplified API
- **Column Analysis Engine**: Intelligent analysis of BIDS event file columns  
- **MCP Server**: FastMCP-based server for tool integration
- **Sidecar Generation**: Automated HED sidecar creation pipeline
- **Modern Python**: Built with Python 3.10+ and latest packaging standards

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) for dependency management

### Quick Start

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/hed-standard/hed-mcp.git
   cd hed-mcp
   ```

3. **Set up development environment**:
   ```bash
   uv sync --dev
   ```
   This command will:
   - Create a virtual environment in `.venv/`
   - Install all project and development dependencies
   - Generate `uv.lock` for reproducible builds
   - Install the package in editable mode

4. **Activate the environment**:
   ```bash
   source .venv/bin/activate
   ```
   
   Or use uv directly without activation:
   ```bash
   uv run python script.py
   uv run pytest
   ```

### Common Development Commands

```bash
# Run tests
uv run pytest

# Code formatting
uv run black src/ tests/
uv run isort src/ tests/

# Type checking
uv run mypy src/

# Add new dependency
uv add package-name

# Add development dependency  
uv add --dev package-name

# Update dependencies
uv sync --upgrade
```

### Project Structure

```
hed-mcp/
├── src/hedtools_integration/     # Main package
│   ├── server/                   # MCP server components
│   ├── tools/                    # Analysis tools
│   ├── hed_integration/         # HED-specific functionality
│   └── utils/                   # General utilities
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── docs/                        # Documentation
├── pyproject.toml              # Project configuration
├── uv.lock                     # Dependency lock file
└── README.md                   # This file
```

## Installation

### From Source

```bash
# Development installation
git clone https://github.com/hed-standard/hed-mcp.git
cd hed-mcp
uv sync --dev
```

### Dependencies

**Core Dependencies:**
- `hedtools>=0.5.0` - HED Python tools library
- `mcp>=1.0.0` - Model Context Protocol framework
- `pandas>=2.0.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computing

**Development Dependencies:**
- `pytest>=7.0.0` - Testing framework
- `black>=23.0.0` - Code formatting
- `isort>=5.12.0` - Import sorting
- `mypy>=1.0.0` - Type checking
- `ruff>=0.1.0` - Fast Python linter

## Usage

### Basic Usage

```python
import hedtools_integration

# Example usage will be added as the package develops
```

### MCP Server

```bash
# Start the MCP server (implementation in progress)
uv run python -m hedtools_integration.server
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `uv run pytest`
5. Format code: `uv run black . && uv run isort .`
6. Commit changes: `git commit -m "feat: description"`
7. Push to the branch: `git push origin feature-name`
8. Submit a pull request

### Code Style

This project uses:
- **Black** for code formatting (88 character line length)
- **isort** for import sorting (Black-compatible)
- **mypy** for type checking
- **ruff** for fast linting

Run `uv run pre-commit install` to set up pre-commit hooks.

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/hedtools_integration

# Run specific test categories
uv run pytest -m unit
uv run pytest -m integration
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links

- **Homepage**: https://github.com/hed-standard/hed-mcp
- **Documentation**: https://hed-mcp.readthedocs.io (coming soon)
- **Issues**: https://github.com/hed-standard/hed-mcp/issues
- **HED Tools**: https://github.com/hed-standard/hed-python

## Development Status

This project is in active development. See the [tasks](tasks/) directory for current development progress and roadmap. 