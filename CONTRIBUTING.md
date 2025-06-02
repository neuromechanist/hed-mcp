# Contributing to HED MCP Server

Welcome to the HED MCP Server project! We appreciate your interest in contributing to tools that make HED annotation more accessible to the neuroimaging research community.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Documentation Guidelines](#documentation-guidelines)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) for dependency management
- Git for version control
- Basic understanding of HED (Hierarchical Event Descriptors) and BIDS

### Understanding the Project

Before contributing, please familiarize yourself with:

- **HED Standard**: [hed-specification.org](https://hed-specification.org)
- **BIDS Specification**: [bids-specification.org](https://bids-specification.org)
- **Model Context Protocol**: [MCP Documentation](https://modelcontextprotocol.io)
- **FastMCP Framework**: Used for our MCP server implementation

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/your-username/hed-mcp.git
cd hed-mcp
```

### 2. Set Up Development Environment

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up the development environment
uv sync --dev

# Activate the environment
source .venv/bin/activate
```

### 3. Verify Installation

```bash
# Run basic import tests
uv run python tests/test_imports.py

# Run the test suite
uv run pytest

# Check code formatting
uv run black --check src/ tests/
uv run isort --check-only src/ tests/
```

## Development Workflow

### Branch Strategy

We use a feature branch workflow:

1. **Main Branch (`main`)**: Production-ready code
2. **Feature Branches**: `feature/description` or `fix/description`
3. **Release Branches**: `release/v0.1.0` (when needed)

### Creating a Feature Branch

```bash
# Create and switch to a new feature branch
git checkout -b feature/your-feature-name
git push -u origin feature/your-feature-name
```

### Adding New Features

1. **Plan Your Changes**: Open an issue to discuss major changes
2. **Write Tests First**: Follow TDD when possible
3. **Implement Changes**: Write clean, documented code
4. **Test Thoroughly**: Ensure all tests pass
5. **Update Documentation**: Update relevant docs
6. **Submit Pull Request**: Follow our PR template

### Workflow for Different Types of Changes

#### Adding New MCP Tools
1. Create tool function in appropriate module
2. Add tool registration in `server/server.py`
3. Write comprehensive tests
4. Add documentation and examples
5. Update API documentation

#### HED Integration Changes
1. Test against latest hedtools version
2. Ensure backward compatibility
3. Add integration tests
4. Update wrapper documentation

#### Performance Improvements
1. Add benchmark tests
2. Profile before and after changes
3. Document performance gains
4. Ensure no regression in functionality

## Code Style Guidelines

We maintain consistent code style using automated tools:

### Formatting

- **Black**: Code formatting (88 character line length)
- **isort**: Import sorting and organization
- **Ruff**: Fast Python linting

```bash
# Format code
uv run black src/ tests/
uv run isort src/ tests/

# Check linting
uv run ruff check src/ tests/

# Type checking
uv run mypy src/
```

### Style Guidelines

#### Python Code
- Use type hints for all function signatures
- Write descriptive variable and function names
- Keep functions focused and small (< 50 lines when possible)
- Use docstrings for all public functions and classes

#### Documentation
- Use clear, concise language
- Include code examples where helpful
- Reference official HED/BIDS documentation when relevant
- Keep examples up-to-date with API changes

#### Comments
- Explain **why**, not **what**
- Use comments sparingly for self-documenting code
- Add TODO comments for future improvements

### Code Organization

```python
# Standard library imports first
import asyncio
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports second
import pandas as pd
from fastmcp import FastMCP

# Local imports last
from hed_tools.hed_integration import HEDWrapper
from hed_tools.utils import FileHandler
```

### Error Handling

- Use specific exception types
- Provide helpful error messages
- Log errors appropriately
- Handle edge cases gracefully

```python
try:
    result = await process_events_file(file_path)
except FileNotFoundError:
    logger.error(f"Events file not found: {file_path}")
    raise ValueError(f"Events file does not exist: {file_path}")
except Exception as e:
    logger.error(f"Failed to process events file: {e}")
    raise
```

## Testing Requirements

### Test Categories

1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Validate performance requirements

### Writing Tests

#### Test Structure
```python
def test_function_name():
    """Test description explaining what is being tested."""
    # Arrange
    input_data = create_test_data()

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result.is_valid
    assert len(result.errors) == 0
```

#### Test Requirements
- All public functions must have tests
- Aim for >90% code coverage
- Include edge cases and error conditions
- Mock external dependencies (HED library, file system)

#### Test Data
- Use fixtures for reusable test data
- Keep test data minimal but realistic
- Avoid committing large test files

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/hed_tools --cov-report=html

# Run specific test categories
uv run pytest -m unit
uv run pytest -m integration
uv run pytest -m performance

# Run tests for specific module
uv run pytest tests/unit/test_column_analyzer.py
```

### Continuous Integration

All pull requests must:
- Pass all existing tests
- Maintain or improve code coverage
- Pass linting and type checking
- Include tests for new functionality

## Pull Request Process

### Before Submitting

1. **Sync with Main**: Rebase your branch on latest main
2. **Run Full Test Suite**: Ensure all tests pass
3. **Check Code Quality**: Run linting and type checking
4. **Update Documentation**: Include relevant doc updates
5. **Write Clear Commit Messages**: Use conventional commit format

### Commit Message Format

We use conventional commits for clear history:

```
feat(server): add analyze_event_columns MCP tool

- Implement column analysis functionality
- Add support for categorical and numeric columns
- Include comprehensive error handling
- Add unit tests with >95% coverage

Closes #123
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`

### Pull Request Template

When creating a PR, please:

1. **Use Descriptive Title**: Summarize the change clearly
2. **Reference Issues**: Link to related issues
3. **Describe Changes**: Explain what and why
4. **List Breaking Changes**: If any
5. **Include Testing**: Describe how you tested
6. **Update Checklist**: Complete the PR checklist

### Review Process

1. **Automated Checks**: CI must pass
2. **Code Review**: At least one maintainer review
3. **Testing**: Reviewer may test changes locally
4. **Documentation**: Verify docs are updated
5. **Merge**: Squash and merge after approval

## Documentation Guidelines

### API Documentation

- Use Sphinx-style docstrings
- Include parameter types and descriptions
- Provide usage examples
- Document exceptions that may be raised

```python
async def analyze_events_file(self, file_path: Path) -> Dict[str, Any]:
    """Analyze BIDS events file structure and columns.

    Args:
        file_path: Path to BIDS events file (TSV or CSV)

    Returns:
        Comprehensive analysis including column classification
        and HED annotation recommendations.

    Raises:
        FileNotFoundError: If the events file doesn't exist
        ValueError: If the file format is not supported

    Examples:
        >>> analyzer = BIDSColumnAnalyzer()
        >>> result = await analyzer.analyze_events_file(Path("events.tsv"))
        >>> print(f"Found {len(result['hed_candidates'])} HED candidates")
    """
```

### User Documentation

- Write for different skill levels
- Include working code examples
- Explain integration with external tools
- Keep examples current with API changes

### Inline Documentation

- Comment complex algorithms
- Explain business logic
- Document assumptions and constraints
- Reference relevant specifications

## Community Guidelines

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code contributions and reviews

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers and answer questions
- Focus on what's best for the community

### Getting Help

- Check existing issues and documentation first
- Provide minimal, reproducible examples
- Include relevant system information
- Be patient and respectful

## Project-Specific Guidelines

### HED Integration

- Always test against latest hedtools version
- Maintain compatibility with HED schema versions 8.0.0+
- Follow HED best practices and conventions
- Reference official HED documentation

### BIDS Compliance

- Ensure all BIDS file handling follows specification
- Validate against BIDS validator when possible
- Support common BIDS dataset structures
- Handle edge cases gracefully

### Performance Requirements

- Column analysis: < 2 seconds
- Sidecar generation: < 10 seconds
- Memory efficient for large datasets
- Optimize for common use cases

### Security Considerations

- Validate all file inputs
- Handle user-provided data safely
- Avoid path traversal vulnerabilities
- Log security-relevant events

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release branch
4. Run full test suite
5. Update documentation
6. Create GitHub release
7. Deploy to PyPI

## Questions?

If you have questions not covered here:

1. Check the [project documentation](docs/)
2. Search [existing issues](https://github.com/hed-standard/hed-mcp/issues)
3. Open a [new issue](https://github.com/hed-standard/hed-mcp/issues/new)

Thank you for contributing to the HED MCP Server project! 🎉
