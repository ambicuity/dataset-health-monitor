# Contributing to Dataset Health Monitor

Thank you for your interest in contributing to Dataset Health Monitor! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting Changes](#submitting-changes)
- [Adding New Datasets](#adding-new-datasets)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/dataset-health-monitor.git
   cd dataset-health-monitor
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/ambicuity/dataset-health-monitor.git
   ```

## Development Setup

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

3. Verify the installation:
   ```bash
   python -m scripts.monitor --help
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=scripts --cov-report=html

# Run specific test file
pytest tests/test_check_links.py
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format and lint code
ruff check scripts/ --fix
ruff format scripts/

# Type checking
mypy scripts/
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-parquet-support`
- `fix/timeout-handling`
- `docs/update-readme`

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `style`: Changes that don't affect code meaning
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests
- `chore`: Changes to the build process or auxiliary tools

Examples:
```
feat(schema): add support for Parquet schema detection
fix(checksum): handle large files without memory overflow
docs: add examples for S3 datasets
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Write meaningful variable names

## Submitting Changes

1. Ensure all tests pass:
   ```bash
   pytest
   ```

2. Ensure code passes linting:
   ```bash
   ruff check scripts/
   mypy scripts/
   ```

3. Push your changes:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a Pull Request on GitHub with:
   - Clear title describing the change
   - Description of what changed and why
   - Reference to any related issues

### Pull Request Checklist

- [ ] Tests added for new functionality
- [ ] Documentation updated if needed
- [ ] Code passes all linting checks
- [ ] Commit messages follow conventions
- [ ] PR description explains the changes

## Adding New Datasets

To add new datasets for monitoring:

1. Edit `datasets/datasets.yaml`
2. Add your dataset configuration:
   ```yaml
   - name: your-dataset-name
     owner: dataset-owner
     source_url: https://example.com/data.csv
     expected_files:
       - file1.csv
       - file2.csv
     schema:
       - column1
       - column2
     frequency: daily
   ```

3. Test locally:
   ```bash
   python -m scripts.monitor --dry-run
   ```

4. Submit a PR with your changes

### Dataset Configuration Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique identifier for the dataset |
| `owner` | Yes | Organization or user who owns the dataset |
| `source_url` | Yes | HTTP/HTTPS URL to the dataset |
| `expected_files` | No | List of files expected in the dataset |
| `checksum` | No | SHA256 checksum for verification |
| `schema` | No | Expected column/field names |
| `frequency` | No | Check frequency (daily/weekly) |
| `branch` | No | Git branch for GitHub repos |

## Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Detailed steps to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Python version, OS, etc.
6. **Logs**: Relevant log output (use verbose mode: `--verbose`)

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed

## Questions?

Feel free to open an issue with the `question` label if you have any questions about contributing.

---

Thank you for contributing to Dataset Health Monitor! 🎉
