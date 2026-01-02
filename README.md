# Dataset Health Monitor

[![Dataset Health Monitor](https://github.com/ambicuity/dataset-health-monitor/actions/workflows/dataset_monitor.yml/badge.svg)](https://github.com/ambicuity/dataset-health-monitor/actions/workflows/dataset_monitor.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Continuous monitoring for ML datasets with automatic GitHub issue creation.**

Dataset Health Monitor is a "CI for datasets" - a production-grade tool that continuously monitors machine learning datasets and automatically detects breakage or integrity issues using GitHub Actions.

## 🎯 Features

- **Link Validation**: Detect broken URLs (HTTP 4xx/5xx, timeouts)
- **File Existence Checks**: Verify expected files exist in archives or repositories
- **Checksum Verification**: SHA256 integrity validation
- **Schema Validation**: Detect CSV/JSON header changes
- **Automatic Issue Creation**: Opens GitHub Issues with detailed diagnostics
- **Smart Issue Management**: Prevents spam and auto-closes issues on recovery
- **State Persistence**: Tracks dataset health history

## 📁 Repository Structure

```
.
├── README.md                           # This file
├── CONTRIBUTING.md                     # Contribution guidelines
├── pyproject.toml                      # Python project configuration
├── requirements.txt                    # Python dependencies
├── datasets/
│   ├── datasets.yaml                   # Dataset configurations (source of truth)
│   └── examples.yaml                   # Example configurations
├── scripts/
│   ├── __init__.py
│   ├── monitor.py                      # Main orchestration script
│   ├── check_links.py                  # URL validation
│   ├── check_files.py                  # File existence checks
│   ├── checksum.py                     # SHA256 verification
│   ├── schema_check.py                 # CSV/JSON schema validation
│   ├── state_store.py                  # State persistence
│   └── open_issue.py                   # GitHub Issue management
├── state/
│   └── dataset_state.json              # Persisted state (auto-updated)
└── .github/
    └── workflows/
        └── dataset_monitor.yml         # GitHub Actions workflow
```

## 🚀 Quick Start

### 1. Fork or Clone

```bash
git clone https://github.com/ambicuity/dataset-health-monitor.git
cd dataset-health-monitor
```

### 2. Configure Datasets

Edit `datasets/datasets.yaml` to add your datasets:

```yaml
datasets:
  - name: my-dataset
    owner: my-org
    source_url: https://example.com/data.csv
    schema:
      - column1
      - column2
      - column3
    frequency: daily

  - name: model-weights
    owner: ml-team
    source_url: https://storage.example.com/weights.bin
    checksum: sha256:abc123def456...
    frequency: weekly
```

### 3. Run Locally (Optional)

```bash
# Install dependencies
pip install -r requirements.txt

# Run with dry-run mode (no issues created)
python -m scripts.monitor --dry-run --verbose
```

### 4. Enable GitHub Actions

The workflow runs automatically:
- **Daily** at 6:00 AM UTC (configurable)
- On **push** to main branch
- **Manually** via workflow_dispatch

## 📋 Dataset Configuration

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique identifier for the dataset |
| `owner` | string | Organization or user who owns the dataset |
| `source_url` | string | HTTP/HTTPS URL to the dataset |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `expected_files` | list | `[]` | Files expected in archives/repos |
| `checksum` | string | `null` | SHA256 checksum (`sha256:hex...`) |
| `schema` | list | `null` | Expected column/field names |
| `frequency` | string | `daily` | Check frequency (`daily`/`weekly`) |
| `branch` | string | `main` | Git branch (for GitHub repos) |

### Examples

#### HTTP URL with Schema Validation

```yaml
- name: covid-data
  owner: public-health
  source_url: https://data.example.com/covid.csv
  schema:
    - date
    - country
    - cases
    - deaths
  frequency: daily
```

#### GitHub Repository with Expected Files

```yaml
- name: ml-model
  owner: ai-team
  source_url: https://github.com/org/ml-model
  branch: main
  expected_files:
    - model.h5
    - config.json
    - requirements.txt
  frequency: daily
```

#### Archive with Checksum

```yaml
- name: training-data
  owner: data-team
  source_url: https://storage.example.com/data.zip
  expected_files:
    - train.csv
    - test.csv
  checksum: sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
  frequency: weekly
```

## 🔍 What Gets Checked

### 1. Link Validation
- HTTP status codes (4xx/5xx = failure)
- Connection timeouts
- SSL certificate issues

### 2. File Existence
- For archives: Extracts and verifies file list
- For GitHub repos: Checks raw file URLs
- Reports missing files with specific names

### 3. Checksum Verification
- Computes SHA256 hash of downloaded content
- Compares against expected checksum
- Detects any file modifications

### 4. Schema Validation
- **CSV**: Validates column headers
- **JSON**: Validates top-level field names
- Reports added/removed columns

## 🎫 Automatic Issue Creation

When a dataset fails health checks, an issue is automatically created with:

### Issue Title
```
[Dataset Health] my-dataset - Health Check Failed
```

### Issue Body
- Dataset name and URL
- List of detected issues
- Last known good state
- Suggested next steps

### Issue Labels
- `dataset-health` - All issues
- `dataset-broken` - URL inaccessible
- `checksum-change` - Checksum mismatch
- `missing-files` - Expected files missing
- `schema-change` - Schema changed

### Spam Prevention
- No duplicate issues for consecutive failures
- Existing open issues are not recreated
- Issues auto-close when dataset recovers

## ⚙️ GitHub Actions Workflow

### Triggers

```yaml
on:
  schedule:
    - cron: '0 6 * * *'     # Daily at 6 AM UTC
  workflow_dispatch:          # Manual trigger
  push:
    branches: [main]
  pull_request:
    branches: [main]
```

### Manual Trigger Options

| Input | Type | Description |
|-------|------|-------------|
| `dry_run` | boolean | Run without creating issues |
| `verbose` | boolean | Enable debug logging |

### Required Permissions

```yaml
permissions:
  contents: write    # Update state file
  issues: write      # Create/close issues
```

## 🛠️ Local Development

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Run Monitor

```bash
# Dry run (no issues created)
python -m scripts.monitor --dry-run

# Verbose output
python -m scripts.monitor --dry-run --verbose

# Custom config path
python -m scripts.monitor --config my-datasets.yaml --state my-state.json
```

### Run Tests

```bash
pytest
pytest --cov=scripts --cov-report=html
```

### Lint Code

```bash
ruff check scripts/
ruff format scripts/
mypy scripts/
```

## 🔧 Customization

### Change Schedule

Edit `.github/workflows/dataset_monitor.yml`:

```yaml
schedule:
  - cron: '0 */6 * * *'  # Every 6 hours
```

### Add Custom Labels

The workflow automatically creates labels on first run. To customize:

```bash
gh label create "custom-label" --description "Description" --color "hex"
```

### Extend Checks

Add new check modules in `scripts/` and integrate them in `scripts/monitor.py`.

## 📊 State Management

The state file (`state/dataset_state.json`) tracks:

```json
{
  "version": "1.0",
  "last_run_timestamp": "2024-01-15T06:00:00Z",
  "datasets": {
    "my-dataset": {
      "name": "my-dataset",
      "last_check_timestamp": "2024-01-15T06:00:00Z",
      "last_success_timestamp": "2024-01-15T06:00:00Z",
      "checksum": "sha256:abc123...",
      "schema_hash": "def456...",
      "is_healthy": true,
      "consecutive_failures": 0,
      "issue_number": null
    }
  }
}
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👤 Author

**Ritesh Rana**
- GitHub: [@ambicuity](https://github.com/ambicuity)
- Email: riteshrana36@gmail.com

---

⭐ Star this repo if you find it useful!