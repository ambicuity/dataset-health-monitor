# Dataset Health Monitor

[![Dataset Health Monitor](https://github.com/ambicuity/dataset-health-monitor/actions/workflows/dataset_monitor.yml/badge.svg)](https://github.com/ambicuity/dataset-health-monitor/actions/workflows/dataset_monitor.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Datasets Health](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/ambicuity/dataset-health-monitor/main/badges/summary.json)](https://github.com/ambicuity/dataset-health-monitor)
[![GitHub App](https://img.shields.io/badge/GitHub%20App-Install-brightgreen)](https://github.com/apps/dataset-health-monitor)

**Continuous monitoring for ML datasets with automatic GitHub issue creation.**

Dataset Health Monitor is a "CI for datasets" - a production-grade tool that continuously monitors machine learning datasets and automatically detects breakage or integrity issues using GitHub Actions. It can also be installed as a **GitHub App** on any repository.

## 🎯 Features

- **Link Validation**: Detect broken URLs (HTTP 4xx/5xx, timeouts)
- **File Existence Checks**: Verify expected files exist in archives or repositories
- **Checksum Verification**: SHA256 integrity validation
- **Schema Validation**: Detect CSV/JSON header changes
- **Schema Drift Visualization**: Track and visualize schema changes over time with markdown diff tables
- **HuggingFace Dataset Support**: Monitor HuggingFace datasets with schema and availability checks
- **GitHub App**: Installable on any repository with app-based authentication
- **Automatic Issue Creation**: Opens GitHub Issues with detailed diagnostics and schema drift reports
- **Smart Issue Management**: Prevents spam and auto-closes issues on recovery
- **State Persistence**: Tracks dataset health history
- **Uptime Badges**: Live shields.io badges showing dataset health and 30-day uptime

## 📁 Repository Structure

```
.
├── README.md                           # This file
├── CONTRIBUTING.md                     # Contribution guidelines
├── DEPLOYMENT.md                       # GitHub App deployment guide
├── Dockerfile                          # Docker image for GitHub App
├── docker-compose.yml                  # Docker Compose configuration
├── pyproject.toml                      # Python project configuration
├── requirements.txt                    # Python dependencies
├── requirements-app.txt                # GitHub App dependencies
├── app/                                # GitHub App (Feature 4)
│   ├── __init__.py
│   ├── app.py                          # FastAPI application
│   ├── auth.py                         # GitHub App authentication (JWT)
│   └── webhook.py                      # Webhook handling
├── datasets/
│   ├── datasets.yaml                   # Dataset configurations (source of truth)
│   └── examples.yaml                   # Example configurations
├── scripts/
│   ├── __init__.py
│   ├── monitor.py                      # Main orchestration script
│   ├── check_links.py                  # URL validation
│   ├── check_files.py                  # File existence checks
│   ├── checksum.py                     # SHA256 verification
│   ├── schema_check.py                 # CSV/JSON schema validation with type inference
│   ├── schema_drift.py                 # Schema drift detection and visualization
│   ├── huggingface.py                  # HuggingFace dataset support
│   ├── state_store.py                  # State persistence with uptime tracking
│   ├── badges.py                       # Badge generation (shields.io)
│   └── open_issue.py                   # GitHub Issue management
├── badges/
│   └── *.json                          # Generated badge JSON files
├── state/
│   ├── dataset_state.json              # Persisted state (auto-updated)
│   └── schema_history/                 # Schema history and drift reports
│       └── *_schema_history.json       # Per-dataset schema history
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

## 📊 Uptime Badges

Dataset Health Monitor automatically generates shields.io-compatible badges showing:

- **Uptime Badge**: 30-day rolling uptime percentage
- **Status Badge**: Current health status (healthy/degraded/broken)
- **Summary Badge**: Overall health of all monitored datasets

### Using Badges in Your README

Add badges to your documentation using the shields.io endpoint format:

```markdown
<!-- Summary badge for all datasets -->
![Datasets Health](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/OWNER/REPO/main/badges/summary.json)

<!-- Per-dataset uptime badge -->
![Dataset Uptime](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/OWNER/REPO/main/badges/DATASET_NAME-uptime.json)

<!-- Per-dataset status badge -->
![Dataset Status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/OWNER/REPO/main/badges/DATASET_NAME-status.json)
```

### Badge Colors

| Uptime % | Color |
|----------|-------|
| ≥99% | ![brightgreen](https://img.shields.io/badge/-brightgreen-brightgreen) |
| ≥95% | ![green](https://img.shields.io/badge/-green-green) |
| ≥90% | ![yellowgreen](https://img.shields.io/badge/-yellowgreen-yellowgreen) |
| ≥80% | ![yellow](https://img.shields.io/badge/-yellow-yellow) |
| ≥70% | ![orange](https://img.shields.io/badge/-orange-orange) |
| <70% | ![red](https://img.shields.io/badge/-red-red) |

### Status Meanings

| Status | Description | Color |
|--------|-------------|-------|
| `healthy` | Dataset passed all checks | ![brightgreen](https://img.shields.io/badge/-brightgreen-brightgreen) |
| `degraded` | Currently failing but >50% uptime in last 7 days | ![yellow](https://img.shields.io/badge/-yellow-yellow) |
| `broken` | Currently failing with <50% uptime in last 7 days | ![red](https://img.shields.io/badge/-red-red) |

## 📈 Schema Drift Visualization

Dataset Health Monitor tracks schema changes over time for CSV and JSON datasets, providing:

### Automatic Type Inference

Column types are automatically inferred:
- **CSV**: Analyzes sample values to detect integer, float, boolean, datetime, or string types
- **JSON**: Uses native JSON types (string, integer, float, boolean, array, object)

### Schema Drift Detection

When a schema change is detected:
1. A markdown diff report is generated highlighting:
   - ➕ **Added columns** with their types
   - ➖ **Removed columns** with previous types
   - 🔄 **Type changes** showing old → new types
2. The report is automatically attached to GitHub Issues
3. Schema history is stored for trend analysis

### Example Schema Drift Report

```markdown
## Schema Drift Report: `my-dataset`

### ➕ Added Columns

| Column Name | Type |
|-------------|------|
| `new_field` | string |

### ➖ Removed Columns

| Column Name | Previous Type |
|-------------|---------------|
| `old_field` | integer |

### Full Schema Comparison

| Column | Status | Type |
|--------|--------|------|
| `id` | ⚪ Unchanged | integer |
| `name` | ⚪ Unchanged | string |
| `new_field` | 🟢 Added | string |
| `old_field` | 🔴 Removed | integer |
```

### Schema History Storage

Schema history is stored in `state/schema_history/`:
- One JSON file per dataset tracking up to 50 schema snapshots
- Each snapshot includes timestamp, schema hash, columns, and types
- Automatic pruning of old entries

## 🤗 HuggingFace Dataset Support

Dataset Health Monitor natively supports monitoring HuggingFace datasets using the `huggingface://` protocol.

### Configuration

```yaml
datasets:
  # Basic HuggingFace dataset
  - name: imdb
    owner: stanfordnlp
    source: huggingface://stanfordnlp/imdb
    split: train
    frequency: weekly

  # With schema validation
  - name: squad
    owner: huggingface
    source: huggingface://squad
    split: train
    schema:
      - id
      - title
      - context
      - question
      - answers
    frequency: weekly

  # Dataset with specific config
  - name: glue-sst2
    owner: huggingface
    source: huggingface://glue/sst2
    split: train
    frequency: weekly
```

### URL Format

```
huggingface://owner/dataset_name
huggingface://owner/dataset_name/config
huggingface://owner/dataset_name/config/split
```

### Features

- **Availability Check**: Verifies the dataset is accessible on HuggingFace Hub
- **Schema Extraction**: Automatically extracts feature names and types
- **Schema Validation**: Validates expected features exist in the dataset
- **Schema Drift**: Tracks schema changes over time (same as HTTP datasets)
- **Metadata Checksum**: Computes checksum based on dataset metadata
- **Split Information**: Reports available splits and row counts
- **Cache Management**: Respects HuggingFace cache for efficiency

### Installation

HuggingFace support requires the `datasets` library:

```bash
pip install datasets>=2.14.0
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## 🤖 GitHub App

Dataset Health Monitor is available as a GitHub App that can be installed on any repository.

### Features

- **Easy Installation**: One-click install on any repository
- **Automatic Monitoring**: Reads `datasets/datasets.yaml` from your repo
- **Issue Creation**: Opens issues in your repository when datasets fail
- **PR Comments**: Posts status updates on pull requests
- **App Authentication**: Uses secure GitHub App tokens (no personal access tokens)

### Installation

1. [Install the GitHub App](https://github.com/apps/dataset-health-monitor) on your repository
2. Add a `datasets/datasets.yaml` file to your repository
3. The app will automatically monitor your datasets

### Self-Hosting

You can also self-host the GitHub App:

```bash
# Clone the repository
git clone https://github.com/ambicuity/dataset-health-monitor.git
cd dataset-health-monitor

# Run with Docker
docker-compose up -d
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions including:
- Creating your own GitHub App
- Docker deployment
- Cloud deployment (Render, Fly.io, AWS)
- Configuration options

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Health check |
| `/webhook` | POST | GitHub webhook handler |
| `/installations` | GET | List app installations |
| `/check/{installation_id}/{owner}/{repo}` | POST | Trigger manual check |

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