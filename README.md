# Medical Device Complaint Handling System

AI-powered complaint handling system for medical device manufacturers with IMDRF coding support.

## Overview

This system helps medical device companies process complaints from multiple channels (forms, emails, calls, letters), extract structured information, suggest IMDRF codes, and support human review—all while maintaining FDA regulatory compliance.

See [implementation-plan.md](implementation-plan.md) for the full project plan.

## Features (Planned)

- **Multi-channel intake**: Online forms, emails, call transcripts, scanned documents
- **AI-powered extraction**: Extract structured complaint data from unstructured text
- **IMDRF coding**: Suggest device problem and patient problem codes with confidence scores
- **Human-in-the-loop**: Review interface for approving/modifying AI suggestions
- **MDR determination**: Flag complaints requiring FDA Medical Device Reports
- **Audit trail**: Complete logging for regulatory compliance

## Project Status

Currently in **Phase 1: Single Channel MVP** - building form ingestion, LLM integration, and IMDRF coding services.

### Completed
- Phase 0: Foundation (IMDRF codes, test data, evaluation framework)
- PR-1: Form ingestion pipeline (`src/intake/`)
- PR-2: LLM integration foundation (`src/llm/`)
- PR-3: IMDRF code suggestion service (`src/coding/`)

## Setup

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd complaints-handling

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (including dev tools)
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Copy environment template
cp .env.example .env
# Edit .env with your configuration
```

### Running Tests

```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_smoke.py

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Lint code
ruff check .

# Format code
ruff format .

# Type check
mypy src/
```

## Project Structure

```
complaints-handling/
├── src/
│   ├── __init__.py
│   ├── models/          # Pydantic data models
│   ├── imdrf/           # IMDRF code reference database
│   ├── testing/         # Test case loader utilities
│   ├── evaluation/      # Evaluation framework
│   ├── intake/          # Form ingestion pipeline
│   ├── llm/             # LLM client and prompts
│   └── coding/          # IMDRF code suggestion service
├── data/
│   ├── imdrf/           # IMDRF code reference files
│   └── test_cases/      # Synthetic test complaints
│       ├── forms/
│       ├── emails/
│       ├── calls/
│       └── documents/
├── tests/
│   └── ...
├── .github/
│   └── workflows/
│       └── ci.yml
├── implementation-plan.md
├── pyproject.toml
└── README.md
```

## Regulatory Context

This system is designed for FDA-regulated environments:

- **21 CFR Part 820.198**: Complaint file requirements
- **21 CFR Part 11**: Electronic records and signatures
- **21 CFR Part 803**: Medical Device Reporting (MDR)

The AI component operates as a **decision support tool** with mandatory human review of all coding decisions.

## License

MIT
