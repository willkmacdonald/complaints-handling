"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_root: Path) -> Path:
    """Return the data directory."""
    return project_root / "data"


@pytest.fixture
def test_cases_dir(data_dir: Path) -> Path:
    """Return the test cases directory."""
    return data_dir / "test_cases"
