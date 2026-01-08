"""Smoke tests to verify basic setup."""


def test_import_src() -> None:
    """Verify src package can be imported."""
    import src

    assert src.__version__ == "0.1.0"


def test_project_structure(project_root) -> None:
    """Verify expected directories exist."""
    assert (project_root / "src").is_dir()
    assert (project_root / "tests").is_dir()
    assert (project_root / "data").is_dir()
