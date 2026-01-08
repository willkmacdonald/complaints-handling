"""CLI module for complaint handling system."""

from src.cli.process import app as process_app
from src.cli.review import app as review_app

__all__ = ["review_app", "process_app"]
