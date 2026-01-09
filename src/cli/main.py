"""Main CLI entry point for complaint handling system."""

import typer

from src.cli.evaluate import app as evaluate_app
from src.cli.process import app as process_app
from src.cli.review import app as review_app

# Create main Typer app
app = typer.Typer(
    name="complaints",
    help="AI-powered medical device complaint handling system with IMDRF coding.",
    no_args_is_help=True,
)

# Register subcommands
app.add_typer(process_app, name="process")
app.add_typer(review_app, name="review")
app.add_typer(evaluate_app, name="evaluate")


def main() -> None:
    """Entry point for the complaints CLI."""
    app()


if __name__ == "__main__":
    main()
