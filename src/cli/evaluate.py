"""CLI commands for running and analyzing evaluations."""

import logging
from typing import Annotated

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.cli.display import (
    console,
    print_error,
    print_info,
    print_success,
    print_warning,
)
from src.evaluation.models import EvaluationFilters, PromptStrategy
from src.evaluation.runner import run_evaluation
from src.evaluation.storage import get_default_storage
from src.llm import LLMClient, LLMConfig
from src.models.enums import DeviceType, IntakeChannel

logger = logging.getLogger(__name__)

# Create Typer app for evaluate commands
app = typer.Typer(
    name="evaluate",
    help="Run and analyze IMDRF coding evaluations.",
    no_args_is_help=True,
)


@app.command("run")
def run_command(
    strategy: Annotated[
        PromptStrategy,
        typer.Option(
            "--strategy",
            "-s",
            help="Prompt strategy to use",
        ),
    ] = PromptStrategy.ZERO_SHOT,
    device_type: Annotated[
        DeviceType | None,
        typer.Option(
            "--device-type",
            "-d",
            help="Filter by device type",
        ),
    ] = None,
    channel: Annotated[
        IntakeChannel | None,
        typer.Option(
            "--channel",
            "-c",
            help="Filter by intake channel",
        ),
    ] = None,
    severity: Annotated[
        str | None,
        typer.Option(
            "--severity",
            help="Filter by severity level",
        ),
    ] = None,
    difficulty: Annotated[
        str | None,
        typer.Option(
            "--difficulty",
            help="Filter by difficulty level",
        ),
    ] = None,
    no_save: Annotated[
        bool,
        typer.Option(
            "--no-save",
            help="Don't save results to storage",
        ),
    ] = False,
) -> None:
    """Run evaluation on test cases with the IMDRF coding service.

    Uses the LLM to suggest IMDRF codes for each test case and calculates
    accuracy metrics (precision, recall, F1) against ground truth.

    Example:
        complaints evaluate run --strategy zero_shot
        complaints evaluate run --channel form --difficulty easy
    """
    try:
        # Build filters
        filters = EvaluationFilters(
            device_type=device_type,
            channel=channel,
            severity=severity,
            difficulty=difficulty,
        )

        # Initialize LLM client
        config = LLMConfig.from_env()
        client = LLMClient(config=config)

        console.print()
        console.print(
            f"[bold]Starting evaluation with strategy: {strategy.value}[/bold]"
        )

        if any([device_type, channel, severity, difficulty]):
            console.print("[dim]Filters applied:[/dim]")
            if device_type:
                console.print(f"  Device Type: {device_type.value}")
            if channel:
                console.print(f"  Channel: {channel.value}")
            if severity:
                console.print(f"  Severity: {severity}")
            if difficulty:
                console.print(f"  Difficulty: {difficulty}")

        console.print()

        # Run evaluation with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating test cases...", total=None)

            def progress_callback(current: int, total: int, test_case_id: str) -> None:
                progress.update(
                    task,
                    description=f"Evaluating [{current}/{total}]: {test_case_id}",
                )

            result = run_evaluation(
                client=client,
                strategy=strategy,
                filters=filters,
                progress_callback=progress_callback,
            )

        # Display results
        console.print()
        console.print("[bold green]Evaluation Complete[/bold green]")
        console.print()

        # Summary table
        _display_evaluation_summary(result)

        # Save results
        if not no_save:
            storage = get_default_storage()
            run_id = storage.save_run(result)
            console.print()
            print_success(f"Results saved with run ID: {run_id}")
            print_info(f"View detailed report: complaints evaluate report {run_id}")

    except ValueError as e:
        print_error(f"Configuration error: {e}")
        print_info(
            "Ensure Azure OpenAI environment variables are set "
            "(AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME)"
        )
        raise typer.Exit(code=1) from None
    except Exception as e:
        logger.exception("Evaluation failed")
        print_error(f"Evaluation failed: {e}")
        raise typer.Exit(code=1) from None


def _display_evaluation_summary(result) -> None:
    """Display evaluation summary in Rich tables."""
    from src.evaluation.models import EvaluationRunResult

    result: EvaluationRunResult = result

    # Overall metrics table
    summary_table = Table(title=f"Evaluation Summary (Run: {result.metadata.run_id})")
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right")

    summary_table.add_row("Strategy", result.metadata.strategy.value)
    summary_table.add_row("Test Cases", str(result.metadata.test_case_count))
    summary_table.add_row("Duration", f"{result.metadata.total_duration_ms:.0f}ms")
    summary_table.add_row("Total Tokens", str(result.metadata.token_stats.total_tokens))
    summary_table.add_row("", "")

    if result.overall_precision is not None:
        summary_table.add_row("Precision", f"{result.overall_precision:.1%}")
    if result.overall_recall is not None:
        summary_table.add_row("Recall", f"{result.overall_recall:.1%}")
    if result.overall_f1 is not None:
        f1_style = (
            "green"
            if result.overall_f1 >= 0.8
            else "yellow"
            if result.overall_f1 >= 0.7
            else "red"
        )
        summary_table.add_row(
            "[bold]F1 Score[/bold]", f"[{f1_style}]{result.overall_f1:.1%}[/{f1_style}]"
        )
    if result.exact_match_rate is not None:
        summary_table.add_row("Exact Match", f"{result.exact_match_rate:.1%}")

    console.print(summary_table)

    # Breakdowns
    if result.by_difficulty:
        console.print()
        _display_breakdown_table("By Difficulty", result.by_difficulty)

    if result.by_device_type:
        console.print()
        _display_breakdown_table("By Device Type", result.by_device_type)

    if result.by_channel:
        console.print()
        _display_breakdown_table("By Channel", result.by_channel)

    # Show errors if any
    failed_cases = [r for r in result.results if not r.is_success]
    if failed_cases:
        console.print()
        print_warning(f"{len(failed_cases)} test case(s) failed:")
        for case in failed_cases[:5]:
            console.print(f"  - {case.test_case_id}: {case.error}")
        if len(failed_cases) > 5:
            console.print(f"  ... and {len(failed_cases) - 5} more")


def _display_breakdown_table(
    title: str, breakdown: dict[str, dict[str, float]]
) -> None:
    """Display a metrics breakdown table."""
    table = Table(title=title)
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")

    for key, metrics in sorted(breakdown.items()):
        f1_style = (
            "green"
            if metrics["f1"] >= 0.8
            else "yellow"
            if metrics["f1"] >= 0.7
            else "red"
        )
        table.add_row(
            key,
            str(int(metrics["count"])),
            f"{metrics['precision']:.1%}",
            f"{metrics['recall']:.1%}",
            f"[{f1_style}]{metrics['f1']:.1%}[/{f1_style}]",
        )

    console.print(table)


@app.command("report")
def report_command(
    run_id: Annotated[
        str,
        typer.Argument(help="Run ID to generate report for"),
    ],
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed per-case results",
        ),
    ] = False,
) -> None:
    """Generate a detailed report for a previous evaluation run.

    Example:
        complaints evaluate report abc123
        complaints evaluate report abc123 --verbose
    """
    storage = get_default_storage()
    result = storage.load_run(run_id)

    if result is None:
        print_error(f"Evaluation run '{run_id}' not found")
        print_info("Use 'complaints evaluate history' to see available runs")
        raise typer.Exit(code=1)

    console.print()
    _display_evaluation_summary(result)

    if verbose:
        console.print()
        console.print("[bold]Per-Case Results[/bold]")
        console.print()

        # Create results table
        results_table = Table(show_header=True)
        results_table.add_column("Test Case", style="cyan", no_wrap=True)
        results_table.add_column("Channel", style="dim")
        results_table.add_column("Difficulty", style="dim")
        results_table.add_column("Precision", justify="right")
        results_table.add_column("Recall", justify="right")
        results_table.add_column("F1", justify="right")
        results_table.add_column("Status")

        for case_result in result.results:
            if case_result.is_success and case_result.coding_metrics:
                metrics = case_result.coding_metrics
                f1_style = (
                    "green"
                    if metrics.f1_score >= 0.8
                    else "yellow"
                    if metrics.f1_score >= 0.7
                    else "red"
                )
                results_table.add_row(
                    case_result.test_case_id,
                    case_result.channel.value,
                    case_result.difficulty,
                    f"{metrics.precision:.1%}",
                    f"{metrics.recall:.1%}",
                    f"[{f1_style}]{metrics.f1_score:.1%}[/{f1_style}]",
                    "[green]OK[/green]",
                )
            else:
                results_table.add_row(
                    case_result.test_case_id,
                    case_result.channel.value,
                    case_result.difficulty,
                    "-",
                    "-",
                    "-",
                    "[red]Error[/red]",
                )

        console.print(results_table)

    console.print()


@app.command("history")
def history_command(
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="Maximum number of runs to show",
        ),
    ] = 20,
) -> None:
    """List recent evaluation runs.

    Example:
        complaints evaluate history
        complaints evaluate history --limit 10
    """
    storage = get_default_storage()
    runs = storage.list_runs(limit=limit)

    if not runs:
        print_info("No evaluation runs found")
        print_info("Run 'complaints evaluate run' to create one")
        return

    console.print()
    table = Table(title=f"Recent Evaluation Runs (showing {len(runs)})")
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("Timestamp")
    table.add_column("Strategy")
    table.add_column("Cases", justify="right")
    table.add_column("F1 Score", justify="right")

    for run in runs:
        # Format timestamp
        timestamp_str = run.get("timestamp", "")
        if timestamp_str:
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                timestamp_str = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, AttributeError):
                pass

        f1_str = run.get("f1_score", "N/A")
        if f1_str != "N/A":
            # Color code F1 score
            try:
                f1_val = float(f1_str.rstrip("%")) / 100
                f1_style = (
                    "green" if f1_val >= 0.8 else "yellow" if f1_val >= 0.7 else "red"
                )
                f1_str = f"[{f1_style}]{f1_str}[/{f1_style}]"
            except ValueError:
                pass

        table.add_row(
            run.get("run_id", "unknown"),
            timestamp_str,
            run.get("strategy", "unknown"),
            run.get("test_case_count", "0"),
            f1_str,
        )

    console.print(table)
    console.print()
    print_info("Use 'complaints evaluate report <run_id>' for detailed results")


# Main entry point for direct execution
if __name__ == "__main__":
    app()
