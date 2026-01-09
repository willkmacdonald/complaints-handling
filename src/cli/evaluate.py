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


@app.command("calibration")
def calibration_command(
    run_id: Annotated[
        str,
        typer.Argument(help="Run ID to analyze for calibration"),
    ],
    num_bins: Annotated[
        int,
        typer.Option(
            "--bins",
            "-b",
            help="Number of calibration bins",
        ),
    ] = 10,
    save_config: Annotated[
        bool,
        typer.Option(
            "--save-config",
            help="Save calibration config to file",
        ),
    ] = False,
) -> None:
    """Analyze confidence calibration for an evaluation run.

    Calculates Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    and finds optimal confidence thresholds.

    Example:
        complaints evaluate calibration abc123
        complaints evaluate calibration abc123 --save-config
    """
    from src.evaluation.calibration import (
        analyze_calibration,
        create_calibration_config,
        save_calibration_config,
    )

    storage = get_default_storage()
    result = storage.load_run(run_id)

    if result is None:
        print_error(f"Evaluation run '{run_id}' not found")
        print_info("Use 'complaints evaluate history' to see available runs")
        raise typer.Exit(code=1)

    console.print()
    console.print(f"[bold]Calibration Analysis for Run: {run_id}[/bold]")
    console.print()

    # Run calibration analysis
    analysis = analyze_calibration(result, num_bins=num_bins)

    # Display overall metrics
    summary_table = Table(title="Calibration Metrics")
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right")

    summary_table.add_row("Total Suggestions", str(analysis.total_suggestions))
    summary_table.add_row("Total Correct", str(analysis.total_correct))
    summary_table.add_row("Overall Accuracy", f"{analysis.overall_accuracy:.1%}")
    summary_table.add_row("Avg Confidence", f"{analysis.avg_confidence:.1%}")
    summary_table.add_row("", "")

    # Color-code calibration errors
    ece_style = (
        "green"
        if analysis.expected_calibration_error < 0.1
        else "yellow"
        if analysis.expected_calibration_error < 0.2
        else "red"
    )
    mce_style = (
        "green"
        if analysis.maximum_calibration_error < 0.15
        else "yellow"
        if analysis.maximum_calibration_error < 0.3
        else "red"
    )

    summary_table.add_row(
        "[bold]ECE[/bold]",
        f"[{ece_style}]{analysis.expected_calibration_error:.3f}[/{ece_style}]",
    )
    summary_table.add_row(
        "[bold]MCE[/bold]",
        f"[{mce_style}]{analysis.maximum_calibration_error:.3f}[/{mce_style}]",
    )
    summary_table.add_row("Brier Score", f"{analysis.brier_score:.3f}")
    summary_table.add_row("", "")
    summary_table.add_row(
        "[bold]Optimal Threshold[/bold]",
        f"{analysis.optimal_threshold:.2f}",
    )

    console.print(summary_table)

    # Display calibration bins
    console.print()
    bin_table = Table(title="Calibration Bins")
    bin_table.add_column("Bin Range", style="cyan")
    bin_table.add_column("Count", justify="right")
    bin_table.add_column("Correct", justify="right")
    bin_table.add_column("Avg Conf", justify="right")
    bin_table.add_column("Accuracy", justify="right")
    bin_table.add_column("Cal Error", justify="right")

    for bin_data in analysis.bins:
        if bin_data.count == 0:
            continue

        error_style = (
            "green"
            if bin_data.calibration_error < 0.1
            else "yellow"
            if bin_data.calibration_error < 0.2
            else "red"
        )

        bin_table.add_row(
            f"[{bin_data.bin_start:.1f}, {bin_data.bin_end:.1f})",
            str(bin_data.count),
            str(bin_data.correct_count),
            f"{bin_data.avg_confidence:.1%}",
            f"{bin_data.accuracy:.1%}",
            f"[{error_style}]{bin_data.calibration_error:.3f}[/{error_style}]",
        )

    console.print(bin_table)

    # Display threshold analysis
    if analysis.threshold_analysis:
        console.print()
        threshold_table = Table(title="Threshold Analysis")
        threshold_table.add_column("Threshold", style="cyan")
        threshold_table.add_column("Precision", justify="right")
        threshold_table.add_column("Recall", justify="right")
        threshold_table.add_column("F1", justify="right")

        for threshold, metrics in sorted(analysis.threshold_analysis.items()):
            is_optimal = float(threshold) == analysis.optimal_threshold
            prefix = "[bold]" if is_optimal else ""
            suffix = " *[/bold]" if is_optimal else ""

            f1_style = (
                "green"
                if metrics["f1"] >= 0.8
                else "yellow"
                if metrics["f1"] >= 0.7
                else "red"
            )

            threshold_table.add_row(
                f"{prefix}{threshold}{suffix}",
                f"{metrics['precision']:.1%}",
                f"{metrics['recall']:.1%}",
                f"[{f1_style}]{metrics['f1']:.1%}[/{f1_style}]",
            )

        console.print(threshold_table)
        console.print("[dim]* = Optimal threshold[/dim]")

    # Save config if requested
    if save_config:
        config = create_calibration_config(analysis)
        path = save_calibration_config(config)
        console.print()
        print_success(f"Calibration config saved to: {path}")

    console.print()


@app.command("ablation")
def ablation_command(
    strategies: Annotated[
        list[PromptStrategy] | None,
        typer.Option(
            "--strategy",
            "-s",
            help="Strategies to test (can specify multiple). Defaults to all.",
        ),
    ] = None,
    baseline: Annotated[
        PromptStrategy,
        typer.Option(
            "--baseline",
            "-b",
            help="Baseline strategy for comparisons",
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
    difficulty: Annotated[
        str | None,
        typer.Option(
            "--difficulty",
            help="Filter by difficulty level",
        ),
    ] = None,
) -> None:
    """Run ablation test comparing multiple prompt strategies.

    Tests each strategy on the same test cases and computes comparative
    metrics with statistical significance testing.

    Example:
        complaints evaluate ablation
        complaints evaluate ablation --strategy zero_shot --strategy few_shot
        complaints evaluate ablation --baseline few_shot
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from src.evaluation.ablation import (
        run_ablation_test,
        save_ablation_report,
    )

    try:
        # Build filters
        filters = EvaluationFilters(
            device_type=device_type,
            channel=channel,
            difficulty=difficulty,
        )

        # Initialize LLM client
        config = LLMConfig.from_env()
        client = LLMClient(config=config)

        # Determine strategies
        test_strategies = strategies or list(PromptStrategy)

        console.print()
        console.print("[bold]Starting Ablation Test[/bold]")
        console.print(f"Strategies: {', '.join(s.value for s in test_strategies)}")
        console.print(f"Baseline: {baseline.value}")

        if any([device_type, channel, difficulty]):
            console.print("[dim]Filters applied:[/dim]")
            if device_type:
                console.print(f"  Device Type: {device_type.value}")
            if channel:
                console.print(f"  Channel: {channel.value}")
            if difficulty:
                console.print(f"  Difficulty: {difficulty}")

        console.print()

        # Run ablation with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running ablation test...", total=None)

            def progress_callback(strategy: str, current: int, total: int) -> None:
                progress.update(
                    task,
                    description=f"Testing [{current}/{total}]: {strategy}",
                )

            report = run_ablation_test(
                client=client,
                strategies=test_strategies,
                filters=filters,
                baseline=baseline,
                progress_callback=progress_callback,
            )

        # Display results
        console.print()
        console.print("[bold green]Ablation Test Complete[/bold green]")
        console.print()

        _display_ablation_report(report)

        # Save report
        report_id = save_ablation_report(report)
        console.print()
        print_success(f"Report saved with ID: {report_id}")
        print_info(f"View report: complaints evaluate ablation-report {report_id}")

    except ValueError as e:
        print_error(f"Configuration error: {e}")
        print_info(
            "Ensure Azure OpenAI environment variables are set "
            "(AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME)"
        )
        raise typer.Exit(code=1) from None
    except Exception as e:
        logger.exception("Ablation test failed")
        print_error(f"Ablation test failed: {e}")
        raise typer.Exit(code=1) from None


@app.command("ablation-report")
def ablation_report_command(
    report_id: Annotated[
        str,
        typer.Argument(help="Report ID to display"),
    ],
) -> None:
    """Display a previous ablation test report.

    Example:
        complaints evaluate ablation-report abc123
    """
    from src.evaluation.ablation import load_ablation_report

    report = load_ablation_report(report_id)

    if report is None:
        print_error(f"Ablation report '{report_id}' not found")
        print_info(
            "Use 'complaints evaluate ablation-history' to see available reports"
        )
        raise typer.Exit(code=1)

    console.print()
    _display_ablation_report(report)
    console.print()


@app.command("ablation-history")
def ablation_history_command(
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="Maximum number of reports to show",
        ),
    ] = 20,
) -> None:
    """List recent ablation test reports.

    Example:
        complaints evaluate ablation-history
        complaints evaluate ablation-history --limit 10
    """
    from src.evaluation.ablation import list_ablation_reports

    reports = list_ablation_reports(limit=limit)

    if not reports:
        print_info("No ablation reports found")
        print_info("Run 'complaints evaluate ablation' to create one")
        return

    console.print()
    table = Table(title=f"Recent Ablation Reports (showing {len(reports)})")
    table.add_column("Report ID", style="cyan", no_wrap=True)
    table.add_column("Timestamp")
    table.add_column("Strategies")
    table.add_column("Cases", justify="right")
    table.add_column("Best Strategy")

    for report in reports:
        timestamp_str = report.get("timestamp", "")
        if timestamp_str:
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                timestamp_str = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, AttributeError):
                pass

        table.add_row(
            report.get("report_id", "unknown"),
            timestamp_str,
            report.get("strategies", ""),
            report.get("test_cases", "0"),
            report.get("best_strategy", "unknown"),
        )

    console.print(table)
    console.print()
    print_info("Use 'complaints evaluate ablation-report <report_id>' for details")


@app.command("compare")
def compare_command(
    run_id_a: Annotated[
        str,
        typer.Argument(help="First run ID (baseline)"),
    ],
    run_id_b: Annotated[
        str,
        typer.Argument(help="Second run ID (comparison)"),
    ],
) -> None:
    """Compare two evaluation runs.

    Computes F1 delta, token usage difference, and statistical significance
    using a paired t-test.

    Example:
        complaints evaluate compare abc123 def456
    """
    from src.evaluation.ablation import StrategyMetrics, compare_strategies

    storage = get_default_storage()

    result_a = storage.load_run(run_id_a)
    if result_a is None:
        print_error(f"Evaluation run '{run_id_a}' not found")
        raise typer.Exit(code=1)

    result_b = storage.load_run(run_id_b)
    if result_b is None:
        print_error(f"Evaluation run '{run_id_b}' not found")
        raise typer.Exit(code=1)

    # Build metrics
    metrics_a = StrategyMetrics(
        strategy=result_a.metadata.strategy,
        run_id=result_a.metadata.run_id,
        test_case_count=result_a.metadata.test_case_count,
        precision=result_a.overall_precision or 0.0,
        recall=result_a.overall_recall or 0.0,
        f1_score=result_a.overall_f1 or 0.0,
        exact_match_rate=result_a.exact_match_rate or 0.0,
        total_tokens=result_a.metadata.token_stats.total_tokens,
        duration_ms=result_a.metadata.total_duration_ms,
    )

    metrics_b = StrategyMetrics(
        strategy=result_b.metadata.strategy,
        run_id=result_b.metadata.run_id,
        test_case_count=result_b.metadata.test_case_count,
        precision=result_b.overall_precision or 0.0,
        recall=result_b.overall_recall or 0.0,
        f1_score=result_b.overall_f1 or 0.0,
        exact_match_rate=result_b.exact_match_rate or 0.0,
        total_tokens=result_b.metadata.token_stats.total_tokens,
        duration_ms=result_b.metadata.total_duration_ms,
    )

    comparison = compare_strategies(result_a, metrics_a, result_b, metrics_b)

    console.print()
    console.print("[bold]Strategy Comparison[/bold]")
    console.print()

    # Summary table
    table = Table()
    table.add_column("Metric", style="bold")
    table.add_column(f"{run_id_a} ({metrics_a.strategy.value})", justify="right")
    table.add_column(f"{run_id_b} ({metrics_b.strategy.value})", justify="right")
    table.add_column("Delta", justify="right")

    # F1 row
    f1_delta_str = f"{comparison.f1_delta:+.1%}"
    f1_style = (
        "green" if comparison.f1_delta > 0 else "red" if comparison.f1_delta < 0 else ""
    )
    table.add_row(
        "F1 Score",
        f"{metrics_a.f1_score:.1%}",
        f"{metrics_b.f1_score:.1%}",
        f"[{f1_style}]{f1_delta_str}[/{f1_style}]" if f1_style else f1_delta_str,
    )

    # Precision row
    p_delta_str = f"{comparison.precision_delta:+.1%}"
    table.add_row(
        "Precision",
        f"{metrics_a.precision:.1%}",
        f"{metrics_b.precision:.1%}",
        p_delta_str,
    )

    # Recall row
    r_delta_str = f"{comparison.recall_delta:+.1%}"
    table.add_row(
        "Recall",
        f"{metrics_a.recall:.1%}",
        f"{metrics_b.recall:.1%}",
        r_delta_str,
    )

    # Token row
    token_delta_str = f"{comparison.token_delta:+,}"
    table.add_row(
        "Tokens",
        f"{metrics_a.total_tokens:,}",
        f"{metrics_b.total_tokens:,}",
        token_delta_str,
    )

    console.print(table)

    # Statistical significance
    console.print()
    if comparison.p_value is not None:
        sig_style = "green" if comparison.is_significant else "yellow"
        console.print(
            f"Statistical Significance: [{sig_style}]"
            f"{'YES' if comparison.is_significant else 'NO'}[/{sig_style}]"
        )
        console.print(f"  p-value: {comparison.p_value:.4f}")
        console.print(f"  t-statistic: {comparison.t_statistic:.4f}")
    else:
        console.print(
            "[yellow]Statistical significance not computed[/yellow] "
            "(install scipy: pip install scipy)"
        )

    console.print()


def _display_ablation_report(report) -> None:
    """Display ablation report in Rich tables."""
    from src.evaluation.ablation import AblationReport

    report: AblationReport = report

    # Summary
    summary_table = Table(title=f"Ablation Report: {report.report_id}")
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right")

    summary_table.add_row("Timestamp", report.timestamp.strftime("%Y-%m-%d %H:%M"))
    summary_table.add_row("Test Cases", str(report.test_case_count))
    summary_table.add_row("Total Tokens", f"{report.total_tokens:,}")
    summary_table.add_row("Total Duration", f"{report.total_duration_ms / 1000:.1f}s")
    summary_table.add_row("", "")
    summary_table.add_row(
        "[bold]Best Strategy (F1)[/bold]",
        report.best_strategy.value if report.best_strategy else "N/A",
    )
    summary_table.add_row(
        "[bold]Best Cost-Effective[/bold]",
        report.best_cost_effective_strategy.value
        if report.best_cost_effective_strategy
        else "N/A",
    )

    console.print(summary_table)

    # Strategy performance table
    console.print()
    perf_table = Table(title="Strategy Performance")
    perf_table.add_column("Strategy", style="cyan")
    perf_table.add_column("Precision", justify="right")
    perf_table.add_column("Recall", justify="right")
    perf_table.add_column("F1", justify="right")
    perf_table.add_column("Exact Match", justify="right")
    perf_table.add_column("Tokens", justify="right")
    perf_table.add_column("F1/1K Tok", justify="right")

    for metrics in sorted(report.strategy_metrics, key=lambda m: -m.f1_score):
        is_best_f1 = metrics.strategy == report.best_strategy
        is_best_cost = metrics.strategy == report.best_cost_effective_strategy

        strategy_name = metrics.strategy.value
        if is_best_f1:
            strategy_name += " *"
        if is_best_cost:
            strategy_name += " $"

        f1_style = (
            "green"
            if metrics.f1_score >= 0.8
            else "yellow"
            if metrics.f1_score >= 0.7
            else "red"
        )

        perf_table.add_row(
            strategy_name,
            f"{metrics.precision:.1%}",
            f"{metrics.recall:.1%}",
            f"[{f1_style}]{metrics.f1_score:.1%}[/{f1_style}]",
            f"{metrics.exact_match_rate:.1%}",
            f"{metrics.total_tokens:,}",
            f"{metrics.f1_per_1k_tokens:.4f}",
        )

    console.print(perf_table)
    console.print("[dim]* = Best F1 Score, $ = Most Cost-Effective[/dim]")

    # Comparisons vs baseline
    if report.comparisons:
        console.print()
        comp_table = Table(title=f"Comparisons vs {report.baseline_strategy.value}")
        comp_table.add_column("Strategy", style="cyan")
        comp_table.add_column("F1 Delta", justify="right")
        comp_table.add_column("Token Delta", justify="right")
        comp_table.add_column("Significant", justify="center")

        for comp in report.comparisons:
            delta_style = (
                "green" if comp.f1_delta > 0 else "red" if comp.f1_delta < 0 else ""
            )
            delta_str = f"{comp.f1_delta:+.1%}"
            if delta_style:
                delta_str = f"[{delta_style}]{delta_str}[/{delta_style}]"

            sig_str = (
                "[green]YES[/green]"
                if comp.is_significant
                else "[yellow]NO[/yellow]"
                if comp.p_value is not None
                else "[dim]N/A[/dim]"
            )

            comp_table.add_row(
                comp.comparison_strategy.value,
                delta_str,
                f"{comp.token_delta:+,}",
                sig_str,
            )

        console.print(comp_table)


# Main entry point for direct execution
if __name__ == "__main__":
    app()
