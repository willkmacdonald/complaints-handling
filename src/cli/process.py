"""CLI commands for processing form submissions through the pipeline.

This module provides Typer-based CLI commands for processing forms
and generating IMDRF code suggestions and MDR determinations.
"""

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel
from rich.table import Table

from src.cli.display import (
    console,
    create_suggestions_table,
    format_confidence,
    print_error,
    print_info,
    print_success,
    print_warning,
)
from src.pipeline import ProcessingStatus, process_form
from src.pipeline.forms import process_form_file
from src.pipeline.models import ProcessingResult

logger = logging.getLogger(__name__)

# Create Typer app for process commands
app = typer.Typer(
    name="process",
    help="Process form submissions through the complaint handling pipeline.",
    no_args_is_help=True,
)


def _display_processing_result(result: ProcessingResult, verbose: bool = False) -> None:
    """Display processing result in a formatted output.

    Args:
        result: Processing result to display.
        verbose: Whether to show detailed output.
    """
    # Status panel
    status_color = {
        ProcessingStatus.SUCCESS: "green",
        ProcessingStatus.PARTIAL: "yellow",
        ProcessingStatus.FAILED: "red",
    }
    color = status_color.get(result.status, "white")

    console.print()
    console.print(
        Panel(
            f"[bold {color}]Status: {result.status.value.upper()}[/bold {color}]\n"
            f"Processing ID: {result.processing_id}\n"
            f"Complaint ID: {result.complaint_id or 'N/A'}",
            title="[bold]Processing Result[/bold]",
            border_style=color,
        )
    )

    # Form validation
    if result.validation_result:
        console.print()
        console.print("[bold cyan]Form Validation:[/bold cyan]")
        if result.validation_result.is_complete:
            console.print("  [green]All required fields present[/green]")
        else:
            console.print("  [red]Missing required fields:[/red]")
            for field in result.validation_result.missing_required:
                console.print(f"    - {field}")

        if result.validation_result.missing_recommended:
            console.print("  [yellow]Missing recommended fields:[/yellow]")
            for field in result.validation_result.missing_recommended:
                console.print(f"    - {field}")

        if result.validation_result.warnings:
            console.print("  [yellow]Warnings:[/yellow]")
            for warning in result.validation_result.warnings:
                console.print(f"    - {warning}")

    # Complaint details
    if result.complaint and verbose:
        console.print()
        console.print("[bold cyan]Complaint Details:[/bold cyan]")
        console.print(f"  Device: {result.complaint.device_info.device_name}")
        console.print(f"  Manufacturer: {result.complaint.device_info.manufacturer}")
        console.print(f"  Channel: {result.complaint.intake_channel.value}")
        console.print(f"  Status: {result.complaint.status.value}")

    # Coding suggestions
    if result.coding_result and result.coding_result.suggestions:
        console.print()
        console.print(create_suggestions_table(result.coding_result.suggestions))

        if verbose:
            console.print()
            console.print("[bold cyan]Coding Metrics:[/bold cyan]")
            console.print(f"  Model: {result.coding_result.model_used or 'N/A'}")
            console.print(f"  Tokens used: {result.coding_result.tokens_used}")
            console.print(f"  Latency: {result.coding_result.latency_ms:.0f}ms")
    elif result.coding_result and result.coding_result.error:
        console.print()
        print_warning(f"Coding failed: {result.coding_result.error}")

    # MDR determination
    if result.mdr_determination:
        console.print()
        mdr = result.mdr_determination

        mdr_color = "red" if mdr.requires_mdr else "green"
        mdr_status = "REQUIRED" if mdr.requires_mdr else "Not Required"

        table = Table(title="MDR Determination", show_header=False)
        table.add_column("Field", style="bold")
        table.add_column("Value")

        table.add_row("MDR Required", f"[{mdr_color}]{mdr_status}[/{mdr_color}]")
        table.add_row("Confidence", str(format_confidence(mdr.ai_confidence)))
        table.add_row("Priority", mdr.review_priority.upper())

        if mdr.mdr_criteria_met:
            criteria_str = ", ".join(c.value for c in mdr.mdr_criteria_met)
            table.add_row("Criteria Met", criteria_str)

        console.print(table)

        if verbose and mdr.key_factors:
            console.print()
            console.print("[bold cyan]Key Factors:[/bold cyan]")
            for factor in mdr.key_factors:
                console.print(f"  - {factor}")

    # Audit events
    if result.audit_event_ids and verbose:
        console.print()
        console.print("[bold cyan]Audit Events Logged:[/bold cyan]")
        for event_id in result.audit_event_ids:
            console.print(f"  - {event_id}")

    # Errors
    if result.errors:
        console.print()
        console.print("[bold red]Errors:[/bold red]")
        for error in result.errors:
            console.print(f"  [{error['step']}] {error['message']}")

    # Processing metrics
    if result.processing_duration_ms:
        console.print()
        console.print(
            f"[dim]Total processing time: {result.processing_duration_ms:.0f}ms[/dim]"
        )

    console.print()


@app.command("form")
def process_form_command(
    file_path: Annotated[
        Path,
        typer.Argument(
            help="Path to JSON file containing form submission data",
            exists=True,
            readable=True,
        ),
    ],
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed output including metrics and audit events",
        ),
    ] = False,
    no_save: Annotated[
        bool,
        typer.Option(
            "--no-save",
            help="Don't save output files (complaint, decision)",
        ),
    ] = False,
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output result as JSON instead of formatted display",
        ),
    ] = False,
    complaints_dir: Annotated[
        Path | None,
        typer.Option(
            "--complaints-dir",
            help="Directory for complaint output files",
        ),
    ] = None,
    decisions_dir: Annotated[
        Path | None,
        typer.Option(
            "--decisions-dir",
            help="Directory for decision output files",
        ),
    ] = None,
) -> None:
    """Process a form submission through the complete pipeline.

    Reads a form submission from a JSON file and processes it through:
    1. Form parsing and validation
    2. Conversion to complaint record
    3. IMDRF code suggestion (using LLM)
    4. MDR determination
    5. Audit logging

    The input file can be either a raw form submission or a test case
    file (with raw_input field).

    Example:
        complaints process form data/test_cases/form/form_001_pacemaker_death.json
    """
    try:
        result = process_form_file(
            file_path=file_path,
            complaints_dir=complaints_dir,
            decisions_dir=decisions_dir,
            save_outputs=not no_save,
        )

        if output_json:
            # Output as JSON
            output_data = result.summary()
            if verbose:
                output_data["errors"] = result.errors
                output_data["audit_event_ids"] = result.audit_event_ids
            console.print_json(json.dumps(output_data, default=str))
        else:
            # Formatted display
            _display_processing_result(result, verbose=verbose)

            # Final status message
            if result.status == ProcessingStatus.SUCCESS:
                print_success(
                    f"Form processed successfully. Complaint ID: {result.complaint_id}"
                )
            elif result.status == ProcessingStatus.PARTIAL:
                print_warning(
                    f"Form partially processed. Complaint ID: {result.complaint_id}"
                )
            else:
                print_error("Form processing failed. See errors above.")
                raise typer.Exit(code=1)

    except FileNotFoundError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from None
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON file: {e}")
        raise typer.Exit(code=1) from None
    except ValueError as e:
        print_error(f"Configuration error: {e}")
        print_info(
            "Ensure Azure OpenAI environment variables are set "
            "(AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME)"
        )
        raise typer.Exit(code=1) from None


@app.command("batch")
def process_batch_command(
    directory: Annotated[
        Path,
        typer.Argument(
            help="Directory containing form submission JSON files",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    pattern: Annotated[
        str,
        typer.Option(
            "--pattern",
            "-p",
            help="Glob pattern for matching files",
        ),
    ] = "*.json",
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed output for each file",
        ),
    ] = False,
    no_save: Annotated[
        bool,
        typer.Option(
            "--no-save",
            help="Don't save output files",
        ),
    ] = False,
) -> None:
    """Process multiple form submissions from a directory.

    Processes all JSON files matching the pattern in the specified directory.
    Useful for batch processing test cases or bulk imports.

    Example:
        complaints process batch data/test_cases/form/ --pattern "form_*.json"
    """
    files = sorted(directory.glob(pattern))

    # Exclude manifest files
    files = [f for f in files if not f.name.startswith("_")]

    if not files:
        print_warning(f"No files matching '{pattern}' found in {directory}")
        raise typer.Exit(code=0)

    console.print(f"\n[bold]Processing {len(files)} file(s)...[/bold]\n")

    results: list[tuple[Path, ProcessingResult]] = []
    success_count = 0
    partial_count = 0
    failed_count = 0

    for file_path in files:
        console.print(f"[dim]Processing: {file_path.name}[/dim]")

        try:
            result = process_form_file(
                file_path=file_path,
                save_outputs=not no_save,
            )
            results.append((file_path, result))

            if result.status == ProcessingStatus.SUCCESS:
                success_count += 1
                if verbose:
                    _display_processing_result(result, verbose=False)
            elif result.status == ProcessingStatus.PARTIAL:
                partial_count += 1
                if verbose:
                    _display_processing_result(result, verbose=False)
            else:
                failed_count += 1
                if verbose:
                    _display_processing_result(result, verbose=False)

        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            failed_count += 1
            print_error(f"Failed to process {file_path.name}: {e}")

    # Summary
    console.print()
    console.print("[bold]Batch Processing Summary[/bold]")

    summary_table = Table(show_header=False)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Count", justify="right")

    summary_table.add_row("Total files", str(len(files)))
    summary_table.add_row("[green]Successful[/green]", str(success_count))
    summary_table.add_row("[yellow]Partial[/yellow]", str(partial_count))
    summary_table.add_row("[red]Failed[/red]", str(failed_count))

    console.print(summary_table)
    console.print()

    if failed_count > 0:
        raise typer.Exit(code=1)


# Main entry point for direct execution
if __name__ == "__main__":
    app()
