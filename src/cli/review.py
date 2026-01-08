"""CLI commands for reviewing complaint coding suggestions.

This module provides a Typer-based CLI for reviewing AI-generated IMDRF code
suggestions and recording human decisions for the audit trail.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from src.audit.logger import AuditLogger, generate_event_id
from src.audit.models import CodingReviewedEvent
from src.cli.display import (
    console,
    create_complaints_table,
    create_suggestions_table,
    display_complaint_summary,
    display_review_result,
    print_error,
    print_info,
    print_success,
    print_warning,
    prompt_additional_codes,
    prompt_code_selection,
    prompt_confirmation,
    prompt_review_notes,
)
from src.imdrf import validate_code
from src.models.coding import CodingDecision
from src.models.complaint import ComplaintRecord
from src.models.enums import ComplaintStatus

logger = logging.getLogger(__name__)

# Create Typer app for review commands
app = typer.Typer(
    name="review",
    help="Review AI-generated IMDRF code suggestions for complaints.",
    no_args_is_help=True,
)

# Default paths
DEFAULT_COMPLAINTS_DIR = Path("data/complaints")
DEFAULT_DECISIONS_DIR = Path("data/decisions")
DEFAULT_AUDIT_DIR = Path("audit_logs")


def _ensure_directories() -> None:
    """Ensure required data directories exist."""
    DEFAULT_COMPLAINTS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_DECISIONS_DIR.mkdir(parents=True, exist_ok=True)


def _load_complaint(complaint_id: str) -> ComplaintRecord | None:
    """Load a complaint from disk.

    Args:
        complaint_id: Complaint identifier.

    Returns:
        ComplaintRecord if found, None otherwise.
    """
    complaint_file = DEFAULT_COMPLAINTS_DIR / f"{complaint_id}.json"
    if not complaint_file.exists():
        return None

    try:
        with open(complaint_file) as f:
            data = json.load(f)
        return ComplaintRecord.model_validate(data)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("Failed to load complaint %s: %s", complaint_id, e)
        return None


def _save_complaint(complaint: ComplaintRecord) -> None:
    """Save a complaint to disk.

    Args:
        complaint: ComplaintRecord to save.
    """
    _ensure_directories()
    complaint_file = DEFAULT_COMPLAINTS_DIR / f"{complaint.complaint_id}.json"
    with open(complaint_file, "w") as f:
        json.dump(complaint.model_dump(mode="json"), f, indent=2, default=str)


def _load_decision(complaint_id: str) -> CodingDecision | None:
    """Load a coding decision from disk.

    Args:
        complaint_id: Complaint identifier.

    Returns:
        CodingDecision if found, None otherwise.
    """
    decision_file = DEFAULT_DECISIONS_DIR / f"{complaint_id}_decision.json"
    if not decision_file.exists():
        return None

    try:
        with open(decision_file) as f:
            data = json.load(f)
        return CodingDecision.model_validate(data)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("Failed to load decision %s: %s", complaint_id, e)
        return None


def _save_decision(decision: CodingDecision) -> None:
    """Save a coding decision to disk.

    Args:
        decision: CodingDecision to save.
    """
    _ensure_directories()
    decision_file = DEFAULT_DECISIONS_DIR / f"{decision.complaint_id}_decision.json"
    with open(decision_file, "w") as f:
        json.dump(decision.model_dump(mode="json"), f, indent=2, default=str)


def _list_all_complaints() -> list[ComplaintRecord]:
    """List all complaints from disk.

    Returns:
        List of ComplaintRecord objects.
    """
    _ensure_directories()
    complaints: list[ComplaintRecord] = []

    for complaint_file in DEFAULT_COMPLAINTS_DIR.glob("*.json"):
        try:
            with open(complaint_file) as f:
                data = json.load(f)
            complaints.append(ComplaintRecord.model_validate(data))
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Skipping invalid complaint file %s: %s", complaint_file, e)
            continue

    return complaints


def _get_pending_complaints() -> list[ComplaintRecord]:
    """Get complaints that need review.

    Returns:
        List of complaints with status CODED (ready for review).
    """
    all_complaints = _list_all_complaints()
    return [c for c in all_complaints if c.status == ComplaintStatus.CODED]


@app.command("list")
def list_complaints(
    status: Annotated[
        str | None,
        typer.Option(
            "--status",
            "-s",
            help="Filter by status (new, extracted, coded, reviewed, closed)",
        ),
    ] = None,
    pending: Annotated[
        bool,
        typer.Option(
            "--pending",
            "-p",
            help="Show only complaints pending review (status=coded)",
        ),
    ] = False,
) -> None:
    """List complaints available for review.

    Shows complaints that have been processed and are ready for human review.
    """
    if pending:
        complaints = _get_pending_complaints()
        if not complaints:
            print_info("No complaints pending review.")
            return
    else:
        complaints = _list_all_complaints()
        if status:
            try:
                status_filter = ComplaintStatus(status.lower())
                complaints = [c for c in complaints if c.status == status_filter]
            except ValueError:
                print_error(
                    f"Invalid status '{status}'. "
                    f"Valid values: {', '.join(s.value for s in ComplaintStatus)}"
                )
                raise typer.Exit(code=1) from None

    if not complaints:
        print_info("No complaints found matching the criteria.")
        return

    # Sort by received date (most recent first)
    complaints.sort(key=lambda c: c.received_date, reverse=True)

    console.print()
    console.print(create_complaints_table(complaints))
    console.print()
    console.print(f"Total: {len(complaints)} complaint(s)")


@app.command("show")
def show_complaint(
    complaint_id: Annotated[
        str,
        typer.Argument(help="Complaint ID to display"),
    ],
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed suggestion information",
        ),
    ] = False,
) -> None:
    """Display a complaint with AI-generated coding suggestions.

    Shows the complaint details, narrative, and all coding suggestions
    from the AI system with confidence scores and reasoning.
    """
    complaint = _load_complaint(complaint_id)
    if complaint is None:
        print_error(f"Complaint '{complaint_id}' not found.")
        raise typer.Exit(code=1)

    # Load existing decision if available
    decision = _load_decision(complaint_id)
    suggestions = decision.suggested_codes if decision else []

    display_complaint_summary(complaint, suggestions if verbose else None)

    if suggestions and not verbose:
        console.print()
        console.print(create_suggestions_table(suggestions))

    if decision and decision.is_reviewed:
        console.print()
        print_info(
            f"This complaint has been reviewed by {decision.reviewer_id} "
            f"on {decision.review_timestamp}"
        )
        console.print(f"Final codes: {', '.join(decision.final_codes)}")


@app.command("approve")
def approve_suggestions(
    complaint_id: Annotated[
        str,
        typer.Argument(help="Complaint ID to approve"),
    ],
    reviewer: Annotated[
        str,
        typer.Option(
            "--reviewer",
            "-r",
            help="Reviewer ID/name",
        ),
    ] = "cli_user",
    all_codes: Annotated[
        bool,
        typer.Option(
            "--all",
            "-a",
            help="Approve all suggested codes without prompting",
        ),
    ] = False,
    audit_dir: Annotated[
        Path,
        typer.Option(
            "--audit-dir",
            help="Directory for audit logs",
        ),
    ] = DEFAULT_AUDIT_DIR,
) -> None:
    """Approve AI-suggested IMDRF codes for a complaint.

    Approves all suggested codes and records the decision in the audit log.
    Use --all to approve without confirmation, or interactively select codes.
    """
    complaint = _load_complaint(complaint_id)
    if complaint is None:
        print_error(f"Complaint '{complaint_id}' not found.")
        raise typer.Exit(code=1)

    decision = _load_decision(complaint_id)
    if decision is None:
        print_error(f"No coding suggestions found for '{complaint_id}'.")
        raise typer.Exit(code=1)

    if decision.is_reviewed:
        print_warning(
            f"Complaint '{complaint_id}' has already been reviewed. "
            "Use 'modify' to change the decision."
        )
        raise typer.Exit(code=1)

    suggestions = decision.suggested_codes
    if not suggestions:
        print_warning("No suggestions to approve.")
        raise typer.Exit(code=1)

    # Display suggestions
    console.print()
    console.print(create_suggestions_table(suggestions))

    # Determine which codes to approve
    if all_codes:
        approved_indices = list(range(1, len(suggestions) + 1))
    else:
        approved_indices = prompt_code_selection(suggestions)

    # Get approved code IDs
    approved_codes = [
        suggestions[i - 1].code_id
        for i in approved_indices
        if 1 <= i <= len(suggestions)
    ]
    rejected_codes = [
        s.code_id for s in suggestions if s.code_id not in approved_codes
    ]

    # Confirm
    if not all_codes:
        console.print()
        console.print(f"Approving codes: {', '.join(approved_codes) or 'none'}")
        console.print(f"Rejecting codes: {', '.join(rejected_codes) or 'none'}")
        if not prompt_confirmation("Confirm this review?"):
            print_info("Review cancelled.")
            raise typer.Exit(code=0)

    # Record the decision
    review_timestamp = datetime.now(UTC)
    decision.approved_codes = approved_codes
    decision.rejected_codes = rejected_codes
    decision.reviewer_id = reviewer
    decision.review_timestamp = review_timestamp

    # Save decision
    _save_decision(decision)

    # Update complaint status
    complaint.status = ComplaintStatus.REVIEWED
    complaint.updated_at = review_timestamp
    _save_complaint(complaint)

    # Log to audit trail
    audit_logger = AuditLogger(log_dir=audit_dir)
    audit_event = CodingReviewedEvent(
        event_id=generate_event_id(),
        resource_id=complaint_id,
        user_id=reviewer,
        user_name=reviewer,
        approved_codes=approved_codes,
        rejected_codes=rejected_codes,
        added_codes=[],
    )
    audit_logger.log_event(audit_event)

    display_review_result(approved_codes, rejected_codes, [])
    print_success(f"Review recorded for complaint '{complaint_id}'.")


@app.command("modify")
def modify_codes(
    complaint_id: Annotated[
        str,
        typer.Argument(help="Complaint ID to modify"),
    ],
    reviewer: Annotated[
        str,
        typer.Option(
            "--reviewer",
            "-r",
            help="Reviewer ID/name",
        ),
    ] = "cli_user",
    audit_dir: Annotated[
        Path,
        typer.Option(
            "--audit-dir",
            help="Directory for audit logs",
        ),
    ] = DEFAULT_AUDIT_DIR,
) -> None:
    """Modify IMDRF codes for a complaint before approval.

    Allows selecting which suggested codes to approve, which to reject,
    and adding new codes manually. All decisions are recorded in the audit log.
    """
    complaint = _load_complaint(complaint_id)
    if complaint is None:
        print_error(f"Complaint '{complaint_id}' not found.")
        raise typer.Exit(code=1)

    decision = _load_decision(complaint_id)
    if decision is None:
        print_error(f"No coding suggestions found for '{complaint_id}'.")
        raise typer.Exit(code=1)

    suggestions = decision.suggested_codes

    # Display complaint and suggestions
    console.print()
    display_complaint_summary(complaint, suggestions)

    # Select codes to approve
    if suggestions:
        console.print()
        console.print(create_suggestions_table(suggestions))
        approved_indices = prompt_code_selection(suggestions)
        approved_codes = [
            suggestions[i - 1].code_id
            for i in approved_indices
            if 1 <= i <= len(suggestions)
        ]
        rejected_codes = [
            s.code_id for s in suggestions if s.code_id not in approved_codes
        ]
    else:
        approved_codes = []
        rejected_codes = []

    # Prompt for additional codes
    additional_codes = prompt_additional_codes()

    # Validate additional codes
    valid_additional: list[str] = []
    for code_id in additional_codes:
        if validate_code(code_id):
            valid_additional.append(code_id)
        else:
            print_warning(f"Invalid IMDRF code: {code_id} (skipped)")

    # Get review notes
    notes = prompt_review_notes()

    # Show summary and confirm
    console.print()
    console.print("[bold]Review Summary:[/bold]")
    console.print(f"  Approved: {', '.join(approved_codes) or 'none'}")
    console.print(f"  Rejected: {', '.join(rejected_codes) or 'none'}")
    console.print(f"  Added: {', '.join(valid_additional) or 'none'}")
    if notes:
        console.print(f"  Notes: {notes}")

    if not prompt_confirmation("\nConfirm this review?"):
        print_info("Review cancelled.")
        raise typer.Exit(code=0)

    # Record the decision
    review_timestamp = datetime.now(UTC)
    decision.approved_codes = approved_codes
    decision.rejected_codes = rejected_codes
    decision.added_codes = valid_additional
    decision.reviewer_id = reviewer
    decision.review_timestamp = review_timestamp
    decision.review_notes = notes

    # Save decision
    _save_decision(decision)

    # Update complaint status
    complaint.status = ComplaintStatus.REVIEWED
    complaint.updated_at = review_timestamp
    _save_complaint(complaint)

    # Log to audit trail
    audit_logger = AuditLogger(log_dir=audit_dir)
    audit_event = CodingReviewedEvent(
        event_id=generate_event_id(),
        resource_id=complaint_id,
        user_id=reviewer,
        user_name=reviewer,
        approved_codes=approved_codes,
        rejected_codes=rejected_codes,
        added_codes=valid_additional,
        review_notes=notes,
    )
    audit_logger.log_event(audit_event)

    display_review_result(approved_codes, rejected_codes, valid_additional)
    print_success(f"Modified review recorded for complaint '{complaint_id}'.")


@app.command("reject")
def reject_suggestions(
    complaint_id: Annotated[
        str,
        typer.Argument(help="Complaint ID to reject"),
    ],
    reviewer: Annotated[
        str,
        typer.Option(
            "--reviewer",
            "-r",
            help="Reviewer ID/name",
        ),
    ] = "cli_user",
    reason: Annotated[
        str | None,
        typer.Option(
            "--reason",
            help="Reason for rejection",
        ),
    ] = None,
    audit_dir: Annotated[
        Path,
        typer.Option(
            "--audit-dir",
            help="Directory for audit logs",
        ),
    ] = DEFAULT_AUDIT_DIR,
) -> None:
    """Reject all AI-suggested codes and re-queue for manual coding.

    Marks all suggested codes as rejected and resets the complaint status
    to EXTRACTED for manual coding by a human specialist.
    """
    complaint = _load_complaint(complaint_id)
    if complaint is None:
        print_error(f"Complaint '{complaint_id}' not found.")
        raise typer.Exit(code=1)

    decision = _load_decision(complaint_id)
    if decision is None:
        print_error(f"No coding suggestions found for '{complaint_id}'.")
        raise typer.Exit(code=1)

    suggestions = decision.suggested_codes

    # Display what will be rejected
    if suggestions:
        console.print()
        console.print(create_suggestions_table(suggestions))
        console.print()
        console.print(
            f"[bold red]All {len(suggestions)} suggestion(s) will be rejected.[/bold red]"
        )

    # Get reason if not provided
    if reason is None:
        console.print("\n[bold]Enter reason for rejection:[/bold]")
        reason = console.input("> ").strip()

    if not prompt_confirmation("Confirm rejection?"):
        print_info("Rejection cancelled.")
        raise typer.Exit(code=0)

    # Record the decision
    review_timestamp = datetime.now(UTC)
    rejected_codes = [s.code_id for s in suggestions]
    decision.approved_codes = []
    decision.rejected_codes = rejected_codes
    decision.added_codes = []
    decision.reviewer_id = reviewer
    decision.review_timestamp = review_timestamp
    decision.review_notes = f"REJECTED: {reason}" if reason else "REJECTED"

    # Save decision
    _save_decision(decision)

    # Reset complaint status to EXTRACTED for manual coding
    complaint.status = ComplaintStatus.EXTRACTED
    complaint.updated_at = review_timestamp
    _save_complaint(complaint)

    # Log to audit trail
    audit_logger = AuditLogger(log_dir=audit_dir)
    audit_event = CodingReviewedEvent(
        event_id=generate_event_id(),
        resource_id=complaint_id,
        user_id=reviewer,
        user_name=reviewer,
        approved_codes=[],
        rejected_codes=rejected_codes,
        added_codes=[],
        review_notes=f"REJECTED: {reason}" if reason else "REJECTED",
    )
    audit_logger.log_event(audit_event)

    print_warning(f"All suggestions rejected for complaint '{complaint_id}'.")
    print_info("Complaint has been re-queued for manual coding (status: EXTRACTED).")


@app.command("history")
def show_history(
    complaint_id: Annotated[
        str,
        typer.Argument(help="Complaint ID to show history for"),
    ],
    audit_dir: Annotated[
        Path,
        typer.Option(
            "--audit-dir",
            help="Directory for audit logs",
        ),
    ] = DEFAULT_AUDIT_DIR,
) -> None:
    """Show the audit history for a complaint.

    Displays all audit events related to the complaint including
    creation, coding suggestions, and review decisions.
    """
    audit_logger = AuditLogger(log_dir=audit_dir)
    events = audit_logger.get_events(complaint_id)

    if not events:
        print_info(f"No audit history found for complaint '{complaint_id}'.")
        return

    console.print()
    console.print(f"[bold]Audit History for {complaint_id}[/bold]")
    console.print()

    for event in events:
        timestamp_str = event.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        console.print(f"[dim]{timestamp_str}[/dim]")
        console.print(f"  Action: [cyan]{event.action.value}[/cyan]")
        console.print(f"  User: {event.user_name}")

        if event.details:
            for key, value in event.details.items():
                if isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value) if value else "-"
                else:
                    value_str = str(value)
                console.print(f"  {key}: {value_str}")

        console.print()


# Main entry point for direct execution
if __name__ == "__main__":
    app()
