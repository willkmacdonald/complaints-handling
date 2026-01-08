"""Rich display utilities for CLI output."""

from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.models.coding import CodingDecision, CodingSuggestion
from src.models.complaint import ComplaintRecord
from src.models.enums import ComplaintStatus

console = Console()


def format_confidence(confidence: float) -> Text:
    """Format a confidence score with color coding.

    Args:
        confidence: Confidence score (0.0-1.0).

    Returns:
        Colored text representation.
    """
    percentage = int(confidence * 100)
    if confidence >= 0.8:
        style = "green"
    elif confidence >= 0.5:
        style = "yellow"
    else:
        style = "red"
    return Text(f"{percentage}%", style=style)


def format_status(status: ComplaintStatus) -> Text:
    """Format a complaint status with color coding.

    Args:
        status: Complaint status.

    Returns:
        Colored text representation.
    """
    style_map = {
        ComplaintStatus.NEW: "blue",
        ComplaintStatus.EXTRACTED: "cyan",
        ComplaintStatus.CODED: "yellow",
        ComplaintStatus.REVIEWED: "green",
        ComplaintStatus.CLOSED: "dim",
    }
    return Text(status.value.upper(), style=style_map.get(status, "white"))


def format_datetime(dt: datetime | None) -> str:
    """Format a datetime for display.

    Args:
        dt: Datetime to format.

    Returns:
        Formatted string or 'N/A'.
    """
    if dt is None:
        return "N/A"
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def create_complaints_table(complaints: list[ComplaintRecord]) -> Table:
    """Create a table listing complaints for review.

    Args:
        complaints: List of complaint records.

    Returns:
        Rich Table object.
    """
    table = Table(title="Complaints Pending Review", show_header=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Device", style="white")
    table.add_column("Manufacturer", style="white")
    table.add_column("Status", justify="center")
    table.add_column("Received", style="dim")

    for complaint in complaints:
        table.add_row(
            complaint.complaint_id,
            complaint.device_info.device_name[:30] + "..."
            if len(complaint.device_info.device_name) > 30
            else complaint.device_info.device_name,
            complaint.device_info.manufacturer[:20] + "..."
            if len(complaint.device_info.manufacturer) > 20
            else complaint.device_info.manufacturer,
            format_status(complaint.status),
            format_datetime(complaint.received_date),
        )

    return table


def create_complaint_panel(complaint: ComplaintRecord) -> Panel:
    """Create a panel displaying complaint details.

    Args:
        complaint: Complaint record to display.

    Returns:
        Rich Panel object.
    """
    lines = [
        f"[bold]Complaint ID:[/bold] {complaint.complaint_id}",
        f"[bold]Status:[/bold] {format_status(complaint.status)}",
        f"[bold]Channel:[/bold] {complaint.intake_channel.value}",
        f"[bold]Received:[/bold] {format_datetime(complaint.received_date)}",
        "",
        "[bold cyan]Device Information[/bold cyan]",
        f"  Name: {complaint.device_info.device_name}",
        f"  Manufacturer: {complaint.device_info.manufacturer}",
        f"  Model: {complaint.device_info.model_number or 'N/A'}",
        f"  Serial: {complaint.device_info.serial_number or 'N/A'}",
        f"  Lot: {complaint.device_info.lot_number or 'N/A'}",
        "",
        "[bold cyan]Event Information[/bold cyan]",
        f"  Date: {complaint.event_info.event_date or 'N/A'}",
        f"  Patient Outcome: {complaint.event_info.patient_outcome or 'N/A'}",
        f"  Device Outcome: {complaint.event_info.device_outcome or 'N/A'}",
    ]

    if complaint.patient_info:
        lines.extend([
            "",
            "[bold cyan]Patient Information[/bold cyan]",
            f"  Age: {complaint.patient_info.age or 'N/A'}",
            f"  Sex: {complaint.patient_info.sex or 'N/A'}",
        ])

    return Panel(
        "\n".join(lines),
        title=f"[bold]Complaint: {complaint.complaint_id}[/bold]",
        border_style="blue",
    )


def create_narrative_panel(narrative: str, max_length: int = 500) -> Panel:
    """Create a panel displaying the complaint narrative.

    Args:
        narrative: Narrative text.
        max_length: Maximum characters to display.

    Returns:
        Rich Panel object.
    """
    display_text = narrative
    if len(narrative) > max_length:
        display_text = narrative[:max_length] + "..."

    return Panel(
        display_text,
        title="[bold]Complaint Narrative[/bold]",
        border_style="cyan",
    )


def create_suggestions_table(suggestions: list[CodingSuggestion]) -> Table:
    """Create a table displaying coding suggestions.

    Args:
        suggestions: List of coding suggestions.

    Returns:
        Rich Table object.
    """
    table = Table(title="AI Coding Suggestions", show_header=True)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Code", style="cyan", no_wrap=True)
    table.add_column("Name", style="white")
    table.add_column("Type", style="magenta")
    table.add_column("Confidence", justify="center")

    for idx, suggestion in enumerate(suggestions, 1):
        table.add_row(
            str(idx),
            suggestion.code_id,
            suggestion.code_name[:40] + "..."
            if len(suggestion.code_name) > 40
            else suggestion.code_name,
            suggestion.code_type.value.replace("_", " ").title(),
            format_confidence(suggestion.confidence),
        )

    return table


def create_suggestion_detail_panel(suggestion: CodingSuggestion, index: int) -> Panel:
    """Create a detailed panel for a single suggestion.

    Args:
        suggestion: Coding suggestion.
        index: Suggestion index (1-based).

    Returns:
        Rich Panel object.
    """
    lines = [
        f"[bold]Code:[/bold] {suggestion.code_id}",
        f"[bold]Name:[/bold] {suggestion.code_name}",
        f"[bold]Type:[/bold] {suggestion.code_type.value.replace('_', ' ').title()}",
        f"[bold]Confidence:[/bold] {format_confidence(suggestion.confidence)}",
        "",
        "[bold]Full Path:[/bold]",
        f"  {suggestion.full_path or 'N/A'}",
        "",
        "[bold cyan]Source Text:[/bold cyan]",
        f"  [italic]{suggestion.source_text}[/italic]",
        "",
        "[bold cyan]Reasoning:[/bold cyan]",
        f"  {suggestion.reasoning}",
    ]

    return Panel(
        "\n".join(lines),
        title=f"[bold]Suggestion #{index}[/bold]",
        border_style="yellow",
    )


def create_coding_decision_table(decision: CodingDecision) -> Table:
    """Create a table showing coding review status.

    Args:
        decision: Coding decision record.

    Returns:
        Rich Table object.
    """
    table = Table(title="Coding Review Status", show_header=True)
    table.add_column("Category", style="bold")
    table.add_column("Codes", style="white")

    approved = ", ".join(decision.approved_codes) if decision.approved_codes else "-"
    rejected = ", ".join(decision.rejected_codes) if decision.rejected_codes else "-"
    added = ", ".join(decision.added_codes) if decision.added_codes else "-"
    final = ", ".join(decision.final_codes) if decision.final_codes else "-"

    table.add_row("[green]Approved[/green]", approved)
    table.add_row("[red]Rejected[/red]", rejected)
    table.add_row("[blue]Manually Added[/blue]", added)
    table.add_row("[bold]Final Codes[/bold]", final)

    return table


def display_complaint_summary(
    complaint: ComplaintRecord,
    suggestions: list[CodingSuggestion] | None = None,
) -> None:
    """Display a complete summary of a complaint with suggestions.

    Args:
        complaint: Complaint record to display.
        suggestions: Optional list of coding suggestions.
    """
    console.print()
    console.print(create_complaint_panel(complaint))
    console.print()
    console.print(create_narrative_panel(complaint.narrative))

    if suggestions:
        console.print()
        console.print(create_suggestions_table(suggestions))
        console.print()
        for idx, suggestion in enumerate(suggestions, 1):
            console.print(create_suggestion_detail_panel(suggestion, idx))
            console.print()


def display_review_result(
    approved: list[str],
    rejected: list[str],
    added: list[str],
) -> None:
    """Display the results of a review decision.

    Args:
        approved: List of approved code IDs.
        rejected: List of rejected code IDs.
        added: List of manually added code IDs.
    """
    console.print()
    console.print("[bold green]Review Decision Recorded[/bold green]")
    console.print()

    if approved:
        console.print(f"  [green]Approved:[/green] {', '.join(approved)}")
    if rejected:
        console.print(f"  [red]Rejected:[/red] {', '.join(rejected)}")
    if added:
        console.print(f"  [blue]Manually Added:[/blue] {', '.join(added)}")

    final_codes = list(set(approved + added))
    if final_codes:
        console.print(f"\n  [bold]Final Codes:[/bold] {', '.join(sorted(final_codes))}")

    console.print()


def print_error(message: str) -> None:
    """Print an error message.

    Args:
        message: Error message to display.
    """
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_success(message: str) -> None:
    """Print a success message.

    Args:
        message: Success message to display.
    """
    console.print(f"[bold green]Success:[/bold green] {message}")


def print_warning(message: str) -> None:
    """Print a warning message.

    Args:
        message: Warning message to display.
    """
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message.

    Args:
        message: Info message to display.
    """
    console.print(f"[bold blue]Info:[/bold blue] {message}")


def prompt_confirmation(message: str) -> bool:
    """Prompt the user for yes/no confirmation.

    Args:
        message: Prompt message.

    Returns:
        True if confirmed, False otherwise.
    """
    response = console.input(f"{message} [y/N]: ").strip().lower()
    return response in ("y", "yes")


def prompt_code_selection(suggestions: list[CodingSuggestion]) -> list[int]:
    """Prompt the user to select codes from suggestions.

    Args:
        suggestions: List of suggestions to choose from.

    Returns:
        List of selected indices (1-based).
    """
    console.print(
        "\n[bold]Enter suggestion numbers to approve (comma-separated), "
        "'all' for all, or 'none':[/bold]"
    )
    response = console.input("> ").strip().lower()

    if response == "none":
        return []
    if response == "all":
        return list(range(1, len(suggestions) + 1))

    try:
        selected = [int(x.strip()) for x in response.split(",") if x.strip()]
        # Validate indices
        valid = [i for i in selected if 1 <= i <= len(suggestions)]
        return valid
    except ValueError:
        print_error("Invalid input. Please enter numbers separated by commas.")
        return []


def prompt_additional_codes() -> list[str]:
    """Prompt the user to enter additional IMDRF codes.

    Returns:
        List of code IDs entered.
    """
    console.print(
        "\n[bold]Enter additional IMDRF codes to add (comma-separated), "
        "or press Enter to skip:[/bold]"
    )
    response = console.input("> ").strip()

    if not response:
        return []

    return [code.strip().upper() for code in response.split(",") if code.strip()]


def prompt_review_notes() -> str | None:
    """Prompt the user for optional review notes.

    Returns:
        Review notes string or None.
    """
    console.print("\n[bold]Enter review notes (optional, press Enter to skip):[/bold]")
    response = console.input("> ").strip()
    return response if response else None
