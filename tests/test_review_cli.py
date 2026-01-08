"""Tests for review CLI commands."""

from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from src.audit.logger import AuditLogger
from src.audit.models import AuditAction
from src.cli.display import (
    create_complaints_table,
    create_suggestion_detail_panel,
    create_suggestions_table,
    format_confidence,
    format_status,
)
from src.cli.review import (
    _load_complaint,
    _load_decision,
    _save_complaint,
    _save_decision,
    app,
)
from src.models.coding import CodingDecision, CodingSuggestion
from src.models.complaint import ComplaintRecord, DeviceInfo, EventInfo
from src.models.enums import ComplaintStatus, DeviceType, IMDRFCodeType, IntakeChannel

runner = CliRunner()


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temporary data directories."""
    complaints_dir = tmp_path / "complaints"
    decisions_dir = tmp_path / "decisions"
    audit_dir = tmp_path / "audit_logs"
    complaints_dir.mkdir()
    decisions_dir.mkdir()
    audit_dir.mkdir()
    return tmp_path


@pytest.fixture
def sample_complaint() -> ComplaintRecord:
    """Create a sample complaint record."""
    return ComplaintRecord(
        complaint_id="TEST-001",
        intake_channel=IntakeChannel.FORM,
        received_date=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
        status=ComplaintStatus.CODED,
        device_info=DeviceInfo(
            device_name="Test Pacemaker Model X",
            manufacturer="Test Medical Inc.",
            model_number="PM-100",
            serial_number="SN-12345",
            lot_number="LOT-2024-001",
            device_type=DeviceType.IMPLANTABLE,
        ),
        event_info=EventInfo(
            event_date=date(2024, 1, 10),
            event_description="Device battery depleted prematurely.",
            patient_outcome="Hospitalization for device replacement",
            device_outcome="Returned for evaluation",
        ),
        narrative="Patient reported sudden low battery warning. "
        "Device was replaced emergently. Battery found to be depleted "
        "after only 2 years of expected 10-year life.",
    )


@pytest.fixture
def sample_suggestions() -> list[CodingSuggestion]:
    """Create sample coding suggestions."""
    return [
        CodingSuggestion(
            code_id="A0601",
            code_name="Battery Problem",
            code_type=IMDRFCodeType.DEVICE_PROBLEM,
            confidence=0.95,
            source_text="battery depleted prematurely",
            reasoning="Narrative explicitly mentions battery depletion.",
            full_path="Energy Source > Battery Problem",
        ),
        CodingSuggestion(
            code_id="A0101",
            code_name="Failure to Function",
            code_type=IMDRFCodeType.DEVICE_PROBLEM,
            confidence=0.7,
            source_text="Device was replaced emergently",
            reasoning="Device required replacement indicating failure.",
            full_path="Failure to Function",
        ),
    ]


@pytest.fixture
def sample_decision(sample_suggestions: list[CodingSuggestion]) -> CodingDecision:
    """Create a sample coding decision."""
    return CodingDecision(
        complaint_id="TEST-001",
        suggested_codes=sample_suggestions,
        suggestion_timestamp=datetime(2024, 1, 15, 11, 0, 0, tzinfo=UTC),
    )


class TestDisplayUtilities:
    """Tests for display utility functions."""

    def test_format_confidence_high(self) -> None:
        """High confidence is formatted in green."""
        result = format_confidence(0.95)
        assert "95%" in str(result)
        assert result.style == "green"

    def test_format_confidence_medium(self) -> None:
        """Medium confidence is formatted in yellow."""
        result = format_confidence(0.6)
        assert "60%" in str(result)
        assert result.style == "yellow"

    def test_format_confidence_low(self) -> None:
        """Low confidence is formatted in red."""
        result = format_confidence(0.3)
        assert "30%" in str(result)
        assert result.style == "red"

    def test_format_status(self) -> None:
        """Status values are formatted correctly."""
        result = format_status(ComplaintStatus.CODED)
        assert "CODED" in str(result)

    def test_create_complaints_table(self, sample_complaint: ComplaintRecord) -> None:
        """Complaints table is created correctly."""
        table = create_complaints_table([sample_complaint])
        assert table.title == "Complaints Pending Review"
        assert table.row_count == 1

    def test_create_suggestions_table(
        self, sample_suggestions: list[CodingSuggestion]
    ) -> None:
        """Suggestions table is created correctly."""
        table = create_suggestions_table(sample_suggestions)
        assert table.title == "AI Coding Suggestions"
        assert table.row_count == 2

    def test_create_suggestion_detail_panel(
        self, sample_suggestions: list[CodingSuggestion]
    ) -> None:
        """Suggestion detail panel is created correctly."""
        panel = create_suggestion_detail_panel(sample_suggestions[0], 1)
        assert "Suggestion #1" in str(panel.title)


class TestDataPersistence:
    """Tests for complaint and decision persistence."""

    def test_save_and_load_complaint(
        self, temp_data_dir: Path, sample_complaint: ComplaintRecord
    ) -> None:
        """Complaints can be saved and loaded."""
        import src.cli.review as review_module

        original_dir = review_module.DEFAULT_COMPLAINTS_DIR
        review_module.DEFAULT_COMPLAINTS_DIR = temp_data_dir / "complaints"

        try:
            _save_complaint(sample_complaint)
            loaded = _load_complaint("TEST-001")

            assert loaded is not None
            assert loaded.complaint_id == sample_complaint.complaint_id
            assert (
                loaded.device_info.device_name
                == sample_complaint.device_info.device_name
            )
        finally:
            review_module.DEFAULT_COMPLAINTS_DIR = original_dir

    def test_save_and_load_decision(
        self, temp_data_dir: Path, sample_decision: CodingDecision
    ) -> None:
        """Decisions can be saved and loaded."""
        import src.cli.review as review_module

        original_dir = review_module.DEFAULT_DECISIONS_DIR
        review_module.DEFAULT_DECISIONS_DIR = temp_data_dir / "decisions"

        try:
            _save_decision(sample_decision)
            loaded = _load_decision("TEST-001")

            assert loaded is not None
            assert loaded.complaint_id == sample_decision.complaint_id
            assert len(loaded.suggested_codes) == 2
        finally:
            review_module.DEFAULT_DECISIONS_DIR = original_dir

    def test_load_nonexistent_complaint(self, temp_data_dir: Path) -> None:
        """Loading nonexistent complaint returns None."""
        import src.cli.review as review_module

        original_dir = review_module.DEFAULT_COMPLAINTS_DIR
        review_module.DEFAULT_COMPLAINTS_DIR = temp_data_dir / "complaints"

        try:
            loaded = _load_complaint("NONEXISTENT")
            assert loaded is None
        finally:
            review_module.DEFAULT_COMPLAINTS_DIR = original_dir


class TestListCommand:
    """Tests for the list command."""

    def test_list_empty(self, temp_data_dir: Path) -> None:
        """List with no complaints shows info message."""
        import src.cli.review as review_module

        original_complaints_dir = review_module.DEFAULT_COMPLAINTS_DIR
        review_module.DEFAULT_COMPLAINTS_DIR = temp_data_dir / "complaints"

        try:
            result = runner.invoke(app, ["list"])
            assert result.exit_code == 0
            assert "No complaints found" in result.stdout
        finally:
            review_module.DEFAULT_COMPLAINTS_DIR = original_complaints_dir

    def test_list_with_complaints(
        self, temp_data_dir: Path, sample_complaint: ComplaintRecord
    ) -> None:
        """List shows complaints in a table."""
        import src.cli.review as review_module

        original_complaints_dir = review_module.DEFAULT_COMPLAINTS_DIR
        review_module.DEFAULT_COMPLAINTS_DIR = temp_data_dir / "complaints"

        try:
            _save_complaint(sample_complaint)
            result = runner.invoke(app, ["list"])
            assert result.exit_code == 0
            assert "TEST-001" in result.stdout
            assert "Total: 1 complaint(s)" in result.stdout
        finally:
            review_module.DEFAULT_COMPLAINTS_DIR = original_complaints_dir

    def test_list_pending_filter(
        self, temp_data_dir: Path, sample_complaint: ComplaintRecord
    ) -> None:
        """List with --pending filter shows only coded complaints."""
        import src.cli.review as review_module

        original_complaints_dir = review_module.DEFAULT_COMPLAINTS_DIR
        review_module.DEFAULT_COMPLAINTS_DIR = temp_data_dir / "complaints"

        try:
            # Save one coded complaint
            _save_complaint(sample_complaint)

            # Save one reviewed complaint
            reviewed = sample_complaint.model_copy(
                update={"complaint_id": "TEST-002", "status": ComplaintStatus.REVIEWED}
            )
            _save_complaint(reviewed)

            result = runner.invoke(app, ["list", "--pending"])
            assert result.exit_code == 0
            assert "TEST-001" in result.stdout
            assert "TEST-002" not in result.stdout
        finally:
            review_module.DEFAULT_COMPLAINTS_DIR = original_complaints_dir

    def test_list_invalid_status(self, temp_data_dir: Path) -> None:
        """List with invalid status shows error."""
        import src.cli.review as review_module

        original_complaints_dir = review_module.DEFAULT_COMPLAINTS_DIR
        review_module.DEFAULT_COMPLAINTS_DIR = temp_data_dir / "complaints"

        try:
            result = runner.invoke(app, ["list", "--status", "invalid"])
            assert result.exit_code == 1
            assert "Invalid status" in result.stdout
        finally:
            review_module.DEFAULT_COMPLAINTS_DIR = original_complaints_dir


class TestShowCommand:
    """Tests for the show command."""

    def test_show_complaint(
        self,
        temp_data_dir: Path,
        sample_complaint: ComplaintRecord,
        sample_decision: CodingDecision,
    ) -> None:
        """Show displays complaint details."""
        import src.cli.review as review_module

        original_complaints_dir = review_module.DEFAULT_COMPLAINTS_DIR
        original_decisions_dir = review_module.DEFAULT_DECISIONS_DIR
        review_module.DEFAULT_COMPLAINTS_DIR = temp_data_dir / "complaints"
        review_module.DEFAULT_DECISIONS_DIR = temp_data_dir / "decisions"

        try:
            _save_complaint(sample_complaint)
            _save_decision(sample_decision)

            result = runner.invoke(app, ["show", "TEST-001"])
            assert result.exit_code == 0
            assert "TEST-001" in result.stdout
            assert "Test Pacemaker Model X" in result.stdout
        finally:
            review_module.DEFAULT_COMPLAINTS_DIR = original_complaints_dir
            review_module.DEFAULT_DECISIONS_DIR = original_decisions_dir

    def test_show_nonexistent_complaint(self, temp_data_dir: Path) -> None:
        """Show with nonexistent ID shows error."""
        import src.cli.review as review_module

        original_complaints_dir = review_module.DEFAULT_COMPLAINTS_DIR
        review_module.DEFAULT_COMPLAINTS_DIR = temp_data_dir / "complaints"

        try:
            result = runner.invoke(app, ["show", "NONEXISTENT"])
            assert result.exit_code == 1
            assert "not found" in result.stdout
        finally:
            review_module.DEFAULT_COMPLAINTS_DIR = original_complaints_dir


class TestApproveCommand:
    """Tests for the approve command."""

    def test_approve_all(
        self,
        temp_data_dir: Path,
        sample_complaint: ComplaintRecord,
        sample_decision: CodingDecision,
    ) -> None:
        """Approve --all approves all suggested codes."""
        import src.cli.review as review_module

        original_complaints_dir = review_module.DEFAULT_COMPLAINTS_DIR
        original_decisions_dir = review_module.DEFAULT_DECISIONS_DIR
        review_module.DEFAULT_COMPLAINTS_DIR = temp_data_dir / "complaints"
        review_module.DEFAULT_DECISIONS_DIR = temp_data_dir / "decisions"

        audit_dir = temp_data_dir / "audit_logs"

        try:
            _save_complaint(sample_complaint)
            _save_decision(sample_decision)

            result = runner.invoke(
                app,
                [
                    "approve",
                    "TEST-001",
                    "--all",
                    "--reviewer",
                    "test_user",
                    "--audit-dir",
                    str(audit_dir),
                ],
            )

            assert result.exit_code == 0
            assert "Review recorded" in result.stdout

            # Verify decision was updated
            decision = _load_decision("TEST-001")
            assert decision is not None
            assert len(decision.approved_codes) == 2
            assert "A0601" in decision.approved_codes
            assert "A0101" in decision.approved_codes
            assert decision.reviewer_id == "test_user"
            assert decision.is_reviewed

            # Verify complaint status updated
            complaint = _load_complaint("TEST-001")
            assert complaint is not None
            assert complaint.status == ComplaintStatus.REVIEWED

            # Verify audit event logged
            audit_logger = AuditLogger(log_dir=audit_dir)
            events = audit_logger.get_events("TEST-001")
            assert len(events) == 1
            assert events[0].action == AuditAction.CODING_REVIEWED
        finally:
            review_module.DEFAULT_COMPLAINTS_DIR = original_complaints_dir
            review_module.DEFAULT_DECISIONS_DIR = original_decisions_dir

    def test_approve_already_reviewed(
        self,
        temp_data_dir: Path,
        sample_complaint: ComplaintRecord,
        sample_decision: CodingDecision,
    ) -> None:
        """Approve on already reviewed complaint shows warning."""
        import src.cli.review as review_module

        original_complaints_dir = review_module.DEFAULT_COMPLAINTS_DIR
        original_decisions_dir = review_module.DEFAULT_DECISIONS_DIR
        review_module.DEFAULT_COMPLAINTS_DIR = temp_data_dir / "complaints"
        review_module.DEFAULT_DECISIONS_DIR = temp_data_dir / "decisions"

        try:
            # Mark decision as already reviewed
            sample_decision.reviewer_id = "previous_reviewer"
            sample_decision.review_timestamp = datetime.now(UTC)
            _save_complaint(sample_complaint)
            _save_decision(sample_decision)

            result = runner.invoke(app, ["approve", "TEST-001", "--all"])
            assert result.exit_code == 1
            assert "already been reviewed" in result.stdout
        finally:
            review_module.DEFAULT_COMPLAINTS_DIR = original_complaints_dir
            review_module.DEFAULT_DECISIONS_DIR = original_decisions_dir


class TestRejectCommand:
    """Tests for the reject command."""

    def test_reject_all(
        self,
        temp_data_dir: Path,
        sample_complaint: ComplaintRecord,
        sample_decision: CodingDecision,
    ) -> None:
        """Reject rejects all suggestions and requeues complaint."""
        import src.cli.review as review_module

        original_complaints_dir = review_module.DEFAULT_COMPLAINTS_DIR
        original_decisions_dir = review_module.DEFAULT_DECISIONS_DIR
        review_module.DEFAULT_COMPLAINTS_DIR = temp_data_dir / "complaints"
        review_module.DEFAULT_DECISIONS_DIR = temp_data_dir / "decisions"

        audit_dir = temp_data_dir / "audit_logs"

        try:
            _save_complaint(sample_complaint)
            _save_decision(sample_decision)

            # Mock confirmation prompt
            with patch("src.cli.review.prompt_confirmation", return_value=True):
                result = runner.invoke(
                    app,
                    [
                        "reject",
                        "TEST-001",
                        "--reason",
                        "Suggestions not applicable",
                        "--reviewer",
                        "test_user",
                        "--audit-dir",
                        str(audit_dir),
                    ],
                )

            assert result.exit_code == 0
            assert "rejected" in result.stdout.lower()

            # Verify decision was updated
            decision = _load_decision("TEST-001")
            assert decision is not None
            assert len(decision.rejected_codes) == 2
            assert len(decision.approved_codes) == 0
            assert "REJECTED" in (decision.review_notes or "")

            # Verify complaint status reset to EXTRACTED
            complaint = _load_complaint("TEST-001")
            assert complaint is not None
            assert complaint.status == ComplaintStatus.EXTRACTED
        finally:
            review_module.DEFAULT_COMPLAINTS_DIR = original_complaints_dir
            review_module.DEFAULT_DECISIONS_DIR = original_decisions_dir


class TestHistoryCommand:
    """Tests for the history command."""

    def test_history_no_events(self, temp_data_dir: Path) -> None:
        """History with no events shows info message."""
        audit_dir = temp_data_dir / "audit_logs"

        result = runner.invoke(
            app,
            ["history", "TEST-001", "--audit-dir", str(audit_dir)],
        )
        assert result.exit_code == 0
        assert "No audit history found" in result.stdout

    def test_history_with_events(
        self,
        temp_data_dir: Path,
        sample_complaint: ComplaintRecord,
        sample_decision: CodingDecision,
    ) -> None:
        """History shows audit events for a complaint."""
        import src.cli.review as review_module

        original_complaints_dir = review_module.DEFAULT_COMPLAINTS_DIR
        original_decisions_dir = review_module.DEFAULT_DECISIONS_DIR
        review_module.DEFAULT_COMPLAINTS_DIR = temp_data_dir / "complaints"
        review_module.DEFAULT_DECISIONS_DIR = temp_data_dir / "decisions"

        audit_dir = temp_data_dir / "audit_logs"

        try:
            _save_complaint(sample_complaint)
            _save_decision(sample_decision)

            # First approve to create an audit event
            runner.invoke(
                app,
                [
                    "approve",
                    "TEST-001",
                    "--all",
                    "--audit-dir",
                    str(audit_dir),
                ],
            )

            # Then check history
            result = runner.invoke(
                app,
                ["history", "TEST-001", "--audit-dir", str(audit_dir)],
            )

            assert result.exit_code == 0
            assert "Audit History" in result.stdout
            assert "coding_reviewed" in result.stdout
        finally:
            review_module.DEFAULT_COMPLAINTS_DIR = original_complaints_dir
            review_module.DEFAULT_DECISIONS_DIR = original_decisions_dir


class TestCodingDecisionModel:
    """Tests for CodingDecision model properties."""

    def test_final_codes(self, sample_decision: CodingDecision) -> None:
        """Final codes combines approved and added codes."""
        sample_decision.approved_codes = ["A0601"]
        sample_decision.added_codes = ["A0701"]

        final = sample_decision.final_codes
        assert "A0601" in final
        assert "A0701" in final
        assert len(final) == 2

    def test_is_reviewed(self, sample_decision: CodingDecision) -> None:
        """is_reviewed returns True when reviewer set."""
        assert not sample_decision.is_reviewed

        sample_decision.reviewer_id = "test_user"
        sample_decision.review_timestamp = datetime.now(UTC)
        assert sample_decision.is_reviewed

    def test_suggestion_accuracy(self, sample_decision: CodingDecision) -> None:
        """Suggestion accuracy calculation."""
        # No approvals yet - returns 0.0 since there are suggestions
        assert sample_decision.suggestion_accuracy == 0.0

        # Approve 1 of 2
        sample_decision.approved_codes = ["A0601"]
        accuracy = sample_decision.suggestion_accuracy
        assert accuracy is not None
        assert accuracy == 0.5  # 1 approved out of 2 suggested

    def test_suggestion_accuracy_no_suggestions(self) -> None:
        """Suggestion accuracy is None when no suggestions exist."""
        decision = CodingDecision(complaint_id="TEST-002")
        assert decision.suggestion_accuracy is None
