"""Tests for audit logging module."""

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.audit.logger import AuditLogger, generate_event_id
from src.audit.models import (
    AuditAction,
    AuditEvent,
    CodingReviewedEvent,
    CodingSuggestedEvent,
    ComplaintCreatedEvent,
    MDRDeterminedEvent,
)


class TestGenerateEventId:
    """Tests for event ID generation."""

    def test_generates_unique_ids(self) -> None:
        """Each call generates a unique ID."""
        ids = [generate_event_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_id_format(self) -> None:
        """ID follows expected format."""
        event_id = generate_event_id()
        assert event_id.startswith("EVT-")
        parts = event_id.split("-")
        assert len(parts) == 3
        assert len(parts[1]) == 14  # YYYYMMDDHHMMSS
        assert len(parts[2]) == 8  # Short UUID


class TestAuditEventModels:
    """Tests for audit event models."""

    def test_audit_event_immutable(self) -> None:
        """AuditEvent is immutable once created."""
        event = AuditEvent(
            event_id="EVT-001",
            action=AuditAction.COMPLAINT_CREATED,
            resource_type="complaint",
            resource_id="COMP-001",
        )

        with pytest.raises(Exception):  # ValidationError for frozen model
            event.action = AuditAction.CODING_SUGGESTED  # type: ignore

    def test_audit_event_default_timestamp(self) -> None:
        """AuditEvent gets UTC timestamp by default."""
        event = AuditEvent(
            event_id="EVT-001",
            action=AuditAction.COMPLAINT_CREATED,
            resource_type="complaint",
            resource_id="COMP-001",
        )

        assert event.timestamp is not None
        assert event.timestamp.tzinfo is not None

    def test_audit_event_default_user(self) -> None:
        """AuditEvent defaults to system user."""
        event = AuditEvent(
            event_id="EVT-001",
            action=AuditAction.COMPLAINT_CREATED,
            resource_type="complaint",
            resource_id="COMP-001",
        )

        assert event.user_id == "system"
        assert event.user_name == "System"

    def test_audit_event_serialization(self) -> None:
        """AuditEvent can be serialized to JSON."""
        event = AuditEvent(
            event_id="EVT-001",
            action=AuditAction.COMPLAINT_CREATED,
            resource_type="complaint",
            resource_id="COMP-001",
            details={"key": "value"},
        )

        json_str = event.model_dump_json()
        data = json.loads(json_str)

        assert data["event_id"] == "EVT-001"
        assert data["action"] == "complaint_created"
        assert data["details"]["key"] == "value"


class TestComplaintCreatedEvent:
    """Tests for ComplaintCreatedEvent."""

    def test_create_event(self) -> None:
        """Create a ComplaintCreatedEvent."""
        event = ComplaintCreatedEvent(
            event_id="EVT-001",
            resource_id="COMP-001",
            intake_channel="form",
            device_name="Test Device",
            manufacturer="Test Manufacturer",
        )

        assert event.action == AuditAction.COMPLAINT_CREATED
        assert event.resource_type == "complaint"
        assert event.intake_channel == "form"

    def test_convert_to_base_event(self) -> None:
        """Convert to base AuditEvent preserves all data."""
        event = ComplaintCreatedEvent(
            event_id="EVT-001",
            resource_id="COMP-001",
            intake_channel="form",
            device_name="Test Device",
            manufacturer="Test Manufacturer",
        )

        base = event.to_base_event()

        assert base.event_id == "EVT-001"
        assert base.resource_id == "COMP-001"
        assert base.details["intake_channel"] == "form"
        assert base.details["device_name"] == "Test Device"
        assert base.details["manufacturer"] == "Test Manufacturer"


class TestCodingSuggestedEvent:
    """Tests for CodingSuggestedEvent."""

    def test_create_event(self) -> None:
        """Create a CodingSuggestedEvent."""
        event = CodingSuggestedEvent(
            event_id="EVT-002",
            resource_id="COMP-001",
            suggested_codes=[
                {"code_id": "A0601", "confidence": 0.85},
                {"code_id": "C01", "confidence": 0.90},
            ],
            model_name="gpt-4o",
            total_tokens=500,
            latency_ms=1200.5,
        )

        assert event.action == AuditAction.CODING_SUGGESTED
        assert len(event.suggested_codes) == 2

    def test_convert_to_base_event(self) -> None:
        """Convert to base AuditEvent preserves suggestions."""
        event = CodingSuggestedEvent(
            event_id="EVT-002",
            resource_id="COMP-001",
            suggested_codes=[{"code_id": "A0601", "confidence": 0.85}],
            model_name="gpt-4o",
        )

        base = event.to_base_event()

        assert len(base.details["suggested_codes"]) == 1
        assert base.details["model_name"] == "gpt-4o"


class TestCodingReviewedEvent:
    """Tests for CodingReviewedEvent."""

    def test_create_event(self) -> None:
        """Create a CodingReviewedEvent."""
        event = CodingReviewedEvent(
            event_id="EVT-003",
            resource_id="COMP-001",
            user_id="reviewer-001",
            user_name="John Doe",
            approved_codes=["A0601"],
            rejected_codes=["C01"],
            added_codes=["A0602"],
            review_notes="Changed primary code based on device analysis.",
            review_duration_seconds=120,
        )

        assert event.action == AuditAction.CODING_REVIEWED
        assert event.user_id == "reviewer-001"
        assert "A0601" in event.approved_codes
        assert "C01" in event.rejected_codes

    def test_convert_to_base_event(self) -> None:
        """Convert to base AuditEvent preserves review details."""
        event = CodingReviewedEvent(
            event_id="EVT-003",
            resource_id="COMP-001",
            user_id="reviewer-001",
            user_name="John Doe",
            approved_codes=["A0601"],
            rejected_codes=["C01"],
        )

        base = event.to_base_event()

        assert base.user_id == "reviewer-001"
        assert "A0601" in base.details["approved_codes"]
        assert "C01" in base.details["rejected_codes"]


class TestMDRDeterminedEvent:
    """Tests for MDRDeterminedEvent."""

    def test_create_event(self) -> None:
        """Create an MDRDeterminedEvent."""
        event = MDRDeterminedEvent(
            event_id="EVT-004",
            resource_id="COMP-001",
            requires_mdr=True,
            mdr_criteria_met=["death"],
            confidence=0.95,
            reasoning="Patient death reported",
            review_priority="urgent",
        )

        assert event.action == AuditAction.MDR_DETERMINED
        assert event.requires_mdr is True
        assert "death" in event.mdr_criteria_met
        assert event.review_priority == "urgent"

    def test_convert_to_base_event(self) -> None:
        """Convert to base AuditEvent preserves MDR details."""
        event = MDRDeterminedEvent(
            event_id="EVT-004",
            resource_id="COMP-001",
            requires_mdr=True,
            mdr_criteria_met=["death"],
            confidence=0.95,
            reasoning="Patient death reported",
        )

        base = event.to_base_event()

        assert base.details["requires_mdr"] is True
        assert base.details["confidence"] == 0.95


class TestAuditLogger:
    """Tests for AuditLogger."""

    @pytest.fixture
    def temp_log_dir(self) -> Path:
        """Create a temporary directory for audit logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_creates_log_directory(self, temp_log_dir: Path) -> None:
        """Logger creates log directory if it doesn't exist."""
        log_dir = temp_log_dir / "audit_logs"
        assert not log_dir.exists()

        AuditLogger(log_dir=log_dir)

        assert log_dir.exists()

    def test_log_event_creates_file(self, temp_log_dir: Path) -> None:
        """Logging an event creates the log file."""
        audit_logger = AuditLogger(log_dir=temp_log_dir)
        event = AuditEvent(
            event_id="EVT-001",
            action=AuditAction.COMPLAINT_CREATED,
            resource_type="complaint",
            resource_id="COMP-001",
        )

        audit_logger.log_event(event)

        log_files = list(temp_log_dir.glob("*.jsonl"))
        assert len(log_files) == 1

    def test_log_event_returns_event_id(self, temp_log_dir: Path) -> None:
        """log_event returns the event ID."""
        audit_logger = AuditLogger(log_dir=temp_log_dir)
        event = AuditEvent(
            event_id="EVT-001",
            action=AuditAction.COMPLAINT_CREATED,
            resource_type="complaint",
            resource_id="COMP-001",
        )

        event_id = audit_logger.log_event(event)

        assert event_id == "EVT-001"

    def test_log_multiple_events_appends(self, temp_log_dir: Path) -> None:
        """Multiple events are appended to the same file."""
        audit_logger = AuditLogger(log_dir=temp_log_dir)

        for i in range(5):
            event = AuditEvent(
                event_id=f"EVT-{i:03d}",
                action=AuditAction.COMPLAINT_CREATED,
                resource_type="complaint",
                resource_id=f"COMP-{i:03d}",
            )
            audit_logger.log_event(event)

        log_files = list(temp_log_dir.glob("*.jsonl"))
        assert len(log_files) == 1

        with open(log_files[0]) as f:
            lines = f.readlines()
        assert len(lines) == 5

    def test_log_typed_event(self, temp_log_dir: Path) -> None:
        """Typed events are logged correctly."""
        audit_logger = AuditLogger(log_dir=temp_log_dir)
        event = ComplaintCreatedEvent(
            event_id="EVT-001",
            resource_id="COMP-001",
            intake_channel="form",
            device_name="Test Device",
            manufacturer="Test Manufacturer",
        )

        audit_logger.log_event(event)

        # Read back and verify
        log_files = list(temp_log_dir.glob("*.jsonl"))
        with open(log_files[0]) as f:
            data = json.loads(f.readline())

        assert data["action"] == "complaint_created"
        assert data["details"]["device_name"] == "Test Device"

    def test_get_events_by_resource_id(self, temp_log_dir: Path) -> None:
        """Retrieve events by resource ID."""
        audit_logger = AuditLogger(log_dir=temp_log_dir)

        # Log events for different resources
        for i in range(3):
            for comp_id in ["COMP-001", "COMP-002"]:
                event = AuditEvent(
                    event_id=f"EVT-{comp_id}-{i}",
                    action=AuditAction.COMPLAINT_CREATED,
                    resource_type="complaint",
                    resource_id=comp_id,
                )
                audit_logger.log_event(event)

        events = audit_logger.get_events("COMP-001")

        assert len(events) == 3
        assert all(e.resource_id == "COMP-001" for e in events)

    def test_get_events_sorted_by_timestamp(self, temp_log_dir: Path) -> None:
        """Events are returned sorted by timestamp."""
        audit_logger = AuditLogger(log_dir=temp_log_dir)

        timestamps = [
            datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC),
        ]

        for i, ts in enumerate(timestamps):
            event = AuditEvent(
                event_id=f"EVT-{i}",
                timestamp=ts,
                action=AuditAction.COMPLAINT_CREATED,
                resource_type="complaint",
                resource_id="COMP-001",
            )
            audit_logger.log_event(event)

        events = audit_logger.get_events("COMP-001")

        assert len(events) == 3
        assert events[0].timestamp < events[1].timestamp < events[2].timestamp

    def test_get_events_by_action(self, temp_log_dir: Path) -> None:
        """Retrieve events by action type."""
        audit_logger = AuditLogger(log_dir=temp_log_dir)

        actions = [
            AuditAction.COMPLAINT_CREATED,
            AuditAction.CODING_SUGGESTED,
            AuditAction.CODING_SUGGESTED,
            AuditAction.MDR_DETERMINED,
        ]

        for i, action in enumerate(actions):
            event = AuditEvent(
                event_id=f"EVT-{i}",
                action=action,
                resource_type="complaint",
                resource_id=f"COMP-{i}",
            )
            audit_logger.log_event(event)

        events = audit_logger.get_events_by_action(AuditAction.CODING_SUGGESTED)

        assert len(events) == 2
        assert all(e.action == AuditAction.CODING_SUGGESTED for e in events)

    def test_get_all_events(self, temp_log_dir: Path) -> None:
        """Retrieve all events."""
        audit_logger = AuditLogger(log_dir=temp_log_dir)

        for i in range(5):
            event = AuditEvent(
                event_id=f"EVT-{i}",
                action=AuditAction.COMPLAINT_CREATED,
                resource_type="complaint",
                resource_id=f"COMP-{i}",
            )
            audit_logger.log_event(event)

        events = audit_logger.get_all_events()

        assert len(events) == 5

    def test_events_persisted_to_disk(self, temp_log_dir: Path) -> None:
        """Events persist across logger instances."""
        # Log events with first logger
        logger1 = AuditLogger(log_dir=temp_log_dir)
        for i in range(3):
            event = AuditEvent(
                event_id=f"EVT-{i}",
                action=AuditAction.COMPLAINT_CREATED,
                resource_type="complaint",
                resource_id="COMP-001",
            )
            logger1.log_event(event)

        # Create new logger and read events
        logger2 = AuditLogger(log_dir=temp_log_dir)
        events = logger2.get_events("COMP-001")

        assert len(events) == 3

    def test_log_file_is_human_readable(self, temp_log_dir: Path) -> None:
        """Log files are human-readable JSON."""
        audit_logger = AuditLogger(log_dir=temp_log_dir)
        event = ComplaintCreatedEvent(
            event_id="EVT-001",
            resource_id="COMP-001",
            intake_channel="form",
            device_name="Test Device",
            manufacturer="Test Manufacturer",
        )

        audit_logger.log_event(event)

        log_files = list(temp_log_dir.glob("*.jsonl"))
        content = log_files[0].read_text()

        # Should be valid JSON
        data = json.loads(content.strip())
        assert "event_id" in data
        assert "timestamp" in data

        # Should be readable (not minified beyond recognition)
        assert "EVT-001" in content
        assert "Test Device" in content


class TestAppendOnlyBehavior:
    """Tests to verify append-only behavior."""

    @pytest.fixture
    def temp_log_dir(self) -> Path:
        """Create a temporary directory for audit logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_cannot_delete_events_programmatically(self, temp_log_dir: Path) -> None:
        """Events cannot be deleted through the API."""
        audit_logger = AuditLogger(log_dir=temp_log_dir)

        # Log an event
        event = AuditEvent(
            event_id="EVT-001",
            action=AuditAction.COMPLAINT_CREATED,
            resource_type="complaint",
            resource_id="COMP-001",
        )
        audit_logger.log_event(event)

        # Verify there's no delete method
        assert not hasattr(audit_logger, "delete_event")
        assert not hasattr(audit_logger, "remove_event")

    def test_cannot_modify_events_programmatically(self, temp_log_dir: Path) -> None:
        """Events cannot be modified through the API."""
        audit_logger = AuditLogger(log_dir=temp_log_dir)

        # Log an event
        event = AuditEvent(
            event_id="EVT-001",
            action=AuditAction.COMPLAINT_CREATED,
            resource_type="complaint",
            resource_id="COMP-001",
        )
        audit_logger.log_event(event)

        # Verify there's no update method
        assert not hasattr(audit_logger, "update_event")
        assert not hasattr(audit_logger, "modify_event")

    def test_subsequent_logs_append(self, temp_log_dir: Path) -> None:
        """Subsequent logs append, don't overwrite."""
        audit_logger = AuditLogger(log_dir=temp_log_dir)

        # Log first event
        event1 = AuditEvent(
            event_id="EVT-001",
            action=AuditAction.COMPLAINT_CREATED,
            resource_type="complaint",
            resource_id="COMP-001",
        )
        audit_logger.log_event(event1)

        # Get file size
        log_files = list(temp_log_dir.glob("*.jsonl"))
        size_after_first = log_files[0].stat().st_size

        # Log second event
        event2 = AuditEvent(
            event_id="EVT-002",
            action=AuditAction.CODING_SUGGESTED,
            resource_type="complaint",
            resource_id="COMP-001",
        )
        audit_logger.log_event(event2)

        # File should be larger
        size_after_second = log_files[0].stat().st_size
        assert size_after_second > size_after_first

        # Both events should be present
        events = audit_logger.get_events("COMP-001")
        assert len(events) == 2


class TestDateFiltering:
    """Tests for date-based filtering."""

    @pytest.fixture
    def temp_log_dir(self) -> Path:
        """Create a temporary directory for audit logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_filter_by_date_range(self, temp_log_dir: Path) -> None:
        """Events can be filtered by date range."""
        audit_logger = AuditLogger(log_dir=temp_log_dir)

        # Create events on different dates (all in one file for simplicity)
        now = datetime.now(UTC)
        timestamps = [
            now - timedelta(days=5),
            now - timedelta(days=3),
            now - timedelta(days=1),
            now,
        ]

        for i, ts in enumerate(timestamps):
            event = AuditEvent(
                event_id=f"EVT-{i}",
                timestamp=ts,
                action=AuditAction.COMPLAINT_CREATED,
                resource_type="complaint",
                resource_id="COMP-001",
            )
            audit_logger.log_event(event)

        # Filter to last 2 days
        start = now - timedelta(days=2)
        events = audit_logger.get_events("COMP-001", start_date=start)

        # Should get events from day 1 ago and today
        assert len(events) == 2
