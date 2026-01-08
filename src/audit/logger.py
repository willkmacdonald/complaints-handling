"""Audit logger implementation with JSON file storage.

Provides append-only audit logging for regulatory compliance.
Events are stored in JSON Lines format for easy parsing and streaming.

Storage is organized by date for efficient retrieval and archival:
    audit_logs/
        2024-01-15.jsonl
        2024-01-16.jsonl
        ...
"""

import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from src.audit.models import (
    AuditAction,
    AuditEvent,
    CodingReviewedEvent,
    CodingSuggestedEvent,
    ComplaintCreatedEvent,
    MDRDeterminedEvent,
)

logger = logging.getLogger(__name__)

# Type alias for all event types
EventType = (
    AuditEvent
    | ComplaintCreatedEvent
    | CodingSuggestedEvent
    | CodingReviewedEvent
    | MDRDeterminedEvent
)


def generate_event_id() -> str:
    """Generate a unique event ID.

    Format: EVT-{timestamp}-{uuid4_short}
    Example: EVT-20240115143052-a1b2c3d4
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"EVT-{timestamp}-{short_uuid}"


class AuditLogger:
    """Append-only audit logger with JSON file storage.

    Events are stored in JSON Lines format (.jsonl) with one event per line.
    Files are organized by date for efficient retrieval.

    Thread-safety: File writes use append mode which is atomic on most
    file systems. For production use with concurrent access, consider
    using a database backend.

    Attributes:
        log_dir: Directory where audit logs are stored.
    """

    def __init__(self, log_dir: Path | str | None = None) -> None:
        """Initialize the audit logger.

        Args:
            log_dir: Directory for audit logs. If None, uses './audit_logs'.
        """
        if log_dir is None:
            log_dir = Path("audit_logs")
        self.log_dir = Path(log_dir)
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        """Create log directory if it doesn't exist."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_file(self, timestamp: datetime | None = None) -> Path:
        """Get the log file path for a given timestamp.

        Args:
            timestamp: Event timestamp. Defaults to current UTC time.

        Returns:
            Path to the log file for that date.
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)
        date_str = timestamp.strftime("%Y-%m-%d")
        return self.log_dir / f"{date_str}.jsonl"

    def _serialize_event(self, event: EventType) -> str:
        """Serialize an event to JSON string.

        Args:
            event: Event to serialize.

        Returns:
            JSON string representation.
        """
        # Convert to base AuditEvent if needed
        base_event = (
            event if isinstance(event, AuditEvent) else event.to_base_event()
        )

        # Serialize with datetime handling
        data = base_event.model_dump(mode="json")
        return json.dumps(data, default=str, ensure_ascii=False)

    def log_event(self, event: EventType) -> str:
        """Log an audit event.

        Appends the event to the appropriate log file. Events cannot be
        modified or deleted once logged.

        Args:
            event: The event to log.

        Returns:
            The event ID of the logged event.

        Raises:
            IOError: If unable to write to log file.
        """
        # Ensure event has an ID
        if isinstance(event, AuditEvent):
            event_id = event.event_id
            timestamp = event.timestamp
        else:
            event_id = event.event_id
            timestamp = event.timestamp

        # Serialize and write
        json_line = self._serialize_event(event)
        log_file = self._get_log_file(timestamp)

        try:
            # Append mode ensures we never overwrite existing data
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")
                f.flush()  # Ensure data is written immediately
                os.fsync(f.fileno())  # Force write to disk

            logger.debug("Logged audit event %s to %s", event_id, log_file)
            return event_id

        except OSError as e:
            logger.error("Failed to write audit event %s: %s", event_id, e)
            raise

    def get_events(
        self,
        resource_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[AuditEvent]:
        """Retrieve events for a specific resource.

        Args:
            resource_id: ID of the resource to get events for.
            start_date: Optional start date for filtering.
            end_date: Optional end date for filtering.

        Returns:
            List of audit events for the resource, sorted by timestamp.
        """
        events: list[AuditEvent] = []

        # Determine date range
        if start_date is None:
            start_date = datetime(2020, 1, 1, tzinfo=UTC)
        if end_date is None:
            end_date = datetime.now(UTC)

        # Iterate through log files in date range
        log_files = sorted(self.log_dir.glob("*.jsonl"))
        for log_file in log_files:
            # Parse date from filename
            try:
                file_date_str = log_file.stem
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d").replace(
                    tzinfo=UTC
                )
            except ValueError:
                continue

            # Skip files outside date range
            if file_date.date() < start_date.date():
                continue
            if file_date.date() > end_date.date():
                continue

            # Read and filter events
            try:
                with open(log_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if data.get("resource_id") == resource_id:
                                events.append(AuditEvent.model_validate(data))
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning(
                                "Skipping malformed event in %s: %s", log_file, e
                            )
                            continue
            except OSError as e:
                logger.error("Failed to read audit log %s: %s", log_file, e)
                continue

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)
        return events

    def get_events_by_action(
        self,
        action: AuditAction,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[AuditEvent]:
        """Retrieve events by action type.

        Args:
            action: Action type to filter by.
            start_date: Optional start date for filtering.
            end_date: Optional end date for filtering.

        Returns:
            List of audit events matching the action, sorted by timestamp.
        """
        events: list[AuditEvent] = []

        if start_date is None:
            start_date = datetime(2020, 1, 1, tzinfo=UTC)
        if end_date is None:
            end_date = datetime.now(UTC)

        log_files = sorted(self.log_dir.glob("*.jsonl"))
        for log_file in log_files:
            try:
                file_date_str = log_file.stem
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d").replace(
                    tzinfo=UTC
                )
            except ValueError:
                continue

            if file_date.date() < start_date.date():
                continue
            if file_date.date() > end_date.date():
                continue

            try:
                with open(log_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if data.get("action") == action.value:
                                events.append(AuditEvent.model_validate(data))
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning(
                                "Skipping malformed event in %s: %s", log_file, e
                            )
                            continue
            except OSError as e:
                logger.error("Failed to read audit log %s: %s", log_file, e)
                continue

        events.sort(key=lambda e: e.timestamp)
        return events

    def get_all_events(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[AuditEvent]:
        """Retrieve all events within a date range.

        Args:
            start_date: Optional start date for filtering.
            end_date: Optional end date for filtering.

        Returns:
            List of all audit events, sorted by timestamp.
        """
        events: list[AuditEvent] = []

        if start_date is None:
            start_date = datetime(2020, 1, 1, tzinfo=UTC)
        if end_date is None:
            end_date = datetime.now(UTC)

        log_files = sorted(self.log_dir.glob("*.jsonl"))
        for log_file in log_files:
            try:
                file_date_str = log_file.stem
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d").replace(
                    tzinfo=UTC
                )
            except ValueError:
                continue

            if file_date.date() < start_date.date():
                continue
            if file_date.date() > end_date.date():
                continue

            try:
                with open(log_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            events.append(AuditEvent.model_validate(data))
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning(
                                "Skipping malformed event in %s: %s", log_file, e
                            )
                            continue
            except OSError as e:
                logger.error("Failed to read audit log %s: %s", log_file, e)
                continue

        events.sort(key=lambda e: e.timestamp)
        return events


# Module-level default logger (lazy initialization)
_default_logger: AuditLogger | None = None


def get_audit_logger(log_dir: Path | str | None = None) -> AuditLogger:
    """Get or create the default audit logger.

    Args:
        log_dir: Optional log directory. Only used on first call.

    Returns:
        The default AuditLogger instance.
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = AuditLogger(log_dir=log_dir)
    return _default_logger


def log_event(event: EventType) -> str:
    """Log an event using the default audit logger.

    Convenience function for logging without explicitly creating a logger.

    Args:
        event: The event to log.

    Returns:
        The event ID of the logged event.
    """
    return get_audit_logger().log_event(event)
