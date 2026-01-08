"""Audit event models for regulatory compliance logging.

All models are immutable once created (Pydantic frozen=True) and include
UTC timestamps for traceability per FDA 21 CFR Part 11 requirements.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(UTC)


class AuditAction(str, Enum):
    """Types of actions that can be audited."""

    COMPLAINT_CREATED = "complaint_created"
    COMPLAINT_UPDATED = "complaint_updated"
    CODING_SUGGESTED = "coding_suggested"
    CODING_REVIEWED = "coding_reviewed"
    MDR_DETERMINED = "mdr_determined"
    MDR_REVIEWED = "mdr_reviewed"
    DOCUMENT_ATTACHED = "document_attached"
    STATUS_CHANGED = "status_changed"


class AuditEvent(BaseModel):
    """Base audit event model.

    All audit events inherit from this base class which provides
    common fields required for regulatory compliance.

    Attributes:
        event_id: Unique identifier for this event.
        timestamp: UTC timestamp when event occurred.
        action: Type of action being logged.
        resource_type: Type of resource affected (e.g., "complaint", "coding").
        resource_id: ID of the resource affected.
        user_id: ID of user who performed the action (or "system" for automated).
        user_name: Display name of user (for audit reports).
        details: Additional action-specific details.
        metadata: Optional metadata for extensibility.
    """

    model_config = {"frozen": True}  # Make events immutable

    event_id: str = Field(..., description="Unique event identifier")
    timestamp: datetime = Field(
        default_factory=_utc_now,
        description="UTC timestamp of the event",
    )
    action: AuditAction = Field(..., description="Type of action performed")
    resource_type: str = Field(..., description="Type of resource affected")
    resource_id: str = Field(..., description="ID of the affected resource")
    user_id: str = Field(
        default="system",
        description="ID of user who performed action",
    )
    user_name: str = Field(
        default="System",
        description="Display name of user",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific details",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata",
    )


class ComplaintCreatedEvent(BaseModel):
    """Event logged when a new complaint is created.

    Captures the intake channel, initial status, and source information.
    """

    model_config = {"frozen": True}

    event_id: str = Field(..., description="Unique event identifier")
    timestamp: datetime = Field(default_factory=_utc_now)
    action: Literal[AuditAction.COMPLAINT_CREATED] = Field(
        default=AuditAction.COMPLAINT_CREATED
    )
    resource_type: Literal["complaint"] = Field(default="complaint")
    resource_id: str = Field(..., description="Complaint ID")
    user_id: str = Field(default="system")
    user_name: str = Field(default="System")

    # Complaint-specific details
    intake_channel: str = Field(..., description="Channel complaint was received from")
    device_name: str = Field(..., description="Name of the medical device")
    manufacturer: str = Field(..., description="Device manufacturer")
    initial_status: str = Field(default="new", description="Initial complaint status")

    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_base_event(self) -> AuditEvent:
        """Convert to base AuditEvent for storage."""
        return AuditEvent(
            event_id=self.event_id,
            timestamp=self.timestamp,
            action=self.action,
            resource_type=self.resource_type,
            resource_id=self.resource_id,
            user_id=self.user_id,
            user_name=self.user_name,
            details={
                "intake_channel": self.intake_channel,
                "device_name": self.device_name,
                "manufacturer": self.manufacturer,
                "initial_status": self.initial_status,
            },
            metadata=self.metadata,
        )


class CodingSuggestedEvent(BaseModel):
    """Event logged when AI suggests IMDRF codes for a complaint.

    Captures all suggested codes with confidence scores for audit trail.
    """

    model_config = {"frozen": True}

    event_id: str = Field(..., description="Unique event identifier")
    timestamp: datetime = Field(default_factory=_utc_now)
    action: Literal[AuditAction.CODING_SUGGESTED] = Field(
        default=AuditAction.CODING_SUGGESTED
    )
    resource_type: Literal["complaint"] = Field(default="complaint")
    resource_id: str = Field(..., description="Complaint ID")
    user_id: str = Field(default="system")
    user_name: str = Field(default="AI System")

    # Coding-specific details
    suggested_codes: list[dict[str, Any]] = Field(
        ..., description="List of suggested codes with confidence"
    )
    model_name: str = Field(default="unknown", description="AI model used")
    total_tokens: int = Field(default=0, description="Tokens used for suggestion")
    latency_ms: float = Field(default=0.0, description="Time to generate suggestions")

    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_base_event(self) -> AuditEvent:
        """Convert to base AuditEvent for storage."""
        return AuditEvent(
            event_id=self.event_id,
            timestamp=self.timestamp,
            action=self.action,
            resource_type=self.resource_type,
            resource_id=self.resource_id,
            user_id=self.user_id,
            user_name=self.user_name,
            details={
                "suggested_codes": self.suggested_codes,
                "model_name": self.model_name,
                "total_tokens": self.total_tokens,
                "latency_ms": self.latency_ms,
            },
            metadata=self.metadata,
        )


class CodingReviewedEvent(BaseModel):
    """Event logged when a human reviews AI coding suggestions.

    Captures the reviewer's decisions including approvals, rejections,
    and any codes added manually.
    """

    model_config = {"frozen": True}

    event_id: str = Field(..., description="Unique event identifier")
    timestamp: datetime = Field(default_factory=_utc_now)
    action: Literal[AuditAction.CODING_REVIEWED] = Field(
        default=AuditAction.CODING_REVIEWED
    )
    resource_type: Literal["complaint"] = Field(default="complaint")
    resource_id: str = Field(..., description="Complaint ID")
    user_id: str = Field(..., description="Reviewer's user ID")
    user_name: str = Field(..., description="Reviewer's name")

    # Review-specific details
    approved_codes: list[str] = Field(
        default_factory=list, description="Code IDs approved by reviewer"
    )
    rejected_codes: list[str] = Field(
        default_factory=list, description="Code IDs rejected by reviewer"
    )
    added_codes: list[str] = Field(
        default_factory=list, description="Code IDs added by reviewer"
    )
    review_notes: str | None = Field(default=None, description="Optional review notes")
    review_duration_seconds: int | None = Field(
        default=None, description="Time spent on review"
    )

    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_base_event(self) -> AuditEvent:
        """Convert to base AuditEvent for storage."""
        return AuditEvent(
            event_id=self.event_id,
            timestamp=self.timestamp,
            action=self.action,
            resource_type=self.resource_type,
            resource_id=self.resource_id,
            user_id=self.user_id,
            user_name=self.user_name,
            details={
                "approved_codes": self.approved_codes,
                "rejected_codes": self.rejected_codes,
                "added_codes": self.added_codes,
                "review_notes": self.review_notes,
                "review_duration_seconds": self.review_duration_seconds,
            },
            metadata=self.metadata,
        )


class MDRDeterminedEvent(BaseModel):
    """Event logged when MDR determination is made.

    Captures the AI's MDR determination with criteria met and confidence.
    """

    model_config = {"frozen": True}

    event_id: str = Field(..., description="Unique event identifier")
    timestamp: datetime = Field(default_factory=_utc_now)
    action: Literal[AuditAction.MDR_DETERMINED] = Field(
        default=AuditAction.MDR_DETERMINED
    )
    resource_type: Literal["complaint"] = Field(default="complaint")
    resource_id: str = Field(..., description="Complaint ID")
    user_id: str = Field(default="system")
    user_name: str = Field(default="AI System")

    # MDR-specific details
    requires_mdr: bool = Field(..., description="Whether MDR filing is required")
    mdr_criteria_met: list[str] = Field(
        default_factory=list, description="MDR criteria that were triggered"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="AI confidence in determination"
    )
    reasoning: str = Field(..., description="Explanation of determination")
    review_priority: str = Field(
        default="normal", description="Priority for human review"
    )

    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_base_event(self) -> AuditEvent:
        """Convert to base AuditEvent for storage."""
        return AuditEvent(
            event_id=self.event_id,
            timestamp=self.timestamp,
            action=self.action,
            resource_type=self.resource_type,
            resource_id=self.resource_id,
            user_id=self.user_id,
            user_name=self.user_name,
            details={
                "requires_mdr": self.requires_mdr,
                "mdr_criteria_met": self.mdr_criteria_met,
                "confidence": self.confidence,
                "reasoning": self.reasoning,
                "review_priority": self.review_priority,
            },
            metadata=self.metadata,
        )
