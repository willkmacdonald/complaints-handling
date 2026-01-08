"""Pipeline result models for end-to-end form processing."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.coding.service import CodingResult
from src.intake.forms import FormValidationResult
from src.models.complaint import ComplaintRecord
from src.models.mdr import MDRDetermination


class ProcessingStatus(str, Enum):
    """Status of pipeline processing."""

    SUCCESS = "success"
    PARTIAL = "partial"  # Some steps succeeded, some failed
    FAILED = "failed"


class PipelineError(Exception):
    """Exception raised when pipeline processing fails."""

    def __init__(self, message: str, step: str, details: dict[str, Any] | None = None):
        """Initialize pipeline error.

        Args:
            message: Error message.
            step: Pipeline step where error occurred.
            details: Additional error details.
        """
        super().__init__(message)
        self.step = step
        self.details = details or {}


class ProcessingResult(BaseModel):
    """Result of end-to-end form processing pipeline.

    Contains all outputs from each step of the pipeline along with
    metadata about the processing run.
    """

    # Processing metadata
    status: ProcessingStatus = Field(..., description="Overall processing status")
    processing_id: str = Field(..., description="Unique ID for this processing run")
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When processing started",
    )
    completed_at: datetime | None = Field(
        default=None, description="When processing completed"
    )

    # Step 1: Form parsing result
    form_valid: bool = Field(
        default=False, description="Whether form parsing succeeded"
    )
    validation_result: FormValidationResult | None = Field(
        default=None, description="Form validation details"
    )

    # Step 2: Complaint record
    complaint: ComplaintRecord | None = Field(
        default=None, description="Converted complaint record"
    )

    # Step 3: IMDRF coding result
    coding_result: CodingResult | None = Field(
        default=None, description="IMDRF code suggestions"
    )

    # Step 4: MDR determination
    mdr_determination: MDRDetermination | None = Field(
        default=None, description="MDR determination result"
    )

    # Audit tracking
    audit_event_ids: list[str] = Field(
        default_factory=list, description="IDs of audit events created"
    )

    # Error tracking
    errors: list[dict[str, Any]] = Field(
        default_factory=list, description="Errors encountered during processing"
    )

    @property
    def complaint_id(self) -> str | None:
        """Return the complaint ID if available."""
        return self.complaint.complaint_id if self.complaint else None

    @property
    def suggested_codes(self) -> list[str]:
        """Return list of suggested IMDRF code IDs."""
        if self.coding_result and self.coding_result.suggestions:
            return [s.code_id for s in self.coding_result.suggestions]
        return []

    @property
    def requires_mdr(self) -> bool | None:
        """Return MDR requirement if determined."""
        return self.mdr_determination.requires_mdr if self.mdr_determination else None

    @property
    def is_complete(self) -> bool:
        """Check if all pipeline steps completed successfully."""
        return (
            self.form_valid
            and self.complaint is not None
            and self.coding_result is not None
            and self.coding_result.is_success
            and self.mdr_determination is not None
        )

    @property
    def processing_duration_ms(self) -> float | None:
        """Calculate processing duration in milliseconds."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return delta.total_seconds() * 1000
        return None

    def add_error(
        self, step: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        """Record an error that occurred during processing.

        Args:
            step: Pipeline step where error occurred.
            message: Error message.
            details: Additional error details.
        """
        self.errors.append(
            {
                "step": step,
                "message": message,
                "details": details or {},
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    def summary(self) -> dict[str, Any]:
        """Return a summary of the processing result.

        Returns:
            Dictionary with key processing metrics.
        """
        return {
            "processing_id": self.processing_id,
            "status": self.status.value,
            "complaint_id": self.complaint_id,
            "form_valid": self.form_valid,
            "num_suggested_codes": len(self.suggested_codes),
            "suggested_codes": self.suggested_codes,
            "requires_mdr": self.requires_mdr,
            "mdr_priority": (
                self.mdr_determination.review_priority
                if self.mdr_determination
                else None
            ),
            "processing_duration_ms": self.processing_duration_ms,
            "error_count": len(self.errors),
        }
