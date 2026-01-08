"""IMDRF coding models for AI suggestions and human decisions."""

from datetime import datetime

from pydantic import BaseModel, Field

from src.models.enums import IMDRFCodeType


class CodingSuggestion(BaseModel):
    """A single IMDRF code suggestion from the AI system."""

    code_id: str = Field(..., description="IMDRF code identifier (e.g., 'A0601')")
    code_name: str = Field(..., description="Human-readable code name")
    code_type: IMDRFCodeType = Field(..., description="Type of IMDRF code")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="AI confidence score (0.0-1.0)"
    )
    source_text: str = Field(
        ..., description="Text from complaint that triggered this suggestion"
    )
    reasoning: str = Field(
        ..., description="Explanation of why this code was suggested"
    )
    full_path: str | None = Field(
        default=None,
        description="Full hierarchical path (e.g., 'Material > Integrity > Crack')",
    )


class CodingDecision(BaseModel):
    """Record of coding suggestions and human review decisions."""

    complaint_id: str = Field(..., description="ID of the complaint being coded")

    # AI suggestions
    suggested_codes: list[CodingSuggestion] = Field(
        default_factory=list, description="Codes suggested by AI"
    )
    suggestion_timestamp: datetime | None = Field(
        default=None, description="When AI suggestions were generated"
    )

    # Human review decisions
    approved_codes: list[str] = Field(
        default_factory=list, description="Code IDs approved by reviewer"
    )
    rejected_codes: list[str] = Field(
        default_factory=list, description="Code IDs rejected by reviewer"
    )
    added_codes: list[str] = Field(
        default_factory=list, description="Code IDs added by reviewer (not suggested)"
    )

    # Review metadata
    reviewer_id: str | None = Field(default=None, description="ID of human reviewer")
    review_timestamp: datetime | None = Field(
        default=None, description="When review was completed"
    )
    review_notes: str | None = Field(
        default=None, description="Optional notes from reviewer"
    )
    review_duration_seconds: int | None = Field(
        default=None, ge=0, description="Time spent on review"
    )

    @property
    def final_codes(self) -> list[str]:
        """Return the final set of codes after human review."""
        return list(set(self.approved_codes + self.added_codes))

    @property
    def is_reviewed(self) -> bool:
        """Check if this coding has been reviewed by a human."""
        return self.reviewer_id is not None and self.review_timestamp is not None

    @property
    def suggestion_accuracy(self) -> float | None:
        """Calculate what percentage of suggestions were approved."""
        if not self.suggested_codes:
            return None
        suggested_ids = {s.code_id for s in self.suggested_codes}
        approved_from_suggestions = suggested_ids & set(self.approved_codes)
        return len(approved_from_suggestions) / len(suggested_ids)
