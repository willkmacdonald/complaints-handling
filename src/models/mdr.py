"""Medical Device Report (MDR) determination models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class MDRCriteria(str, Enum):
    """Criteria that trigger MDR filing requirement."""

    DEATH = "death"
    SERIOUS_INJURY = "serious_injury"
    MALFUNCTION_COULD_CAUSE_DEATH = "malfunction_could_cause_death"
    MALFUNCTION_COULD_CAUSE_SERIOUS_INJURY = "malfunction_could_cause_serious_injury"


class MDRDetermination(BaseModel):
    """Determination of whether a complaint requires MDR filing."""

    complaint_id: str = Field(..., description="ID of the complaint")

    # Primary determination
    requires_mdr: bool = Field(..., description="Whether MDR filing is required")
    mdr_criteria_met: list[MDRCriteria] = Field(
        default_factory=list, description="Which MDR criteria were triggered"
    )

    # AI analysis
    ai_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="AI confidence in determination"
    )
    ai_reasoning: str = Field(..., description="AI explanation for determination")
    key_factors: list[str] = Field(
        default_factory=list, description="Key factors influencing determination"
    )

    # Conservative flagging
    needs_human_review: bool = Field(
        default=True, description="Whether human review is required"
    )
    review_priority: str = Field(
        default="normal",
        description="Priority level for review (low/normal/high/urgent)",
    )

    # Human review
    human_confirmed: bool | None = Field(
        default=None, description="Human reviewer's confirmation of determination"
    )
    human_override_reason: str | None = Field(
        default=None, description="Reason if human overrode AI determination"
    )
    reviewer_id: str | None = Field(default=None, description="ID of reviewer")
    review_timestamp: datetime | None = Field(default=None, description="Review time")

    # MDR filing details (if required)
    mdr_report_number: str | None = Field(
        default=None, description="Assigned MDR report number"
    )
    mdr_due_date: datetime | None = Field(
        default=None, description="Due date for MDR submission"
    )

    @property
    def is_finalized(self) -> bool:
        """Check if MDR determination has been finalized by human review."""
        return self.human_confirmed is not None

    @property
    def final_requires_mdr(self) -> bool | None:
        """Return final MDR requirement after human review, or None if not reviewed."""
        if self.human_confirmed is None:
            return None
        if self.human_override_reason:
            return not self.requires_mdr  # Human overrode
        return self.requires_mdr  # Human confirmed AI determination
