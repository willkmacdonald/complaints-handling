"""IMDRF code data models."""

from pydantic import BaseModel, Field

from src.models.enums import IMDRFCodeType


class IMDRFCode(BaseModel):
    """Represents a single IMDRF code in the hierarchy."""

    code_id: str = Field(..., description="Unique code identifier (e.g., 'A0601')")
    name: str = Field(..., description="Human-readable code name")
    code_type: IMDRFCodeType = Field(..., description="Type of IMDRF code (Annex)")
    parent_id: str | None = Field(
        default=None, description="Parent code ID (None for top-level)"
    )
    description: str | None = Field(
        default=None, description="Detailed description of the code"
    )
    examples: list[str] = Field(
        default_factory=list, description="Example scenarios for this code"
    )
    level: int = Field(default=1, ge=1, description="Hierarchy level (1 = top level)")

    @property
    def full_path(self) -> str:
        """Return the code name (full path computed by codes module)."""
        return self.name

    @property
    def is_top_level(self) -> bool:
        """Check if this is a top-level code."""
        return self.parent_id is None
