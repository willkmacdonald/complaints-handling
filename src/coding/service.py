"""IMDRF code suggestion service using LLM."""

import logging
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from src.imdrf import get_code_by_id, get_codes_by_type, get_full_path
from src.llm import LLMClient, ParseError, parse_json_response
from src.llm.prompts import IMDRF_CODING_TEMPLATE, create_messages, render_prompt
from src.models.coding import CodingSuggestion
from src.models.complaint import ComplaintRecord
from src.models.enums import IMDRFCodeType

logger = logging.getLogger(__name__)

# Minimum confidence threshold for suggestions
DEFAULT_MIN_CONFIDENCE = 0.3

# Maximum suggestions to return
DEFAULT_MAX_SUGGESTIONS = 10


class CodingResult(BaseModel):
    """Result of IMDRF code suggestion for a complaint."""

    complaint_id: str = Field(..., description="ID of the analyzed complaint")
    suggestions: list[CodingSuggestion] = Field(
        default_factory=list, description="Suggested IMDRF codes"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When suggestions were generated",
    )
    model_used: str | None = Field(
        default=None, description="LLM model used for suggestions"
    )
    tokens_used: int = Field(default=0, description="Total tokens consumed")
    latency_ms: float = Field(default=0.0, description="Request latency in ms")
    error: str | None = Field(default=None, description="Error message if failed")

    @property
    def is_success(self) -> bool:
        """Check if coding was successful."""
        return self.error is None

    @property
    def device_problem_codes(self) -> list[CodingSuggestion]:
        """Get only device problem code suggestions."""
        return [
            s for s in self.suggestions if s.code_type == IMDRFCodeType.DEVICE_PROBLEM
        ]

    @property
    def patient_problem_codes(self) -> list[CodingSuggestion]:
        """Get only patient problem code suggestions."""
        return [
            s for s in self.suggestions if s.code_type == IMDRFCodeType.PATIENT_PROBLEM
        ]


def _format_codes_for_prompt(max_codes: int = 100) -> str:
    """Format IMDRF codes as a reference for the LLM prompt.

    Args:
        max_codes: Maximum number of codes to include.

    Returns:
        Formatted string with code reference.
    """
    lines: list[str] = []

    # Get device problem codes (Annex A)
    device_codes = get_codes_by_type(IMDRFCodeType.DEVICE_PROBLEM)
    lines.append("### Annex A - Device Problem Codes")
    for code in device_codes[: max_codes // 2]:
        indent = "  " * (code.level - 1)
        desc = f" - {code.description}" if code.description else ""
        lines.append(f"{indent}- {code.code_id}: {code.name}{desc}")

    lines.append("")

    # Get patient problem codes (Annex C)
    patient_codes = get_codes_by_type(IMDRFCodeType.PATIENT_PROBLEM)
    lines.append("### Annex C - Patient Problem Codes")
    for code in patient_codes[: max_codes // 2]:
        indent = "  " * (code.level - 1)
        desc = f" - {code.description}" if code.description else ""
        lines.append(f"{indent}- {code.code_id}: {code.name}{desc}")

    return "\n".join(lines)


def _parse_suggestion(raw: dict[str, Any]) -> CodingSuggestion | None:
    """Parse a raw suggestion dict into a CodingSuggestion.

    Args:
        raw: Dictionary from LLM response.

    Returns:
        CodingSuggestion if valid, None otherwise.
    """
    code_id = raw.get("code_id", "").strip()
    if not code_id:
        return None

    # Validate against IMDRF reference
    imdrf_code = get_code_by_id(code_id)
    if imdrf_code is None:
        logger.warning("Invalid IMDRF code suggested: %s", code_id)
        return None

    # Extract confidence
    confidence = raw.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    confidence = max(0.0, min(1.0, float(confidence)))

    # Get full path for the code
    full_path = get_full_path(code_id)

    return CodingSuggestion(
        code_id=code_id,
        code_name=imdrf_code.name,
        code_type=imdrf_code.code_type,
        confidence=confidence,
        source_text=str(raw.get("source_text", "")),
        reasoning=str(raw.get("reasoning", "")),
        full_path=full_path,
    )


def _validate_and_filter_suggestions(
    suggestions: list[CodingSuggestion],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_suggestions: int = DEFAULT_MAX_SUGGESTIONS,
) -> list[CodingSuggestion]:
    """Validate and filter suggestions.

    Args:
        suggestions: Raw suggestions from LLM.
        min_confidence: Minimum confidence threshold.
        max_suggestions: Maximum number to return.

    Returns:
        Filtered and sorted list of suggestions.
    """
    # Filter by confidence
    filtered = [s for s in suggestions if s.confidence >= min_confidence]

    # Sort by confidence descending
    filtered.sort(key=lambda s: s.confidence, reverse=True)

    # Limit count
    return filtered[:max_suggestions]


class CodingService:
    """Service for suggesting IMDRF codes for complaints.

    This service uses an LLM to analyze complaint narratives and suggest
    appropriate IMDRF codes with confidence scores and reasoning.
    """

    def __init__(
        self,
        client: LLMClient,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        max_suggestions: int = DEFAULT_MAX_SUGGESTIONS,
    ):
        """Initialize the coding service.

        Args:
            client: LLM client for making requests.
            min_confidence: Minimum confidence threshold for suggestions.
            max_suggestions: Maximum number of suggestions to return.
        """
        self.client = client
        self.min_confidence = min_confidence
        self.max_suggestions = max_suggestions

    def suggest_codes(self, complaint: ComplaintRecord) -> CodingResult:
        """Suggest IMDRF codes for a complaint.

        Args:
            complaint: The complaint record to analyze.

        Returns:
            CodingResult with suggestions or error.
        """
        try:
            # Build the prompt
            available_codes = _format_codes_for_prompt()

            system_prompt, user_prompt = render_prompt(
                IMDRF_CODING_TEMPLATE,
                {
                    "available_codes": available_codes,
                    "device_name": complaint.device_info.device_name,
                    "manufacturer": complaint.device_info.manufacturer,
                    "device_type": complaint.device_info.device_type.value,
                    "narrative": complaint.narrative,
                    "patient_outcome": (
                        complaint.event_info.patient_outcome or "Not specified"
                    ),
                },
            )

            messages = create_messages(system_prompt, user_prompt)

            # Make LLM request
            response = self.client.complete(
                messages=messages,
                temperature=0.0,  # Deterministic for consistency
                response_format={"type": "json_object"},
            )

            # Parse response
            try:
                data = parse_json_response(response.content)
            except ParseError as e:
                logger.error("Failed to parse LLM response: %s", e)
                return CodingResult(
                    complaint_id=complaint.complaint_id,
                    error=f"Failed to parse LLM response: {e}",
                    model_used=response.model,
                    tokens_used=response.usage.total_tokens,
                    latency_ms=response.latency_ms,
                )

            # Extract suggestions
            raw_suggestions = (
                data.get("suggestions", []) if isinstance(data, dict) else []
            )
            suggestions: list[CodingSuggestion] = []

            for raw in raw_suggestions:
                if isinstance(raw, dict):
                    suggestion = _parse_suggestion(raw)
                    if suggestion is not None:
                        suggestions.append(suggestion)

            # Filter and validate
            validated = _validate_and_filter_suggestions(
                suggestions,
                min_confidence=self.min_confidence,
                max_suggestions=self.max_suggestions,
            )

            return CodingResult(
                complaint_id=complaint.complaint_id,
                suggestions=validated,
                model_used=response.model,
                tokens_used=response.usage.total_tokens,
                latency_ms=response.latency_ms,
            )

        except Exception as e:
            logger.exception(
                "Error suggesting codes for complaint %s", complaint.complaint_id
            )
            return CodingResult(
                complaint_id=complaint.complaint_id,
                error=str(e),
            )


def suggest_codes(
    complaint: ComplaintRecord,
    client: LLMClient,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_suggestions: int = DEFAULT_MAX_SUGGESTIONS,
) -> CodingResult:
    """Convenience function to suggest codes for a complaint.

    Args:
        complaint: The complaint record to analyze.
        client: LLM client for making requests.
        min_confidence: Minimum confidence threshold.
        max_suggestions: Maximum number of suggestions.

    Returns:
        CodingResult with suggestions or error.
    """
    service = CodingService(
        client=client,
        min_confidence=min_confidence,
        max_suggestions=max_suggestions,
    )
    return service.suggest_codes(complaint)
