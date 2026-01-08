"""Test case loader for synthetic complaint data."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.models import ComplaintRecord, IntakeChannel
from src.models.enums import DeviceType


class GroundTruth(BaseModel):
    """Ground truth data for a test case."""

    expected_imdrf_codes: list[str] = Field(
        ..., description="Expected IMDRF codes for this complaint"
    )
    alternative_codes: list[str] = Field(
        default_factory=list, description="Alternative acceptable codes"
    )
    requires_mdr: bool = Field(..., description="Whether MDR filing is required")
    mdr_criteria: list[str] = Field(
        default_factory=list, description="MDR criteria that apply"
    )
    coding_rationale: str = Field(
        ..., description="Explanation of why these codes were chosen"
    )


class ComplaintTestCase(BaseModel):
    """A complete test case with raw input and expected output."""

    test_case_id: str = Field(..., description="Unique test case identifier")
    name: str = Field(..., description="Human-readable test case name")
    description: str = Field(..., description="Description of what this tests")
    channel: IntakeChannel = Field(..., description="Intake channel for this test")
    device_type: DeviceType = Field(..., description="Type of device involved")
    severity: str = Field(
        ...,
        description="Severity level (death, serious_injury, malfunction, user_error, quality)",
    )
    difficulty: str = Field(
        default="medium", description="Extraction difficulty (easy, medium, hard)"
    )

    # Raw input data
    raw_input: dict[str, Any] = Field(
        ..., description="Raw input data (form fields, email content, etc.)"
    )

    # Expected extraction result
    expected_complaint: ComplaintRecord = Field(
        ..., description="Expected ComplaintRecord after extraction"
    )

    # Ground truth for coding
    ground_truth: GroundTruth = Field(
        ..., description="Ground truth coding information"
    )

    # Notes for test case
    extraction_challenges: list[str] = Field(
        default_factory=list, description="Known challenges for extraction"
    )
    notes: str | None = Field(default=None, description="Additional notes")


# Test cases directory
TEST_CASES_DIR = Path(__file__).parent.parent.parent / "data" / "test_cases"


def load_test_case(test_case_id: str) -> ComplaintTestCase | None:
    """Load a specific test case by ID.

    Args:
        test_case_id: The test case identifier.

    Returns:
        ComplaintTestCase if found, None otherwise.
    """
    # Search all channel directories
    for channel_dir in TEST_CASES_DIR.iterdir():
        if not channel_dir.is_dir():
            continue

        for file_path in channel_dir.glob("*.json"):
            if file_path.name.startswith("_"):
                continue  # Skip manifest files

            with open(file_path) as f:
                data = json.load(f)

            if data.get("test_case_id") == test_case_id:
                return ComplaintTestCase.model_validate(data)

    return None


def load_test_cases_by_channel(channel: IntakeChannel) -> list[ComplaintTestCase]:
    """Load all test cases for a specific channel.

    Args:
        channel: The intake channel.

    Returns:
        List of ComplaintTestCase objects for that channel.
    """
    channel_dir = TEST_CASES_DIR / channel.value
    if not channel_dir.exists():
        return []

    test_cases: list[ComplaintTestCase] = []

    for file_path in sorted(channel_dir.glob("*.json")):
        if file_path.name.startswith("_"):
            continue  # Skip manifest files

        with open(file_path) as f:
            data = json.load(f)

        test_cases.append(ComplaintTestCase.model_validate(data))

    return test_cases


def load_all_test_cases(
    device_type: DeviceType | None = None,
    severity: str | None = None,
    difficulty: str | None = None,
) -> list[ComplaintTestCase]:
    """Load all test cases, optionally filtered.

    Args:
        device_type: Optional filter by device type.
        severity: Optional filter by severity level.
        difficulty: Optional filter by difficulty.

    Returns:
        List of ComplaintTestCase objects matching filters.
    """
    all_cases: list[ComplaintTestCase] = []

    for channel_dir in TEST_CASES_DIR.iterdir():
        if not channel_dir.is_dir():
            continue

        for file_path in sorted(channel_dir.glob("*.json")):
            if file_path.name.startswith("_"):
                continue

            with open(file_path) as f:
                data = json.load(f)

            test_case = ComplaintTestCase.model_validate(data)

            # Apply filters
            if device_type is not None and test_case.device_type != device_type:
                continue
            if severity is not None and test_case.severity != severity:
                continue
            if difficulty is not None and test_case.difficulty != difficulty:
                continue

            all_cases.append(test_case)

    return all_cases


def get_test_case_summary() -> dict[str, Any]:
    """Get a summary of all available test cases.

    Returns:
        Dictionary with counts by channel, device type, and severity.
    """
    all_cases = load_all_test_cases()

    summary: dict[str, Any] = {
        "total": len(all_cases),
        "by_channel": {},
        "by_device_type": {},
        "by_severity": {},
        "by_difficulty": {},
    }

    for case in all_cases:
        channel = case.channel.value
        device = case.device_type.value
        severity = case.severity
        difficulty = case.difficulty

        summary["by_channel"][channel] = summary["by_channel"].get(channel, 0) + 1
        summary["by_device_type"][device] = summary["by_device_type"].get(device, 0) + 1
        summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
        summary["by_difficulty"][difficulty] = (
            summary["by_difficulty"].get(difficulty, 0) + 1
        )

    return summary
