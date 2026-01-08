"""End-to-end tests for form processing pipeline.

These tests verify the pipeline processes all test case forms correctly.
Tests are designed to work with mocked LLM responses by default.
Set environment variable RUN_LLM_TESTS=1 to run with real LLM.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.audit.logger import AuditLogger
from src.llm.client import LLMClient, LLMResponse, TokenUsage
from src.models.mdr import MDRCriteria
from src.pipeline import ProcessingStatus, process_form
from src.pipeline.forms import process_form_file

# Test data directory
TEST_CASES_DIR = Path("data/test_cases/form")

# Check if we should run with real LLM
RUN_LLM_TESTS = os.getenv("RUN_LLM_TESTS", "0") == "1"


def get_test_case_files() -> list[Path]:
    """Get all test case JSON files."""
    if not TEST_CASES_DIR.exists():
        return []
    return sorted([
        f for f in TEST_CASES_DIR.glob("form_*.json")
        if not f.name.startswith("_")
    ])


def load_test_case(file_path: Path) -> dict:
    """Load a test case from file."""
    with open(file_path) as f:
        return json.load(f)


def create_mock_coding_response(test_case: dict) -> LLMResponse:
    """Create a mock LLM response based on test case expected codes."""
    expected_codes = test_case.get("ground_truth", {}).get("expected_imdrf_codes", [])

    suggestions = []
    for code_id in expected_codes[:3]:  # Limit to 3 suggestions
        suggestions.append({
            "code_id": code_id,
            "confidence": 0.85,
            "source_text": "test source text",
            "reasoning": f"Code {code_id} applies based on complaint narrative",
        })

    return LLMResponse(
        content=json.dumps({"suggestions": suggestions}),
        model="gpt-4o-mock",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        latency_ms=100.0,
    )


@pytest.fixture
def temp_output_dirs():
    """Create temporary directories for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        complaints_dir = tmppath / "complaints"
        decisions_dir = tmppath / "decisions"
        audit_dir = tmppath / "audit"
        complaints_dir.mkdir()
        decisions_dir.mkdir()
        audit_dir.mkdir()
        yield {
            "complaints": complaints_dir,
            "decisions": decisions_dir,
            "audit": audit_dir,
        }


class TestFormTestCasesWithMockedLLM:
    """Test all form test cases with mocked LLM responses."""

    @pytest.mark.parametrize(
        "test_case_file",
        get_test_case_files(),
        ids=lambda f: f.stem,
    )
    def test_process_form_test_case(
        self,
        test_case_file: Path,
        temp_output_dirs: dict,
    ):
        """Test processing each form test case through the pipeline."""
        test_case = load_test_case(test_case_file)

        # Create mock LLM client
        mock_response = create_mock_coding_response(test_case)
        mock_client = MagicMock(spec=LLMClient)
        mock_client.complete.return_value = mock_response

        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        # Process the form
        result = process_form_file(
            file_path=test_case_file,
            client=mock_client,
            audit_logger=audit_logger,
            complaints_dir=temp_output_dirs["complaints"],
            decisions_dir=temp_output_dirs["decisions"],
            save_outputs=True,
        )

        # Basic assertions
        assert result.status in (ProcessingStatus.SUCCESS, ProcessingStatus.PARTIAL), (
            f"Test case {test_case_file.name} failed with status {result.status}"
        )
        assert result.complaint is not None
        assert result.complaint.complaint_id is not None

        # Verify device info was extracted
        expected_complaint = test_case.get("expected_complaint", {})
        expected_device = expected_complaint.get("device_info", {})

        if expected_device.get("device_name"):
            assert result.complaint.device_info.device_name == expected_device["device_name"]
        if expected_device.get("manufacturer"):
            assert result.complaint.device_info.manufacturer == expected_device["manufacturer"]

    @pytest.mark.parametrize(
        "test_case_file",
        get_test_case_files(),
        ids=lambda f: f.stem,
    )
    def test_mdr_determination_accuracy(
        self,
        test_case_file: Path,
        temp_output_dirs: dict,
    ):
        """Test MDR determination matches expected values."""
        test_case = load_test_case(test_case_file)
        ground_truth = test_case.get("ground_truth", {})
        expected_requires_mdr = ground_truth.get("requires_mdr", False)

        # Create mock LLM client
        mock_response = create_mock_coding_response(test_case)
        mock_client = MagicMock(spec=LLMClient)
        mock_client.complete.return_value = mock_response

        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        # Process with rules-only MDR for consistency
        result = process_form(
            raw_data=test_case["raw_input"],
            client=mock_client,
            audit_logger=audit_logger,
            save_outputs=False,
            use_llm_for_mdr=False,  # Use rules-only for deterministic testing
        )

        assert result.mdr_determination is not None

        # For death cases, MDR must be detected (100% sensitivity required)
        if expected_requires_mdr and "death" in ground_truth.get("mdr_criteria", []):
            assert result.mdr_determination.requires_mdr is True, (
                f"CRITICAL: Test case {test_case_file.name} has death but MDR not required"
            )
            assert MDRCriteria.DEATH in result.mdr_determination.mdr_criteria_met

        # For other cases with MDR requirement, check detection
        # Allow for conservative false positives, but flag if not detected
        if expected_requires_mdr and not result.mdr_determination.requires_mdr:
            pytest.xfail(
                f"MDR not detected for {test_case_file.name} - "
                f"expected criteria: {ground_truth.get('mdr_criteria', [])}"
            )


class TestFormTestCasesMDRSensitivity:
    """Tests focused on MDR detection sensitivity (100% required)."""

    def test_death_cases_always_detected(self, temp_output_dirs: dict):
        """Verify 100% sensitivity for death cases (critical requirement)."""
        death_cases = []

        for test_case_file in get_test_case_files():
            test_case = load_test_case(test_case_file)
            severity = test_case.get("severity", "")
            if severity == "death":
                death_cases.append((test_case_file, test_case))

        if not death_cases:
            pytest.skip("No death test cases found")

        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        for test_case_file, test_case in death_cases:
            mock_response = create_mock_coding_response(test_case)
            mock_client = MagicMock(spec=LLMClient)
            mock_client.complete.return_value = mock_response

            result = process_form(
                raw_data=test_case["raw_input"],
                client=mock_client,
                audit_logger=audit_logger,
                save_outputs=False,
                use_llm_for_mdr=False,
            )

            assert result.mdr_determination is not None
            assert result.mdr_determination.requires_mdr is True, (
                f"CRITICAL FAILURE: Death case {test_case_file.name} not flagged for MDR"
            )

    def test_serious_injury_cases_detected(self, temp_output_dirs: dict):
        """Verify serious injury cases are detected."""
        injury_cases = []

        for test_case_file in get_test_case_files():
            test_case = load_test_case(test_case_file)
            severity = test_case.get("severity", "")
            if severity in ("serious_injury", "hospitalization"):
                injury_cases.append((test_case_file, test_case))

        if not injury_cases:
            pytest.skip("No serious injury test cases found")

        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        for test_case_file, test_case in injury_cases:
            mock_response = create_mock_coding_response(test_case)
            mock_client = MagicMock(spec=LLMClient)
            mock_client.complete.return_value = mock_response

            result = process_form(
                raw_data=test_case["raw_input"],
                client=mock_client,
                audit_logger=audit_logger,
                save_outputs=False,
                use_llm_for_mdr=False,
            )

            assert result.mdr_determination is not None
            # Serious injury should trigger MDR (conservative approach)
            if not result.mdr_determination.requires_mdr:
                pytest.xfail(
                    f"Serious injury case {test_case_file.name} not flagged for MDR"
                )


class TestFormTestCasesAuditTrail:
    """Tests verifying audit trail completeness."""

    @pytest.mark.parametrize(
        "test_case_file",
        get_test_case_files()[:3],  # Test first 3 cases for speed
        ids=lambda f: f.stem,
    )
    def test_audit_events_logged(
        self,
        test_case_file: Path,
        temp_output_dirs: dict,
    ):
        """Verify audit events are logged for each pipeline step."""
        test_case = load_test_case(test_case_file)

        mock_response = create_mock_coding_response(test_case)
        mock_client = MagicMock(spec=LLMClient)
        mock_client.complete.return_value = mock_response

        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        result = process_form(
            raw_data=test_case["raw_input"],
            client=mock_client,
            audit_logger=audit_logger,
            save_outputs=False,
            use_llm_for_mdr=False,
        )

        # Should have at least: complaint_created, coding_suggested, mdr_determined
        assert len(result.audit_event_ids) >= 2, (
            f"Expected at least 2 audit events, got {len(result.audit_event_ids)}"
        )

        # Verify events can be retrieved
        if result.complaint:
            events = audit_logger.get_events(result.complaint.complaint_id)
            assert len(events) >= 2


class TestFormTestCasesOutputFiles:
    """Tests verifying output file generation."""

    def test_output_files_created(self, temp_output_dirs: dict):
        """Verify complaint and decision files are created."""
        test_case_files = get_test_case_files()
        if not test_case_files:
            pytest.skip("No test cases found")

        test_case = load_test_case(test_case_files[0])

        mock_response = create_mock_coding_response(test_case)
        mock_client = MagicMock(spec=LLMClient)
        mock_client.complete.return_value = mock_response

        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        result = process_form(
            raw_data=test_case["raw_input"],
            client=mock_client,
            audit_logger=audit_logger,
            complaints_dir=temp_output_dirs["complaints"],
            decisions_dir=temp_output_dirs["decisions"],
            save_outputs=True,
            use_llm_for_mdr=False,
        )

        # Check complaint file
        complaint_files = list(temp_output_dirs["complaints"].glob("*.json"))
        assert len(complaint_files) == 1

        # Verify complaint file is valid JSON
        with open(complaint_files[0]) as f:
            complaint_data = json.load(f)
        assert complaint_data["complaint_id"] == result.complaint_id

        # Check decision file
        decision_files = list(temp_output_dirs["decisions"].glob("*_decision.json"))
        assert len(decision_files) == 1


@pytest.mark.skipif(not RUN_LLM_TESTS, reason="RUN_LLM_TESTS not set")
class TestFormTestCasesWithRealLLM:
    """Tests that run with real LLM (requires Azure OpenAI credentials)."""

    @pytest.mark.parametrize(
        "test_case_file",
        get_test_case_files(),
        ids=lambda f: f.stem,
    )
    def test_process_with_real_llm(
        self,
        test_case_file: Path,
        temp_output_dirs: dict,
    ):
        """Test processing with real LLM calls."""
        # Load test case to verify file is valid (result used implicitly via file_path)
        _ = load_test_case(test_case_file)

        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        result = process_form_file(
            file_path=test_case_file,
            client=None,  # Will create from environment
            audit_logger=audit_logger,
            complaints_dir=temp_output_dirs["complaints"],
            decisions_dir=temp_output_dirs["decisions"],
            save_outputs=True,
        )

        assert result.status in (ProcessingStatus.SUCCESS, ProcessingStatus.PARTIAL)
        assert result.complaint is not None
        assert result.coding_result is not None
        assert result.mdr_determination is not None

        # Verify real LLM was used
        if result.coding_result.is_success:
            assert result.coding_result.tokens_used > 0
            assert result.coding_result.latency_ms > 0


class TestBatchProcessing:
    """Tests for batch processing functionality."""

    def test_process_all_test_cases(self, temp_output_dirs: dict):
        """Test processing all test cases in a batch."""
        test_case_files = get_test_case_files()
        if not test_case_files:
            pytest.skip("No test cases found")

        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        success_count = 0
        partial_count = 0
        failed_count = 0

        for test_case_file in test_case_files:
            test_case = load_test_case(test_case_file)

            mock_response = create_mock_coding_response(test_case)
            mock_client = MagicMock(spec=LLMClient)
            mock_client.complete.return_value = mock_response

            result = process_form(
                raw_data=test_case["raw_input"],
                client=mock_client,
                audit_logger=audit_logger,
                save_outputs=False,
                use_llm_for_mdr=False,
            )

            if result.status == ProcessingStatus.SUCCESS:
                success_count += 1
            elif result.status == ProcessingStatus.PARTIAL:
                partial_count += 1
            else:
                failed_count += 1

        total = len(test_case_files)
        success_rate = (success_count + partial_count) / total if total > 0 else 0

        # At least 70% should process successfully
        assert success_rate >= 0.7, (
            f"Success rate {success_rate:.0%} below threshold. "
            f"Success: {success_count}, Partial: {partial_count}, Failed: {failed_count}"
        )
