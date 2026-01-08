"""Tests for the form processing pipeline.

These tests use mocked LLM responses to test the pipeline orchestration
without requiring actual Azure OpenAI calls.
"""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.audit.logger import AuditLogger
from src.intake.forms import FormValidationResult
from src.llm.client import LLMClient, LLMResponse, TokenUsage
from src.models.mdr import MDRCriteria
from src.pipeline import ProcessingStatus, process_form
from src.pipeline.forms import process_form_file
from src.pipeline.models import PipelineError, ProcessingResult


# Test fixtures
@pytest.fixture
def valid_form_data() -> dict:
    """Return valid form submission data."""
    return {
        "form_type": "online_complaint",
        "submission_date": "2024-01-15T14:30:00Z",
        "fields": {
            "device_name": "Test Device Model X",
            "manufacturer": "Test Manufacturer Inc.",
            "model_number": "TX-100",
            "serial_number": "SN-12345",
            "event_date": "2024-01-10",
            "event_description": (
                "The device malfunctioned during use, causing the patient "
                "to experience temporary discomfort. No serious injury occurred."
            ),
            "patient_outcome": "Temporary discomfort, no lasting effects",
            "device_returned": True,
            "reporter_type": "physician",
            "reporter_organization": "Test Hospital",
            "patient_age": 45,
            "patient_sex": "F",
        },
    }


@pytest.fixture
def death_form_data() -> dict:
    """Return form data with death outcome (requires MDR)."""
    return {
        "form_type": "online_complaint",
        "submission_date": "2024-01-15T14:30:00Z",
        "fields": {
            "device_name": "CardioRhythm Pacemaker",
            "manufacturer": "CardioRhythm Medical Inc.",
            "model_number": "CR-500",
            "serial_number": "CR500-2021-78432",
            "event_date": "2024-01-10",
            "event_description": (
                "Patient's pacemaker battery depleted earlier than expected. "
                "Patient experienced complete heart block and subsequently died."
            ),
            "patient_outcome": "Death",
            "device_returned": True,
            "reporter_type": "physician",
            "patient_age": 78,
            "patient_sex": "M",
        },
    }


@pytest.fixture
def incomplete_form_data() -> dict:
    """Return form data with missing required fields."""
    return {
        "form_type": "online_complaint",
        "submission_date": "2024-01-15T14:30:00Z",
        "fields": {
            "device_name": "Test Device",
            # Missing manufacturer and event_description
        },
    }


@pytest.fixture
def mock_llm_response() -> LLMResponse:
    """Return a mock LLM response for coding."""
    return LLMResponse(
        content=json.dumps({
            "suggestions": [
                {
                    "code_id": "A0601",
                    "confidence": 0.85,
                    "source_text": "device malfunctioned",
                    "reasoning": "Device malfunction is indicated",
                },
            ]
        }),
        model="gpt-4o",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        latency_ms=500.0,
    )


@pytest.fixture
def mock_llm_client(mock_llm_response: LLMResponse) -> MagicMock:
    """Return a mock LLM client."""
    client = MagicMock(spec=LLMClient)
    client.complete.return_value = mock_llm_response
    return client


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


class TestProcessingResult:
    """Tests for ProcessingResult model."""

    def test_processing_result_initialization(self):
        """Test ProcessingResult can be initialized with required fields."""
        result = ProcessingResult(
            status=ProcessingStatus.FAILED,
            processing_id="PROC-TEST-001",
        )
        assert result.status == ProcessingStatus.FAILED
        assert result.processing_id == "PROC-TEST-001"
        assert result.complaint is None
        assert result.coding_result is None
        assert result.mdr_determination is None
        assert result.errors == []

    def test_is_complete_false_when_missing_steps(self):
        """Test is_complete is False when steps are missing."""
        result = ProcessingResult(
            status=ProcessingStatus.PARTIAL,
            processing_id="PROC-TEST-002",
            form_valid=True,
        )
        assert not result.is_complete

    def test_add_error(self):
        """Test adding errors to result."""
        result = ProcessingResult(
            status=ProcessingStatus.FAILED,
            processing_id="PROC-TEST-003",
        )
        result.add_error(
            step="test_step",
            message="Test error message",
            details={"key": "value"},
        )
        assert len(result.errors) == 1
        assert result.errors[0]["step"] == "test_step"
        assert result.errors[0]["message"] == "Test error message"
        assert result.errors[0]["details"]["key"] == "value"

    def test_summary(self):
        """Test summary generation."""
        result = ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            processing_id="PROC-TEST-004",
            form_valid=True,
        )
        summary = result.summary()
        assert summary["processing_id"] == "PROC-TEST-004"
        assert summary["status"] == "success"
        assert summary["form_valid"] is True

    def test_processing_duration(self):
        """Test processing duration calculation."""
        result = ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            processing_id="PROC-TEST-005",
        )
        result.started_at = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        result.completed_at = datetime(2024, 1, 15, 10, 0, 1, tzinfo=UTC)

        assert result.processing_duration_ms == 1000.0


class TestPipelineError:
    """Tests for PipelineError exception."""

    def test_pipeline_error_creation(self):
        """Test PipelineError can be created with step and details."""
        error = PipelineError(
            message="Test error",
            step="parse_form",
            details={"field": "device_name"},
        )
        assert str(error) == "Test error"
        assert error.step == "parse_form"
        assert error.details["field"] == "device_name"

    def test_pipeline_error_default_details(self):
        """Test PipelineError has empty details by default."""
        error = PipelineError(message="Test", step="test")
        assert error.details == {}


class TestProcessForm:
    """Tests for the process_form function."""

    def test_process_valid_form(
        self,
        valid_form_data: dict,
        mock_llm_client: MagicMock,
        temp_output_dirs: dict,
    ):
        """Test processing a valid form submission."""
        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        result = process_form(
            raw_data=valid_form_data,
            client=mock_llm_client,
            audit_logger=audit_logger,
            complaints_dir=temp_output_dirs["complaints"],
            decisions_dir=temp_output_dirs["decisions"],
            save_outputs=True,
            use_llm_for_mdr=False,  # Use rules-only for predictable testing
        )

        assert result.status in (ProcessingStatus.SUCCESS, ProcessingStatus.PARTIAL)
        assert result.form_valid is True
        assert result.complaint is not None
        assert result.complaint.complaint_id is not None
        assert result.complaint.device_info.device_name == "Test Device Model X"
        assert len(result.audit_event_ids) >= 1

    def test_process_form_with_death_mdr(
        self,
        death_form_data: dict,
        mock_llm_client: MagicMock,
        temp_output_dirs: dict,
    ):
        """Test processing a form that requires MDR due to death."""
        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        result = process_form(
            raw_data=death_form_data,
            client=mock_llm_client,
            audit_logger=audit_logger,
            complaints_dir=temp_output_dirs["complaints"],
            decisions_dir=temp_output_dirs["decisions"],
            save_outputs=False,
            use_llm_for_mdr=False,
        )

        assert result.complaint is not None
        assert result.mdr_determination is not None
        assert result.mdr_determination.requires_mdr is True
        assert MDRCriteria.DEATH in result.mdr_determination.mdr_criteria_met

    def test_process_incomplete_form(
        self,
        incomplete_form_data: dict,
        mock_llm_client: MagicMock,
        temp_output_dirs: dict,
    ):
        """Test processing an incomplete form fails gracefully."""
        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        result = process_form(
            raw_data=incomplete_form_data,
            client=mock_llm_client,
            audit_logger=audit_logger,
            complaints_dir=temp_output_dirs["complaints"],
            decisions_dir=temp_output_dirs["decisions"],
            save_outputs=False,
        )

        # Should fail due to missing required fields
        assert result.status == ProcessingStatus.FAILED
        assert result.form_valid is False
        assert len(result.errors) > 0

    def test_process_form_missing_submission_date(
        self,
        mock_llm_client: MagicMock,
        temp_output_dirs: dict,
    ):
        """Test processing fails when submission_date is missing."""
        invalid_data = {
            "form_type": "online_complaint",
            # Missing submission_date
            "fields": {
                "device_name": "Test Device",
                "manufacturer": "Test Manufacturer",
                "event_description": "Test description",
            },
        }

        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        result = process_form(
            raw_data=invalid_data,
            client=mock_llm_client,
            audit_logger=audit_logger,
            save_outputs=False,
        )

        assert result.status == ProcessingStatus.FAILED
        assert len(result.errors) > 0
        assert "submission_date" in result.errors[0]["message"].lower()

    def test_process_form_saves_files(
        self,
        valid_form_data: dict,
        mock_llm_client: MagicMock,
        temp_output_dirs: dict,
    ):
        """Test that process_form saves complaint and decision files."""
        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        # Run process_form - result is implicitly validated by file checks below
        process_form(
            raw_data=valid_form_data,
            client=mock_llm_client,
            audit_logger=audit_logger,
            complaints_dir=temp_output_dirs["complaints"],
            decisions_dir=temp_output_dirs["decisions"],
            save_outputs=True,
            use_llm_for_mdr=False,
        )

        # Check complaint file was saved
        complaint_files = list(temp_output_dirs["complaints"].glob("*.json"))
        assert len(complaint_files) >= 1

        # Check decision file was saved
        decision_files = list(temp_output_dirs["decisions"].glob("*_decision.json"))
        assert len(decision_files) >= 1

    def test_process_form_logs_audit_events(
        self,
        valid_form_data: dict,
        mock_llm_client: MagicMock,
        temp_output_dirs: dict,
    ):
        """Test that process_form logs audit events."""
        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        result = process_form(
            raw_data=valid_form_data,
            client=mock_llm_client,
            audit_logger=audit_logger,
            complaints_dir=temp_output_dirs["complaints"],
            decisions_dir=temp_output_dirs["decisions"],
            save_outputs=False,
            use_llm_for_mdr=False,
        )

        # Should have at least complaint_created and mdr_determined events
        assert len(result.audit_event_ids) >= 2

        # Verify audit files were created
        audit_files = list(temp_output_dirs["audit"].glob("*.jsonl"))
        assert len(audit_files) >= 1


class TestProcessFormFile:
    """Tests for the process_form_file function."""

    def test_process_form_file(
        self,
        valid_form_data: dict,
        mock_llm_client: MagicMock,
        temp_output_dirs: dict,
    ):
        """Test processing a form from a file."""
        # Create a temp file with form data
        form_file = temp_output_dirs["complaints"] / "test_form.json"
        with open(form_file, "w") as f:
            json.dump(valid_form_data, f)

        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        result = process_form_file(
            file_path=form_file,
            client=mock_llm_client,
            audit_logger=audit_logger,
            save_outputs=False,
        )

        assert result.status in (ProcessingStatus.SUCCESS, ProcessingStatus.PARTIAL)
        assert result.complaint is not None

    def test_process_form_file_test_case_format(
        self,
        mock_llm_client: MagicMock,
        temp_output_dirs: dict,
    ):
        """Test processing a file in test case format (with raw_input)."""
        test_case_data = {
            "test_case_id": "FORM-TEST",
            "name": "Test Case",
            "raw_input": {
                "form_type": "online_complaint",
                "submission_date": "2024-01-15T14:30:00Z",
                "fields": {
                    "device_name": "Test Device",
                    "manufacturer": "Test Manufacturer",
                    "event_description": "Test description",
                },
            },
        }

        form_file = temp_output_dirs["complaints"] / "test_case.json"
        with open(form_file, "w") as f:
            json.dump(test_case_data, f)

        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        result = process_form_file(
            file_path=form_file,
            client=mock_llm_client,
            audit_logger=audit_logger,
            save_outputs=False,
        )

        assert result.complaint is not None
        assert result.complaint.device_info.device_name == "Test Device"

    def test_process_form_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            process_form_file(file_path=Path("/nonexistent/file.json"))


class TestProcessFormIntegration:
    """Integration tests for the pipeline with mocked LLM."""

    def test_full_pipeline_flow(
        self,
        valid_form_data: dict,
        temp_output_dirs: dict,
    ):
        """Test the complete pipeline flow with all steps."""
        # Create a more comprehensive mock response
        mock_response = LLMResponse(
            content=json.dumps({
                "suggestions": [
                    {
                        "code_id": "A0601",
                        "confidence": 0.85,
                        "source_text": "device malfunctioned",
                        "reasoning": "Malfunction indicated in narrative",
                    },
                    {
                        "code_id": "C0601",
                        "confidence": 0.75,
                        "source_text": "temporary discomfort",
                        "reasoning": "Patient experienced discomfort",
                    },
                ]
            }),
            model="gpt-4o",
            usage=TokenUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300),
            latency_ms=750.0,
        )

        mock_client = MagicMock(spec=LLMClient)
        mock_client.complete.return_value = mock_response

        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        result = process_form(
            raw_data=valid_form_data,
            client=mock_client,
            audit_logger=audit_logger,
            complaints_dir=temp_output_dirs["complaints"],
            decisions_dir=temp_output_dirs["decisions"],
            save_outputs=True,
            use_llm_for_mdr=False,
        )

        # Verify all steps completed
        assert result.form_valid is True
        assert result.complaint is not None
        assert result.mdr_determination is not None
        assert result.completed_at is not None
        assert result.processing_duration_ms is not None
        assert result.processing_duration_ms > 0

    def test_pipeline_continues_after_llm_error(
        self,
        valid_form_data: dict,
        temp_output_dirs: dict,
    ):
        """Test pipeline continues MDR determination if LLM coding fails."""
        # Create a mock that raises an error
        mock_client = MagicMock(spec=LLMClient)
        mock_client.complete.side_effect = Exception("LLM connection failed")

        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        result = process_form(
            raw_data=valid_form_data,
            client=mock_client,
            audit_logger=audit_logger,
            save_outputs=False,
            use_llm_for_mdr=False,  # Use rules-only to avoid LLM for MDR
        )

        # Should still have complaint and MDR (rules-based)
        assert result.complaint is not None
        assert result.mdr_determination is not None
        # Should have error recorded
        coding_errors = [e for e in result.errors if e["step"] == "suggest_codes"]
        assert len(coding_errors) >= 1


class TestProcessFormValidation:
    """Tests focused on form validation during pipeline processing."""

    def test_validation_result_included(
        self,
        valid_form_data: dict,
        mock_llm_client: MagicMock,
        temp_output_dirs: dict,
    ):
        """Test that validation result is included in processing result."""
        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        result = process_form(
            raw_data=valid_form_data,
            client=mock_llm_client,
            audit_logger=audit_logger,
            save_outputs=False,
        )

        assert result.validation_result is not None
        assert isinstance(result.validation_result, FormValidationResult)

    def test_missing_recommended_fields_still_processes(
        self,
        mock_llm_client: MagicMock,
        temp_output_dirs: dict,
    ):
        """Test that missing recommended fields don't block processing."""
        minimal_valid_data = {
            "form_type": "online_complaint",
            "submission_date": "2024-01-15T14:30:00Z",
            "fields": {
                "device_name": "Test Device",
                "manufacturer": "Test Manufacturer",
                "event_description": "Device malfunction occurred",
                # Missing recommended: event_date, model_number, serial_number, etc.
            },
        }

        audit_logger = AuditLogger(log_dir=temp_output_dirs["audit"])

        result = process_form(
            raw_data=minimal_valid_data,
            client=mock_llm_client,
            audit_logger=audit_logger,
            save_outputs=False,
            use_llm_for_mdr=False,
        )

        assert result.form_valid is True
        assert result.complaint is not None
        # Should have warnings about missing recommended fields
        assert result.validation_result is not None
        assert len(result.validation_result.missing_recommended) > 0
