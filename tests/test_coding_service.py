"""Tests for IMDRF coding service."""

import json
from datetime import UTC, date, datetime
from unittest.mock import MagicMock

from src.coding import CodingResult, CodingService, suggest_codes
from src.llm import LLMResponse, TokenUsage
from src.models.coding import CodingSuggestion
from src.models.complaint import ComplaintRecord, DeviceInfo, EventInfo
from src.models.enums import ComplaintStatus, DeviceType, IMDRFCodeType, IntakeChannel


def create_test_complaint(
    complaint_id: str = "TEST-001",
    device_name: str = "Test Device",
    manufacturer: str = "Test Manufacturer",
    narrative: str = "The device malfunctioned during use.",
    patient_outcome: str | None = None,
) -> ComplaintRecord:
    """Create a test complaint record."""
    return ComplaintRecord(
        complaint_id=complaint_id,
        intake_channel=IntakeChannel.FORM,
        received_date=datetime.now(UTC),
        status=ComplaintStatus.NEW,
        device_info=DeviceInfo(
            device_name=device_name,
            manufacturer=manufacturer,
            device_type=DeviceType.DIAGNOSTIC,
        ),
        event_info=EventInfo(
            event_date=date(2024, 1, 15),
            event_description=narrative,
            patient_outcome=patient_outcome,
        ),
        narrative=narrative,
    )


def create_mock_llm_response(
    suggestions: list[dict],
    model: str = "gpt-4o",
    tokens: int = 500,
    latency_ms: float = 150.0,
) -> LLMResponse:
    """Create a mock LLM response with suggestions."""
    content = json.dumps({"suggestions": suggestions})
    return LLMResponse(
        content=content,
        model=model,
        usage=TokenUsage(
            prompt_tokens=tokens // 2,
            completion_tokens=tokens // 2,
            total_tokens=tokens,
        ),
        finish_reason="stop",
        latency_ms=latency_ms,
    )


class TestCodingResult:
    """Tests for CodingResult model."""

    def test_create_result(self) -> None:
        """Create a coding result."""
        result = CodingResult(
            complaint_id="TEST-001",
            suggestions=[],
        )

        assert result.complaint_id == "TEST-001"
        assert result.is_success is True
        assert len(result.suggestions) == 0

    def test_result_with_error(self) -> None:
        """Result with error is not successful."""
        result = CodingResult(
            complaint_id="TEST-001",
            error="Something went wrong",
        )

        assert result.is_success is False
        assert result.error == "Something went wrong"

    def test_filter_by_code_type(self) -> None:
        """Filter suggestions by code type."""
        result = CodingResult(
            complaint_id="TEST-001",
            suggestions=[
                CodingSuggestion(
                    code_id="A01",
                    code_name="Device Problem",
                    code_type=IMDRFCodeType.DEVICE_PROBLEM,
                    confidence=0.9,
                    source_text="text",
                    reasoning="reason",
                ),
                CodingSuggestion(
                    code_id="C01",
                    code_name="Patient Problem",
                    code_type=IMDRFCodeType.PATIENT_PROBLEM,
                    confidence=0.8,
                    source_text="text",
                    reasoning="reason",
                ),
            ],
        )

        assert len(result.device_problem_codes) == 1
        assert len(result.patient_problem_codes) == 1
        assert result.device_problem_codes[0].code_id == "A01"
        assert result.patient_problem_codes[0].code_id == "C01"


class TestCodingService:
    """Tests for CodingService."""

    def test_suggest_codes_success(self) -> None:
        """Successfully suggest codes for a complaint."""
        # Create mock client
        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_llm_response(
            [
                {
                    "code_id": "A01",
                    "code_name": "Adverse Event Without Identified Device or Use Problem",
                    "confidence": 0.85,
                    "source_text": "device malfunctioned",
                    "reasoning": "The complaint describes a device malfunction.",
                },
            ]
        )

        # Create service and suggest codes
        service = CodingService(client=mock_client)
        complaint = create_test_complaint()

        result = service.suggest_codes(complaint)

        assert result.is_success
        assert len(result.suggestions) == 1
        assert result.suggestions[0].code_id == "A01"
        assert result.suggestions[0].confidence == 0.85
        assert result.model_used == "gpt-4o"
        assert result.tokens_used == 500

    def test_suggest_codes_filters_low_confidence(self) -> None:
        """Low confidence suggestions are filtered out."""
        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_llm_response(
            [
                {
                    "code_id": "A01",
                    "confidence": 0.9,
                    "source_text": "text",
                    "reasoning": "reason",
                },
                {
                    "code_id": "A02",
                    "confidence": 0.1,  # Below default threshold of 0.3
                    "source_text": "text",
                    "reasoning": "reason",
                },
            ]
        )

        service = CodingService(client=mock_client, min_confidence=0.3)
        complaint = create_test_complaint()

        result = service.suggest_codes(complaint)

        assert result.is_success
        assert len(result.suggestions) == 1
        assert result.suggestions[0].code_id == "A01"

    def test_suggest_codes_validates_against_imdrf(self) -> None:
        """Invalid IMDRF codes are filtered out."""
        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_llm_response(
            [
                {
                    "code_id": "A01",  # Valid code
                    "confidence": 0.9,
                    "source_text": "text",
                    "reasoning": "reason",
                },
                {
                    "code_id": "INVALID_CODE",  # Not a real IMDRF code
                    "confidence": 0.8,
                    "source_text": "text",
                    "reasoning": "reason",
                },
            ]
        )

        service = CodingService(client=mock_client)
        complaint = create_test_complaint()

        result = service.suggest_codes(complaint)

        assert result.is_success
        assert len(result.suggestions) == 1
        assert result.suggestions[0].code_id == "A01"

    def test_suggest_codes_limits_count(self) -> None:
        """Number of suggestions is limited."""
        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_llm_response(
            [
                {
                    "code_id": "A01",
                    "confidence": 0.9,
                    "source_text": "t",
                    "reasoning": "r",
                },
                {
                    "code_id": "A02",
                    "confidence": 0.85,
                    "source_text": "t",
                    "reasoning": "r",
                },
                {
                    "code_id": "A03",
                    "confidence": 0.8,
                    "source_text": "t",
                    "reasoning": "r",
                },
                {
                    "code_id": "A04",
                    "confidence": 0.75,
                    "source_text": "t",
                    "reasoning": "r",
                },
            ]
        )

        service = CodingService(client=mock_client, max_suggestions=2)
        complaint = create_test_complaint()

        result = service.suggest_codes(complaint)

        assert result.is_success
        assert len(result.suggestions) == 2
        # Should be sorted by confidence
        assert result.suggestions[0].code_id == "A01"
        assert result.suggestions[1].code_id == "A02"

    def test_suggest_codes_handles_parse_error(self) -> None:
        """Parse errors are handled gracefully."""
        mock_client = MagicMock()
        mock_client.complete.return_value = LLMResponse(
            content="This is not valid JSON",
            model="gpt-4o",
            usage=TokenUsage(total_tokens=100),
            latency_ms=100.0,
        )

        service = CodingService(client=mock_client)
        complaint = create_test_complaint()

        result = service.suggest_codes(complaint)

        assert not result.is_success
        assert "parse" in result.error.lower()

    def test_suggest_codes_handles_llm_error(self) -> None:
        """LLM errors are handled gracefully."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("LLM service unavailable")

        service = CodingService(client=mock_client)
        complaint = create_test_complaint()

        result = service.suggest_codes(complaint)

        assert not result.is_success
        assert "LLM service unavailable" in result.error

    def test_suggest_codes_empty_suggestions(self) -> None:
        """Empty suggestions list is handled."""
        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_llm_response([])

        service = CodingService(client=mock_client)
        complaint = create_test_complaint()

        result = service.suggest_codes(complaint)

        assert result.is_success
        assert len(result.suggestions) == 0

    def test_suggest_codes_adds_full_path(self) -> None:
        """Full path is added to suggestions."""
        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_llm_response(
            [
                {
                    "code_id": "A01",
                    "confidence": 0.9,
                    "source_text": "text",
                    "reasoning": "reason",
                },
            ]
        )

        service = CodingService(client=mock_client)
        complaint = create_test_complaint()

        result = service.suggest_codes(complaint)

        assert result.is_success
        assert result.suggestions[0].full_path is not None

    def test_suggest_codes_normalizes_confidence(self) -> None:
        """Confidence values outside [0, 1] are normalized."""
        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_llm_response(
            [
                {
                    "code_id": "A01",
                    "confidence": 1.5,  # Above 1.0
                    "source_text": "text",
                    "reasoning": "reason",
                },
                {
                    "code_id": "A02",
                    "confidence": -0.5,  # Below 0.0
                    "source_text": "text",
                    "reasoning": "reason",
                },
            ]
        )

        service = CodingService(client=mock_client, min_confidence=0.0)
        complaint = create_test_complaint()

        result = service.suggest_codes(complaint)

        # A01 should have confidence capped at 1.0
        assert result.suggestions[0].confidence == 1.0
        # A02 should have confidence capped at 0.0
        assert result.suggestions[1].confidence == 0.0


class TestSuggestCodesFunction:
    """Tests for the suggest_codes convenience function."""

    def test_suggest_codes_function(self) -> None:
        """suggest_codes function works correctly."""
        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_llm_response(
            [
                {
                    "code_id": "A01",
                    "confidence": 0.9,
                    "source_text": "text",
                    "reasoning": "reason",
                },
            ]
        )

        complaint = create_test_complaint()
        result = suggest_codes(complaint, client=mock_client)

        assert result.is_success
        assert len(result.suggestions) == 1

    def test_suggest_codes_with_custom_thresholds(self) -> None:
        """Custom thresholds can be passed to suggest_codes."""
        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_llm_response(
            [
                {
                    "code_id": "A01",
                    "confidence": 0.6,
                    "source_text": "t",
                    "reasoning": "r",
                },
                {
                    "code_id": "A02",
                    "confidence": 0.4,
                    "source_text": "t",
                    "reasoning": "r",
                },
            ]
        )

        complaint = create_test_complaint()
        result = suggest_codes(
            complaint,
            client=mock_client,
            min_confidence=0.5,
            max_suggestions=5,
        )

        assert result.is_success
        assert len(result.suggestions) == 1  # Only A01 passes 0.5 threshold


class TestEdgeCases:
    """Tests for edge cases in coding service."""

    def test_malformed_suggestion_skipped(self) -> None:
        """Malformed suggestions are skipped."""
        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_llm_response(
            [
                {
                    "code_id": "A01",
                    "confidence": 0.9,
                    "source_text": "text",
                    "reasoning": "reason",
                },
                {
                    # Missing code_id
                    "confidence": 0.8,
                    "source_text": "text",
                },
                {
                    "code_id": "",  # Empty code_id
                    "confidence": 0.7,
                },
            ]
        )

        service = CodingService(client=mock_client)
        complaint = create_test_complaint()

        result = service.suggest_codes(complaint)

        assert result.is_success
        assert len(result.suggestions) == 1
        assert result.suggestions[0].code_id == "A01"

    def test_non_dict_suggestions_skipped(self) -> None:
        """Non-dict suggestions in list are skipped."""
        mock_client = MagicMock()
        mock_client.complete.return_value = LLMResponse(
            content=json.dumps(
                {
                    "suggestions": [
                        {
                            "code_id": "A01",
                            "confidence": 0.9,
                            "source_text": "t",
                            "reasoning": "r",
                        },
                        "not a dict",
                        123,
                        None,
                    ]
                }
            ),
            model="gpt-4o",
            usage=TokenUsage(total_tokens=100),
            latency_ms=100.0,
        )

        service = CodingService(client=mock_client)
        complaint = create_test_complaint()

        result = service.suggest_codes(complaint)

        assert result.is_success
        assert len(result.suggestions) == 1

    def test_confidence_type_coercion(self) -> None:
        """Non-numeric confidence values default to 0.5."""
        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_llm_response(
            [
                {
                    "code_id": "A01",
                    "confidence": "high",  # String instead of number
                    "source_text": "text",
                    "reasoning": "reason",
                },
            ]
        )

        service = CodingService(client=mock_client)
        complaint = create_test_complaint()

        result = service.suggest_codes(complaint)

        assert result.is_success
        assert len(result.suggestions) == 1
        assert result.suggestions[0].confidence == 0.5  # Default value

    def test_missing_suggestions_key(self) -> None:
        """Missing suggestions key in response handled."""
        mock_client = MagicMock()
        mock_client.complete.return_value = LLMResponse(
            content=json.dumps({"other_key": "value"}),
            model="gpt-4o",
            usage=TokenUsage(total_tokens=100),
            latency_ms=100.0,
        )

        service = CodingService(client=mock_client)
        complaint = create_test_complaint()

        result = service.suggest_codes(complaint)

        assert result.is_success
        assert len(result.suggestions) == 0

    def test_complaint_with_patient_outcome(self) -> None:
        """Complaints with patient outcome are processed correctly."""
        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_llm_response(
            [
                {
                    "code_id": "C01",  # Patient problem code
                    "confidence": 0.95,
                    "source_text": "patient died",
                    "reasoning": "Death is clearly stated.",
                },
            ]
        )

        service = CodingService(client=mock_client)
        complaint = create_test_complaint(
            narrative="The device failed and the patient died.",
            patient_outcome="Death",
        )

        result = service.suggest_codes(complaint)

        assert result.is_success
        assert len(result.suggestions) == 1
        assert result.suggestions[0].code_type == IMDRFCodeType.PATIENT_PROBLEM
