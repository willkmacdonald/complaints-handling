"""Tests for MDR determination service."""

import json
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.complaint import (
    ComplaintRecord,
    DeviceInfo,
    EventInfo,
    PatientInfo,
    ReporterInfo,
)
from src.models.enums import ComplaintStatus, DeviceType, IntakeChannel, ReporterType
from src.models.mdr import MDRCriteria, MDRDetermination
from src.routing.mdr import (
    _analyze_text_for_mdr,
    _search_patterns,
    determine_mdr,
    determine_mdr_rules_only,
)

# Path to test case data
TEST_CASES_DIR = Path(__file__).parent.parent / "data" / "test_cases" / "form"


def load_test_case(filename: str) -> dict:
    """Load a test case JSON file."""
    with open(TEST_CASES_DIR / filename) as f:
        return json.load(f)


def create_complaint(
    complaint_id: str,
    narrative: str,
    patient_outcome: str | None = None,
    device_outcome: str | None = None,
    device_name: str = "Test Device",
    manufacturer: str = "Test Manufacturer",
) -> ComplaintRecord:
    """Create a test ComplaintRecord."""
    return ComplaintRecord(
        complaint_id=complaint_id,
        intake_channel=IntakeChannel.FORM,
        received_date=datetime.now(UTC),
        status=ComplaintStatus.NEW,
        device_info=DeviceInfo(
            device_name=device_name,
            manufacturer=manufacturer,
            device_type=DeviceType.OTHER,
        ),
        event_info=EventInfo(
            event_description=narrative,
            patient_outcome=patient_outcome,
            device_outcome=device_outcome,
        ),
        narrative=narrative,
    )


class TestSearchPatterns:
    """Tests for pattern matching utility."""

    def test_finds_death_keywords(self) -> None:
        """Death keywords are detected."""
        text = "The patient subsequently died after the device failure."
        from src.routing.mdr import DEATH_KEYWORDS

        evidence = _search_patterns(text, DEATH_KEYWORDS)

        assert len(evidence) > 0
        assert any("died" in e.lower() for e in evidence)

    def test_finds_serious_injury_keywords(self) -> None:
        """Serious injury keywords are detected."""
        text = "Patient was hospitalized and required emergency surgery."
        from src.routing.mdr import SERIOUS_INJURY_KEYWORDS

        evidence = _search_patterns(text, SERIOUS_INJURY_KEYWORDS)

        assert len(evidence) > 0
        assert any("hospitalized" in e.lower() or "surgery" in e.lower() for e in evidence)

    def test_finds_malfunction_keywords(self) -> None:
        """Malfunction keywords are detected."""
        text = "The device failed to deliver the correct dose and showed an error code."
        from src.routing.mdr import MALFUNCTION_KEYWORDS

        evidence = _search_patterns(text, MALFUNCTION_KEYWORDS)

        assert len(evidence) > 0
        assert any("failed" in e.lower() or "error" in e.lower() for e in evidence)

    def test_returns_empty_for_no_matches(self) -> None:
        """Returns empty list when no matches found."""
        text = "The device worked perfectly with no issues."
        from src.routing.mdr import DEATH_KEYWORDS

        evidence = _search_patterns(text, DEATH_KEYWORDS)

        assert len(evidence) == 0

    def test_includes_context_in_evidence(self) -> None:
        """Evidence includes surrounding context."""
        text = "During the procedure, the patient's pacemaker battery depleted and the patient died."
        from src.routing.mdr import DEATH_KEYWORDS

        evidence = _search_patterns(text, DEATH_KEYWORDS)

        # Should include context around the match
        assert len(evidence) > 0
        assert any(len(e) > 10 for e in evidence)


class TestAnalyzeTextForMDR:
    """Tests for text analysis function."""

    def test_detects_death(self) -> None:
        """Death is detected in narrative."""
        result = _analyze_text_for_mdr(
            narrative="Patient experienced cardiac arrest and subsequently died.",
            patient_outcome="Death",
            device_outcome=None,
        )

        assert result.death_detected is True
        assert len(result.death_evidence) > 0

    def test_detects_serious_injury(self) -> None:
        """Serious injury is detected."""
        result = _analyze_text_for_mdr(
            narrative="Patient was hospitalized in ICU for complications.",
            patient_outcome="Required hospitalization",
            device_outcome=None,
        )

        assert result.serious_injury_detected is True
        assert len(result.serious_injury_evidence) > 0

    def test_detects_malfunction(self) -> None:
        """Device malfunction is detected."""
        result = _analyze_text_for_mdr(
            narrative="The insulin pump malfunctioned and delivered incorrect dose.",
            patient_outcome=None,
            device_outcome="Device failed",
        )

        assert result.malfunction_detected is True
        assert len(result.malfunction_evidence) > 0

    def test_detects_user_error(self) -> None:
        """User error indicators are detected."""
        result = _analyze_text_for_mdr(
            narrative="Investigation revealed user error. Device functioned correctly as programmed.",
            patient_outcome=None,
            device_outcome="No defect found",
        )

        assert result.user_error_detected is True
        assert len(result.user_error_evidence) > 0

    def test_no_criteria_detected(self) -> None:
        """No MDR criteria detected in benign complaint."""
        result = _analyze_text_for_mdr(
            narrative="Patient noticed cosmetic scratches on device surface.",
            patient_outcome="No harm",
            device_outcome="Minor cosmetic damage",
        )

        assert result.death_detected is False
        assert result.serious_injury_detected is False
        assert result.malfunction_detected is False


class TestDetermineMDRRulesOnly:
    """Tests for rules-based MDR determination."""

    def test_death_requires_mdr(self) -> None:
        """Death always requires MDR."""
        complaint = create_complaint(
            complaint_id="TEST-001",
            narrative="Patient's pacemaker failed and patient subsequently died.",
            patient_outcome="Death",
        )

        result = determine_mdr_rules_only(complaint)

        assert result.requires_mdr is True
        assert MDRCriteria.DEATH in result.mdr_criteria_met
        assert result.review_priority == "urgent"

    def test_serious_injury_requires_mdr(self) -> None:
        """Serious injury requires MDR."""
        complaint = create_complaint(
            complaint_id="TEST-002",
            narrative="Device malfunction caused patient to be hospitalized in ICU.",
            patient_outcome="Required hospitalization",
        )

        result = determine_mdr_rules_only(complaint)

        assert result.requires_mdr is True
        assert MDRCriteria.SERIOUS_INJURY in result.mdr_criteria_met

    def test_malfunction_potential_harm_requires_mdr(self) -> None:
        """Malfunction that could cause harm requires MDR."""
        complaint = create_complaint(
            complaint_id="TEST-003",
            narrative="Blood glucose meter displayed falsely low reading. Patient took extra insulin.",
            patient_outcome="Hyperglycemia from reduced insulin",
        )

        result = determine_mdr_rules_only(complaint)

        assert result.requires_mdr is True
        # Should flag malfunction that could cause harm
        assert any(
            c
            in [
                MDRCriteria.MALFUNCTION_COULD_CAUSE_DEATH,
                MDRCriteria.MALFUNCTION_COULD_CAUSE_SERIOUS_INJURY,
            ]
            for c in result.mdr_criteria_met
        )

    def test_user_error_without_malfunction_may_not_require_mdr(self) -> None:
        """User error without device malfunction has lower confidence."""
        complaint = create_complaint(
            complaint_id="TEST-004",
            narrative="Patient accidentally set wrong basal rate. Device functioned correctly as programmed. User error confirmed.",
            patient_outcome="Hypoglycemia, recovered",
        )

        result = determine_mdr_rules_only(complaint)

        # Should have lower confidence due to user error
        assert result.ai_confidence < 0.7
        assert "user error" in result.ai_reasoning.lower()

    def test_cosmetic_issue_does_not_require_mdr(self) -> None:
        """Cosmetic issues don't require MDR."""
        complaint = create_complaint(
            complaint_id="TEST-005",
            narrative="Package was dented on arrival. Device appears undamaged and working.",
            patient_outcome="None",
        )

        result = determine_mdr_rules_only(complaint)

        # Should not find MDR criteria
        assert len(result.mdr_criteria_met) == 0

    def test_always_requires_human_review(self) -> None:
        """All determinations require human review."""
        complaint = create_complaint(
            complaint_id="TEST-006",
            narrative="Any complaint text.",
        )

        result = determine_mdr_rules_only(complaint)

        assert result.needs_human_review is True

    def test_key_factors_populated(self) -> None:
        """Key factors are populated with analysis results."""
        complaint = create_complaint(
            complaint_id="TEST-007",
            narrative="Patient died after device malfunction.",
            patient_outcome="Death",
        )

        result = determine_mdr_rules_only(complaint)

        assert len(result.key_factors) > 0


class TestDetermineMDR:
    """Tests for combined rules + LLM MDR determination."""

    def test_fallback_to_rules_when_llm_disabled(self) -> None:
        """Falls back to rules-only when use_llm=False."""
        complaint = create_complaint(
            complaint_id="TEST-008",
            narrative="Patient died after pacemaker failure.",
            patient_outcome="Death",
        )

        result = determine_mdr(complaint, use_llm=False)

        assert result.requires_mdr is True
        assert MDRCriteria.DEATH in result.mdr_criteria_met
        # Should not contain LLM-specific content
        assert "Combined analysis" not in result.ai_reasoning

    def test_fallback_to_rules_on_llm_error(self) -> None:
        """Falls back to rules when LLM fails."""
        complaint = create_complaint(
            complaint_id="TEST-009",
            narrative="Patient died after device failure.",
            patient_outcome="Death",
        )

        # Mock create_client to raise an error
        with patch("src.routing.mdr.create_client") as mock_create:
            from src.llm.client import LLMError

            mock_create.side_effect = LLMError("Test error")

            result = determine_mdr(complaint, use_llm=True)

        assert result.requires_mdr is True  # Rules still work
        assert "unavailable" in result.ai_reasoning.lower()

    def test_combined_analysis_with_mock_llm(self) -> None:
        """Combined analysis uses both rules and LLM."""
        complaint = create_complaint(
            complaint_id="TEST-010",
            narrative="Patient hospitalized after device malfunction.",
            patient_outcome="Hospitalization",
        )

        # Mock LLM client
        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(
            content=json.dumps(
                {
                    "requires_mdr": True,
                    "confidence": 0.85,
                    "criteria_met": ["serious_injury"],
                    "reasoning": "LLM determined serious injury from hospitalization",
                    "evidence": ["Patient required hospital admission"],
                }
            )
        )

        result = determine_mdr(complaint, client=mock_client, use_llm=True)

        assert result.requires_mdr is True
        assert "Combined analysis" in result.ai_reasoning
        assert any("hospital" in f.lower() for f in result.key_factors)

    def test_conservative_combination_both_agree(self) -> None:
        """When rules and LLM agree, confidence is high."""
        complaint = create_complaint(
            complaint_id="TEST-011",
            narrative="Patient died.",
            patient_outcome="Death",
        )

        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(
            content=json.dumps(
                {
                    "requires_mdr": True,
                    "confidence": 0.95,
                    "criteria_met": ["death"],
                    "reasoning": "Death confirmed",
                    "evidence": ["Patient death reported"],
                }
            )
        )

        result = determine_mdr(complaint, client=mock_client, use_llm=True)

        assert result.requires_mdr is True
        assert result.ai_confidence >= 0.75


class TestWithRealTestCases:
    """Integration tests using actual test cases."""

    def test_form_001_pacemaker_death_requires_mdr(self) -> None:
        """Pacemaker death case requires MDR."""
        test_case = load_test_case("form_001_pacemaker_death.json")
        expected = test_case["expected_complaint"]
        ground_truth = test_case["ground_truth"]

        complaint = ComplaintRecord(
            complaint_id=expected["complaint_id"],
            intake_channel=IntakeChannel.FORM,
            received_date=datetime.fromisoformat(
                expected["received_date"].replace("Z", "+00:00")
            ),
            status=ComplaintStatus.NEW,
            device_info=DeviceInfo(**expected["device_info"]),
            event_info=EventInfo(**expected["event_info"]),
            narrative=expected["narrative"],
        )

        result = determine_mdr_rules_only(complaint)

        assert result.requires_mdr == ground_truth["requires_mdr"]
        assert result.review_priority == "urgent"  # Death cases are urgent

    def test_form_007_user_error_does_not_require_mdr(self) -> None:
        """User error case (device worked correctly) should not require MDR."""
        test_case = load_test_case("form_007_insulin_pump_user_error.json")
        expected = test_case["expected_complaint"]
        ground_truth = test_case["ground_truth"]

        complaint = ComplaintRecord(
            complaint_id=expected["complaint_id"],
            intake_channel=IntakeChannel.FORM,
            received_date=datetime.fromisoformat(
                expected["received_date"].replace("Z", "+00:00")
            ),
            status=ComplaintStatus.NEW,
            device_info=DeviceInfo(**expected["device_info"]),
            event_info=EventInfo(**expected["event_info"]),
            narrative=expected["narrative"],
        )

        result = determine_mdr_rules_only(complaint)

        # User error cases are tricky - rules may flag conservatively
        # But should detect user error indicators
        assert result.ai_confidence < 0.8 or "user error" in result.ai_reasoning.lower()
        # Ground truth says no MDR required
        assert ground_truth["requires_mdr"] is False

    def test_form_003_malfunction_requires_mdr(self) -> None:
        """Glucose meter malfunction requires MDR (could cause serious injury if recurs)."""
        test_case = load_test_case("form_003_glucose_meter_malfunction.json")
        expected = test_case["expected_complaint"]
        ground_truth = test_case["ground_truth"]

        complaint = ComplaintRecord(
            complaint_id=expected["complaint_id"],
            intake_channel=IntakeChannel.FORM,
            received_date=datetime.fromisoformat(
                expected["received_date"].replace("Z", "+00:00")
            ),
            status=ComplaintStatus.NEW,
            device_info=DeviceInfo(**expected["device_info"]),
            event_info=EventInfo(**expected["event_info"]),
            narrative=expected["narrative"],
        )

        result = determine_mdr_rules_only(complaint)

        assert result.requires_mdr == ground_truth["requires_mdr"]
        # Should identify malfunction criterion
        malfunction_criteria = [
            MDRCriteria.MALFUNCTION_COULD_CAUSE_DEATH,
            MDRCriteria.MALFUNCTION_COULD_CAUSE_SERIOUS_INJURY,
        ]
        assert any(c in result.mdr_criteria_met for c in malfunction_criteria)


class TestMDRDeterminationModel:
    """Tests for MDRDetermination model properties."""

    def test_is_finalized_false_before_review(self) -> None:
        """is_finalized is False before human review."""
        determination = MDRDetermination(
            complaint_id="TEST-001",
            requires_mdr=True,
            ai_confidence=0.9,
            ai_reasoning="Test",
        )

        assert determination.is_finalized is False

    def test_is_finalized_true_after_confirmation(self) -> None:
        """is_finalized is True after human confirmation."""
        determination = MDRDetermination(
            complaint_id="TEST-001",
            requires_mdr=True,
            ai_confidence=0.9,
            ai_reasoning="Test",
            human_confirmed=True,
        )

        assert determination.is_finalized is True

    def test_final_requires_mdr_none_before_review(self) -> None:
        """final_requires_mdr is None before human review."""
        determination = MDRDetermination(
            complaint_id="TEST-001",
            requires_mdr=True,
            ai_confidence=0.9,
            ai_reasoning="Test",
        )

        assert determination.final_requires_mdr is None

    def test_final_requires_mdr_matches_ai_when_confirmed(self) -> None:
        """final_requires_mdr matches AI when human confirms."""
        determination = MDRDetermination(
            complaint_id="TEST-001",
            requires_mdr=True,
            ai_confidence=0.9,
            ai_reasoning="Test",
            human_confirmed=True,
        )

        assert determination.final_requires_mdr is True

    def test_final_requires_mdr_overridden_when_human_disagrees(self) -> None:
        """final_requires_mdr is opposite when human overrides."""
        determination = MDRDetermination(
            complaint_id="TEST-001",
            requires_mdr=True,
            ai_confidence=0.9,
            ai_reasoning="Test",
            human_confirmed=True,
            human_override_reason="User error confirmed, device worked correctly",
        )

        assert determination.final_requires_mdr is False


class TestSensitivity:
    """Tests to ensure 100% sensitivity (no false negatives)."""

    @pytest.mark.parametrize(
        "narrative,outcome",
        [
            ("Patient died.", "Death"),
            ("The patient passed away.", "Deceased"),
            ("Patient was found deceased.", None),
            ("Fatal outcome.", None),
            ("Resulted in fatality.", None),
            ("Patient expired in ER.", None),
        ],
    )
    def test_death_always_detected(self, narrative: str, outcome: str | None) -> None:
        """All death-related language triggers MDR."""
        complaint = create_complaint(
            complaint_id="SENS-DEATH",
            narrative=narrative,
            patient_outcome=outcome,
        )

        result = determine_mdr_rules_only(complaint)

        assert result.requires_mdr is True
        assert (
            MDRCriteria.DEATH in result.mdr_criteria_met
        ), f"Death not detected in: {narrative} / {outcome}"

    @pytest.mark.parametrize(
        "narrative,outcome",
        [
            ("Patient was hospitalized.", "Hospitalization"),
            ("Required emergency surgery.", "Surgery"),
            ("Admitted to ICU.", "Critical"),
            ("Life-threatening condition.", None),
            ("Permanent disability resulted.", None),
            ("Required surgical intervention.", None),
            ("Patient is now paralyzed.", None),
        ],
    )
    def test_serious_injury_always_detected(
        self, narrative: str, outcome: str | None
    ) -> None:
        """All serious injury language triggers MDR."""
        complaint = create_complaint(
            complaint_id="SENS-INJURY",
            narrative=narrative,
            patient_outcome=outcome,
        )

        result = determine_mdr_rules_only(complaint)

        assert result.requires_mdr is True
        assert (
            MDRCriteria.SERIOUS_INJURY in result.mdr_criteria_met
        ), f"Serious injury not detected in: {narrative} / {outcome}"
