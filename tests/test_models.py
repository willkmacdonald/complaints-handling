"""Tests for data models."""

import json
from datetime import date, datetime

import pytest
from pydantic import ValidationError

from src.models import (
    CodingDecision,
    CodingSuggestion,
    ComplaintRecord,
    ComplaintStatus,
    DeviceInfo,
    DeviceType,
    EventInfo,
    IMDRFCodeType,
    IntakeChannel,
    MDRCriteria,
    MDRDetermination,
    PatientInfo,
    ReporterInfo,
    ReporterType,
)


class TestDeviceInfo:
    """Tests for DeviceInfo model."""

    def test_minimal_device_info(self) -> None:
        """Test creating DeviceInfo with only required fields."""
        device = DeviceInfo(
            device_name="Pacemaker Model X",
            manufacturer="CardiacCorp",
        )
        assert device.device_name == "Pacemaker Model X"
        assert device.manufacturer == "CardiacCorp"
        assert device.model_number is None
        assert device.device_type == DeviceType.OTHER

    def test_full_device_info(self) -> None:
        """Test creating DeviceInfo with all fields."""
        device = DeviceInfo(
            device_name="Pacemaker Model X",
            manufacturer="CardiacCorp",
            model_number="PM-X100",
            serial_number="SN123456",
            lot_number="LOT2024A",
            device_type=DeviceType.IMPLANTABLE,
            udi="(01)00844588003288",
        )
        assert device.serial_number == "SN123456"
        assert device.device_type == DeviceType.IMPLANTABLE

    def test_device_info_json_roundtrip(self) -> None:
        """Test JSON serialization and deserialization."""
        device = DeviceInfo(
            device_name="Test Device",
            manufacturer="TestCorp",
            device_type=DeviceType.DIAGNOSTIC,
        )
        json_str = device.model_dump_json()
        loaded = DeviceInfo.model_validate_json(json_str)
        assert loaded == device


class TestPatientInfo:
    """Tests for PatientInfo model."""

    def test_empty_patient_info(self) -> None:
        """Test creating PatientInfo with no fields."""
        patient = PatientInfo()
        assert patient.age is None
        assert patient.sex is None
        assert patient.relevant_conditions == []

    def test_patient_info_with_conditions(self) -> None:
        """Test patient with relevant conditions."""
        patient = PatientInfo(
            age=65,
            sex="M",
            relevant_conditions=["diabetes", "hypertension"],
        )
        assert patient.age == 65
        assert len(patient.relevant_conditions) == 2

    def test_invalid_age(self) -> None:
        """Test that invalid age is rejected."""
        with pytest.raises(ValidationError):
            PatientInfo(age=-5)

        with pytest.raises(ValidationError):
            PatientInfo(age=200)


class TestEventInfo:
    """Tests for EventInfo model."""

    def test_minimal_event_info(self) -> None:
        """Test creating EventInfo with only required fields."""
        event = EventInfo(event_description="Device stopped working")
        assert event.event_description == "Device stopped working"
        assert event.event_date is None

    def test_full_event_info(self) -> None:
        """Test creating EventInfo with all fields."""
        event = EventInfo(
            event_date=date(2024, 1, 15),
            event_description="Device alarmed and shut down",
            patient_outcome="Patient experienced discomfort",
            device_outcome="Device returned to manufacturer",
            location="Hospital ICU",
            was_device_available_for_evaluation=True,
        )
        assert event.event_date == date(2024, 1, 15)
        assert event.was_device_available_for_evaluation is True


class TestComplaintRecord:
    """Tests for ComplaintRecord model."""

    @pytest.fixture
    def sample_complaint(self) -> ComplaintRecord:
        """Create a sample complaint for testing."""
        return ComplaintRecord(
            complaint_id="COMP-2024-001",
            intake_channel=IntakeChannel.FORM,
            received_date=datetime(2024, 1, 15, 10, 30),
            device_info=DeviceInfo(
                device_name="Blood Glucose Meter",
                manufacturer="DiabetesCare Inc",
                device_type=DeviceType.DIAGNOSTIC,
            ),
            event_info=EventInfo(
                event_description="Meter displayed incorrect reading",
                patient_outcome="Patient took incorrect insulin dose",
            ),
            narrative="The blood glucose meter showed a reading of 50 mg/dL when actual was 150 mg/dL...",
        )

    def test_create_complaint(self, sample_complaint: ComplaintRecord) -> None:
        """Test creating a complaint record."""
        assert sample_complaint.complaint_id == "COMP-2024-001"
        assert sample_complaint.status == ComplaintStatus.NEW
        assert sample_complaint.intake_channel == IntakeChannel.FORM

    def test_complaint_json_roundtrip(self, sample_complaint: ComplaintRecord) -> None:
        """Test JSON serialization preserves all data."""
        json_str = sample_complaint.model_dump_json()
        loaded = ComplaintRecord.model_validate_json(json_str)
        assert loaded.complaint_id == sample_complaint.complaint_id
        assert (
            loaded.device_info.device_name == sample_complaint.device_info.device_name
        )
        assert (
            loaded.event_info.event_description
            == sample_complaint.event_info.event_description
        )

    def test_complaint_with_all_fields(self) -> None:
        """Test complaint with all optional fields populated."""
        complaint = ComplaintRecord(
            complaint_id="COMP-2024-002",
            intake_channel=IntakeChannel.EMAIL,
            received_date=datetime(2024, 2, 1, 14, 0),
            status=ComplaintStatus.EXTRACTED,
            device_info=DeviceInfo(
                device_name="Hip Implant",
                manufacturer="OrthoMed",
                device_type=DeviceType.IMPLANTABLE,
            ),
            event_info=EventInfo(
                event_date=date(2024, 1, 28),
                event_description="Implant dislocation",
            ),
            patient_info=PatientInfo(age=72, sex="F"),
            reporter_info=ReporterInfo(
                reporter_type=ReporterType.PHYSICIAN,
                organization="City Hospital",
            ),
            narrative="Patient presented with hip dislocation...",
            extraction_confidence=0.85,
        )
        assert complaint.patient_info is not None
        assert complaint.patient_info.age == 72
        assert complaint.extraction_confidence == 0.85


class TestCodingSuggestion:
    """Tests for CodingSuggestion model."""

    def test_create_suggestion(self) -> None:
        """Test creating a coding suggestion."""
        suggestion = CodingSuggestion(
            code_id="A0601",
            code_name="Material Problem",
            code_type=IMDRFCodeType.DEVICE_PROBLEM,
            confidence=0.87,
            source_text="the device cracked during use",
            reasoning="Text mentions device cracking which indicates material integrity issue",
        )
        assert suggestion.code_id == "A0601"
        assert suggestion.confidence == 0.87

    def test_invalid_confidence(self) -> None:
        """Test that confidence must be 0-1."""
        with pytest.raises(ValidationError):
            CodingSuggestion(
                code_id="A0601",
                code_name="Test",
                code_type=IMDRFCodeType.DEVICE_PROBLEM,
                confidence=1.5,  # Invalid
                source_text="test",
                reasoning="test",
            )


class TestCodingDecision:
    """Tests for CodingDecision model."""

    @pytest.fixture
    def sample_decision(self) -> CodingDecision:
        """Create a sample coding decision."""
        return CodingDecision(
            complaint_id="COMP-2024-001",
            suggested_codes=[
                CodingSuggestion(
                    code_id="A0601",
                    code_name="Material Problem",
                    code_type=IMDRFCodeType.DEVICE_PROBLEM,
                    confidence=0.9,
                    source_text="device cracked",
                    reasoning="Material integrity issue",
                ),
                CodingSuggestion(
                    code_id="C0102",
                    code_name="Injury",
                    code_type=IMDRFCodeType.PATIENT_PROBLEM,
                    confidence=0.75,
                    source_text="patient was cut",
                    reasoning="Patient experienced injury",
                ),
            ],
            suggestion_timestamp=datetime(2024, 1, 15, 11, 0),
        )

    def test_unreviewd_decision(self, sample_decision: CodingDecision) -> None:
        """Test decision before human review."""
        assert not sample_decision.is_reviewed
        assert sample_decision.final_codes == []

    def test_reviewed_decision(self, sample_decision: CodingDecision) -> None:
        """Test decision after human review."""
        sample_decision.approved_codes = ["A0601"]
        sample_decision.rejected_codes = ["C0102"]
        sample_decision.added_codes = ["A0701"]
        sample_decision.reviewer_id = "reviewer-001"
        sample_decision.review_timestamp = datetime(2024, 1, 15, 12, 0)

        assert sample_decision.is_reviewed
        assert set(sample_decision.final_codes) == {"A0601", "A0701"}

    def test_suggestion_accuracy(self, sample_decision: CodingDecision) -> None:
        """Test accuracy calculation."""
        # Approve 1 of 2 suggestions
        sample_decision.approved_codes = ["A0601"]
        sample_decision.rejected_codes = ["C0102"]
        sample_decision.reviewer_id = "reviewer-001"
        sample_decision.review_timestamp = datetime.now()

        assert sample_decision.suggestion_accuracy == 0.5


class TestMDRDetermination:
    """Tests for MDRDetermination model."""

    def test_mdr_required(self) -> None:
        """Test MDR determination when filing is required."""
        mdr = MDRDetermination(
            complaint_id="COMP-2024-001",
            requires_mdr=True,
            mdr_criteria_met=[MDRCriteria.SERIOUS_INJURY],
            ai_confidence=0.95,
            ai_reasoning="Patient required hospitalization due to device malfunction",
            key_factors=["hospitalization", "device malfunction"],
            review_priority="urgent",
        )
        assert mdr.requires_mdr is True
        assert MDRCriteria.SERIOUS_INJURY in mdr.mdr_criteria_met
        assert mdr.review_priority == "urgent"

    def test_mdr_not_required(self) -> None:
        """Test MDR determination when filing is not required."""
        mdr = MDRDetermination(
            complaint_id="COMP-2024-002",
            requires_mdr=False,
            mdr_criteria_met=[],
            ai_confidence=0.88,
            ai_reasoning="Cosmetic issue only, no patient harm or risk",
            key_factors=["cosmetic defect", "no patient contact"],
        )
        assert mdr.requires_mdr is False
        assert mdr.mdr_criteria_met == []

    def test_human_confirmation(self) -> None:
        """Test human confirmation of AI determination."""
        mdr = MDRDetermination(
            complaint_id="COMP-2024-001",
            requires_mdr=True,
            mdr_criteria_met=[MDRCriteria.MALFUNCTION_COULD_CAUSE_DEATH],
            ai_confidence=0.85,
            ai_reasoning="Device could cause harm if malfunction recurs",
        )

        assert not mdr.is_finalized
        assert mdr.final_requires_mdr is None

        # Human confirms
        mdr.human_confirmed = True
        mdr.reviewer_id = "reviewer-001"
        mdr.review_timestamp = datetime.now()

        assert mdr.is_finalized
        assert mdr.final_requires_mdr is True

    def test_human_override(self) -> None:
        """Test human override of AI determination."""
        mdr = MDRDetermination(
            complaint_id="COMP-2024-003",
            requires_mdr=False,
            mdr_criteria_met=[],
            ai_confidence=0.7,
            ai_reasoning="No harm indicated in narrative",
        )

        # Human overrides to require MDR
        mdr.human_confirmed = False  # Did not confirm AI decision
        mdr.human_override_reason = (
            "Upon further review, device could cause serious injury"
        )
        mdr.reviewer_id = "reviewer-001"
        mdr.review_timestamp = datetime.now()

        assert mdr.is_finalized
        assert mdr.final_requires_mdr is True  # Opposite of AI determination


class TestJsonSerialization:
    """Test JSON serialization across all models."""

    def test_full_complaint_json_roundtrip(self) -> None:
        """Test complete complaint with all nested models."""
        complaint = ComplaintRecord(
            complaint_id="COMP-2024-FULL",
            intake_channel=IntakeChannel.CALL,
            received_date=datetime(2024, 3, 1, 9, 0),
            status=ComplaintStatus.CODED,
            device_info=DeviceInfo(
                device_name="Insulin Pump",
                manufacturer="DiabetesTech",
                model_number="IP-500",
                device_type=DeviceType.CONSUMABLE,
            ),
            event_info=EventInfo(
                event_date=date(2024, 2, 28),
                event_description="Pump delivered incorrect dose",
                patient_outcome="Hypoglycemic episode",
            ),
            patient_info=PatientInfo(
                age=45,
                sex="M",
                relevant_conditions=["Type 1 Diabetes"],
            ),
            reporter_info=ReporterInfo(
                reporter_type=ReporterType.PATIENT,
            ),
            narrative="I am calling about my insulin pump...",
            extracted_fields={"dose_error": "overdose", "amount": "5 units"},
            extraction_confidence=0.92,
        )

        # Serialize to JSON
        json_str = complaint.model_dump_json(indent=2)

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["complaint_id"] == "COMP-2024-FULL"

        # Deserialize back
        loaded = ComplaintRecord.model_validate_json(json_str)
        assert loaded == complaint

    def test_coding_decision_json_roundtrip(self) -> None:
        """Test coding decision serialization."""
        decision = CodingDecision(
            complaint_id="COMP-2024-001",
            suggested_codes=[
                CodingSuggestion(
                    code_id="A1234",
                    code_name="Test Code",
                    code_type=IMDRFCodeType.DEVICE_PROBLEM,
                    confidence=0.8,
                    source_text="test text",
                    reasoning="test reasoning",
                ),
            ],
            approved_codes=["A1234"],
            reviewer_id="test-reviewer",
            review_timestamp=datetime(2024, 1, 1, 12, 0),
        )

        json_str = decision.model_dump_json()
        loaded = CodingDecision.model_validate_json(json_str)

        assert loaded.complaint_id == decision.complaint_id
        assert len(loaded.suggested_codes) == 1
        assert loaded.suggested_codes[0].code_id == "A1234"
