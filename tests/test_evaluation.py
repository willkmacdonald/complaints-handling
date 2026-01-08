"""Tests for the evaluation framework."""

from datetime import UTC, date, datetime

from src.evaluation import (
    CodingMetrics,
    EvaluationReport,
    ExtractionMetrics,
    MDRMetrics,
    TestCaseResult,
    evaluate_coding,
    evaluate_extraction,
    evaluate_mdr,
    format_report_summary,
    generate_report,
)
from src.models import ComplaintRecord, DeviceInfo, EventInfo
from src.models.enums import (
    ComplaintStatus,
    DeviceType,
    IntakeChannel,
)
from src.models.mdr import MDRCriteria, MDRDetermination


class TestExtractionMetrics:
    """Tests for ExtractionMetrics model."""

    def test_perfect_extraction(self) -> None:
        """Test metrics for perfect extraction."""
        metrics = ExtractionMetrics(
            total_fields=10,
            correct_fields=10,
            missing_fields=0,
            extra_fields=0,
            partial_matches=0,
        )
        assert metrics.accuracy == 1.0
        assert metrics.completeness == 1.0
        assert metrics.precision == 1.0

    def test_partial_extraction(self) -> None:
        """Test metrics for partial extraction."""
        metrics = ExtractionMetrics(
            total_fields=10,
            correct_fields=7,
            missing_fields=3,
            extra_fields=0,
            partial_matches=0,
        )
        assert metrics.accuracy == 0.7
        assert metrics.completeness == 0.7
        assert metrics.precision == 1.0

    def test_extraction_with_extra_fields(self) -> None:
        """Test metrics when extra fields are extracted."""
        metrics = ExtractionMetrics(
            total_fields=8,  # 10 expected - 2 extra
            correct_fields=6,
            missing_fields=2,
            extra_fields=2,
            partial_matches=0,
        )
        assert metrics.accuracy == 0.75
        assert metrics.precision == 0.75  # 6 correct / (6 + 0 + 2)

    def test_extraction_with_partial_matches(self) -> None:
        """Test metrics with partial matches."""
        metrics = ExtractionMetrics(
            total_fields=10,
            correct_fields=5,
            missing_fields=2,
            extra_fields=0,
            partial_matches=3,
        )
        assert metrics.accuracy == 0.5
        assert metrics.precision == 5 / 8  # 5 / (5 + 3 + 0)

    def test_empty_extraction(self) -> None:
        """Test metrics for empty extraction."""
        metrics = ExtractionMetrics(
            total_fields=0,
            correct_fields=0,
            missing_fields=0,
            extra_fields=0,
        )
        assert metrics.accuracy == 1.0
        assert metrics.completeness == 1.0


class TestCodingMetrics:
    """Tests for CodingMetrics model."""

    def test_perfect_coding(self) -> None:
        """Test metrics for perfect code prediction."""
        metrics = CodingMetrics(
            predicted_codes=["A0601", "C01"],
            expected_codes=["A0601", "C01"],
            true_positives=2,
            false_positives=0,
            false_negatives=0,
        )
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.exact_match is True

    def test_partial_coding(self) -> None:
        """Test metrics for partial code prediction."""
        metrics = CodingMetrics(
            predicted_codes=["A0601"],
            expected_codes=["A0601", "C01"],
            true_positives=1,
            false_positives=0,
            false_negatives=1,
        )
        assert metrics.precision == 1.0
        assert metrics.recall == 0.5
        assert 0.66 < metrics.f1_score < 0.67
        assert metrics.exact_match is False

    def test_overprediction(self) -> None:
        """Test metrics when too many codes are predicted."""
        metrics = CodingMetrics(
            predicted_codes=["A0601", "C01", "A0701"],
            expected_codes=["A0601", "C01"],
            true_positives=2,
            false_positives=1,
            false_negatives=0,
        )
        assert metrics.precision == 2 / 3
        assert metrics.recall == 1.0
        assert metrics.exact_match is False

    def test_no_predictions(self) -> None:
        """Test metrics when no codes are predicted."""
        metrics = CodingMetrics(
            predicted_codes=[],
            expected_codes=["A0601"],
            true_positives=0,
            false_positives=0,
            false_negatives=1,
        )
        assert metrics.precision == 1.0  # No FPs
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0

    def test_no_expected(self) -> None:
        """Test metrics when no codes are expected."""
        metrics = CodingMetrics(
            predicted_codes=[],
            expected_codes=[],
            true_positives=0,
            false_positives=0,
            false_negatives=0,
        )
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.exact_match is True


class TestMDRMetrics:
    """Tests for MDRMetrics model."""

    def test_correct_positive(self) -> None:
        """Test metrics for correct positive MDR determination."""
        metrics = MDRMetrics(
            predicted_requires_mdr=True,
            expected_requires_mdr=True,
            predicted_criteria=["death"],
            expected_criteria=["death"],
        )
        assert metrics.is_correct is True
        assert metrics.is_false_negative is False
        assert metrics.is_false_positive is False

    def test_correct_negative(self) -> None:
        """Test metrics for correct negative MDR determination."""
        metrics = MDRMetrics(
            predicted_requires_mdr=False,
            expected_requires_mdr=False,
        )
        assert metrics.is_correct is True
        assert metrics.is_false_negative is False
        assert metrics.is_false_positive is False

    def test_false_negative(self) -> None:
        """Test metrics for false negative (missed MDR case)."""
        metrics = MDRMetrics(
            predicted_requires_mdr=False,
            expected_requires_mdr=True,
            expected_criteria=["death"],
        )
        assert metrics.is_correct is False
        assert metrics.is_false_negative is True
        assert metrics.is_false_positive is False

    def test_false_positive(self) -> None:
        """Test metrics for false positive (unnecessary MDR flag)."""
        metrics = MDRMetrics(
            predicted_requires_mdr=True,
            expected_requires_mdr=False,
        )
        assert metrics.is_correct is False
        assert metrics.is_false_negative is False
        assert metrics.is_false_positive is True

    def test_criteria_metrics(self) -> None:
        """Test criteria precision and recall."""
        metrics = MDRMetrics(
            predicted_requires_mdr=True,
            expected_requires_mdr=True,
            predicted_criteria=["death", "serious_injury"],
            expected_criteria=["death"],
        )
        assert metrics.criteria_precision == 0.5  # 1/2 predicted are correct
        assert metrics.criteria_recall == 1.0  # 1/1 expected are found


class TestEvaluateExtraction:
    """Tests for evaluate_extraction function."""

    def test_identical_records(self) -> None:
        """Test evaluation of identical records."""
        record = ComplaintRecord(
            complaint_id="TEST-001",
            intake_channel=IntakeChannel.FORM,
            received_date=datetime.now(UTC),
            status=ComplaintStatus.NEW,
            device_info=DeviceInfo(
                device_name="Test Device",
                manufacturer="Test Manufacturer",
                model_number="MODEL-123",
                device_type=DeviceType.DIAGNOSTIC,
            ),
            event_info=EventInfo(
                event_description="Test event",
                event_date=date(2024, 1, 15),
                patient_outcome="No harm",
            ),
            narrative="Test narrative",
        )

        metrics = evaluate_extraction(record, record)
        assert metrics.accuracy == 1.0
        assert metrics.missing_fields == 0

    def test_different_records(self) -> None:
        """Test evaluation of different records."""
        predicted = ComplaintRecord(
            complaint_id="TEST-001",
            intake_channel=IntakeChannel.FORM,
            received_date=datetime.now(UTC),
            device_info=DeviceInfo(
                device_name="Test Device",
                manufacturer="Wrong Manufacturer",
                device_type=DeviceType.DIAGNOSTIC,
            ),
            event_info=EventInfo(
                event_description="Test event",
            ),
            narrative="Test narrative",
        )

        expected = ComplaintRecord(
            complaint_id="TEST-001",
            intake_channel=IntakeChannel.FORM,
            received_date=datetime.now(UTC),
            device_info=DeviceInfo(
                device_name="Test Device",
                manufacturer="Correct Manufacturer",
                device_type=DeviceType.DIAGNOSTIC,
            ),
            event_info=EventInfo(
                event_description="Test event",
                patient_outcome="No harm",
            ),
            narrative="Test narrative",
        )

        metrics = evaluate_extraction(predicted, expected)
        assert metrics.accuracy < 1.0
        assert "manufacturer" in metrics.field_scores

    def test_missing_optional_fields(self) -> None:
        """Test evaluation when optional fields are missing."""
        predicted = ComplaintRecord(
            complaint_id="TEST-001",
            intake_channel=IntakeChannel.FORM,
            received_date=datetime.now(UTC),
            device_info=DeviceInfo(
                device_name="Test Device",
                manufacturer="Test Manufacturer",
            ),
            event_info=EventInfo(event_description="Test"),
            narrative="Test",
        )

        expected = ComplaintRecord(
            complaint_id="TEST-001",
            intake_channel=IntakeChannel.FORM,
            received_date=datetime.now(UTC),
            device_info=DeviceInfo(
                device_name="Test Device",
                manufacturer="Test Manufacturer",
                serial_number="SN123",
            ),
            event_info=EventInfo(event_description="Test"),
            narrative="Test",
        )

        metrics = evaluate_extraction(predicted, expected)
        assert metrics.missing_fields > 0


class TestEvaluateCoding:
    """Tests for evaluate_coding function."""

    def test_exact_match(self) -> None:
        """Test evaluation of exact code match."""
        metrics = evaluate_coding(
            predicted_codes=["A0601", "C01"],
            expected_codes=["A0601", "C01"],
        )
        assert metrics.exact_match is True
        assert metrics.f1_score == 1.0

    def test_with_alternatives(self) -> None:
        """Test that alternative codes count as correct."""
        metrics = evaluate_coding(
            predicted_codes=["A06"],  # Alternative to A0601
            expected_codes=["A0601"],
            alternative_codes=["A06"],
        )
        # A06 matches alternative, so it's not a false positive
        assert metrics.false_positives == 0
        # But it's counted as a true positive because it matches alternatives
        assert metrics.true_positives == 1

    def test_missed_codes(self) -> None:
        """Test evaluation when codes are missed."""
        metrics = evaluate_coding(
            predicted_codes=["A0601"],
            expected_codes=["A0601", "C01"],
        )
        assert metrics.recall == 0.5
        assert metrics.false_negatives == 1

    def test_extra_codes(self) -> None:
        """Test evaluation when extra codes are predicted."""
        metrics = evaluate_coding(
            predicted_codes=["A0601", "C01", "A0701"],
            expected_codes=["A0601", "C01"],
        )
        assert metrics.precision == 2 / 3
        assert metrics.false_positives == 1


class TestEvaluateMDR:
    """Tests for evaluate_mdr function."""

    def test_correct_determination(self) -> None:
        """Test evaluation of correct MDR determination."""
        determination = MDRDetermination(
            complaint_id="TEST-001",
            requires_mdr=True,
            mdr_criteria_met=[MDRCriteria.DEATH],
            ai_confidence=0.95,
            ai_reasoning="Patient died",
        )

        metrics = evaluate_mdr(
            predicted=determination,
            expected_requires_mdr=True,
            expected_criteria=["death"],
        )

        assert metrics.is_correct is True
        assert metrics.criteria_recall == 1.0

    def test_false_negative_detection(self) -> None:
        """Test that false negatives are correctly identified."""
        determination = MDRDetermination(
            complaint_id="TEST-001",
            requires_mdr=False,
            ai_confidence=0.8,
            ai_reasoning="No serious event",
        )

        metrics = evaluate_mdr(
            predicted=determination,
            expected_requires_mdr=True,
            expected_criteria=["death"],
        )

        assert metrics.is_false_negative is True


class TestGenerateReport:
    """Tests for report generation."""

    def test_empty_report(self) -> None:
        """Test generating report with no results."""
        report = generate_report([])
        assert report.overall.count == 0
        assert report.results == []

    def test_single_result(self) -> None:
        """Test generating report with single result."""
        result = TestCaseResult(
            test_case_id="TEST-001",
            name="Test Case",
            channel=IntakeChannel.FORM,
            device_type=DeviceType.DIAGNOSTIC,
            severity="malfunction",
            coding_metrics=CodingMetrics(
                predicted_codes=["A0601"],
                expected_codes=["A0601"],
                true_positives=1,
                false_positives=0,
                false_negatives=0,
            ),
        )

        report = generate_report([result])
        assert report.overall.count == 1
        assert report.overall.avg_coding_f1 == 1.0
        assert "form" in report.by_channel

    def test_report_breakdowns(self) -> None:
        """Test that report correctly groups by channel, device, severity."""
        results = [
            TestCaseResult(
                test_case_id="TEST-001",
                name="Form Test",
                channel=IntakeChannel.FORM,
                device_type=DeviceType.DIAGNOSTIC,
                severity="malfunction",
            ),
            TestCaseResult(
                test_case_id="TEST-002",
                name="Email Test",
                channel=IntakeChannel.EMAIL,
                device_type=DeviceType.IMPLANTABLE,
                severity="death",
            ),
        ]

        report = generate_report(results)
        assert len(report.by_channel) == 2
        assert "form" in report.by_channel
        assert "email" in report.by_channel
        assert len(report.by_severity) == 2

    def test_mdr_metrics_aggregation(self) -> None:
        """Test that MDR metrics are correctly aggregated."""
        results = [
            TestCaseResult(
                test_case_id="TEST-001",
                name="Death Case",
                channel=IntakeChannel.FORM,
                device_type=DeviceType.IMPLANTABLE,
                severity="death",
                mdr_metrics=MDRMetrics(
                    predicted_requires_mdr=True,
                    expected_requires_mdr=True,
                ),
            ),
            TestCaseResult(
                test_case_id="TEST-002",
                name="No MDR Case",
                channel=IntakeChannel.FORM,
                device_type=DeviceType.DIAGNOSTIC,
                severity="malfunction",
                mdr_metrics=MDRMetrics(
                    predicted_requires_mdr=False,
                    expected_requires_mdr=False,
                ),
            ),
        ]

        report = generate_report(results)
        assert report.overall.mdr_accuracy == 1.0
        assert report.overall.mdr_false_negative_count == 0

    def test_false_negative_counting(self) -> None:
        """Test that false negatives are counted correctly."""
        results = [
            TestCaseResult(
                test_case_id="TEST-001",
                name="Missed MDR",
                channel=IntakeChannel.FORM,
                device_type=DeviceType.IMPLANTABLE,
                severity="death",
                mdr_metrics=MDRMetrics(
                    predicted_requires_mdr=False,
                    expected_requires_mdr=True,
                ),
            ),
        ]

        report = generate_report(results)
        assert report.overall.mdr_false_negative_count == 1
        assert report.overall.mdr_accuracy == 0.0


class TestFormatReportSummary:
    """Tests for format_report_summary function."""

    def test_summary_formatting(self) -> None:
        """Test that summary is correctly formatted."""
        result = TestCaseResult(
            test_case_id="TEST-001",
            name="Test Case",
            channel=IntakeChannel.FORM,
            device_type=DeviceType.DIAGNOSTIC,
            severity="malfunction",
            coding_metrics=CodingMetrics(
                predicted_codes=["A0601"],
                expected_codes=["A0601"],
                true_positives=1,
                false_positives=0,
                false_negatives=0,
            ),
            mdr_metrics=MDRMetrics(
                predicted_requires_mdr=False,
                expected_requires_mdr=False,
            ),
        )

        report = generate_report([result])
        summary = format_report_summary(report)

        assert "1 test cases" in summary
        assert "Coding F1: 100.0%" in summary
        assert "MDR Accuracy: 100.0%" in summary

    def test_summary_with_false_negatives(self) -> None:
        """Test that summary warns about false negatives."""
        result = TestCaseResult(
            test_case_id="TEST-001",
            name="Test Case",
            channel=IntakeChannel.FORM,
            device_type=DeviceType.IMPLANTABLE,
            severity="death",
            mdr_metrics=MDRMetrics(
                predicted_requires_mdr=False,
                expected_requires_mdr=True,
            ),
        )

        report = generate_report([result])
        summary = format_report_summary(report)

        assert "WARNING" in summary
        assert "1 MDR false negative" in summary


class TestJsonRoundtrip:
    """Tests for JSON serialization/deserialization."""

    def test_extraction_metrics_roundtrip(self) -> None:
        """Test ExtractionMetrics JSON roundtrip."""
        metrics = ExtractionMetrics(
            total_fields=10,
            correct_fields=8,
            missing_fields=2,
            extra_fields=0,
            field_scores={"device_name": 1.0, "manufacturer": 0.5},
        )
        json_data = metrics.model_dump_json()
        restored = ExtractionMetrics.model_validate_json(json_data)
        assert restored.total_fields == metrics.total_fields
        assert restored.field_scores == metrics.field_scores

    def test_report_roundtrip(self) -> None:
        """Test EvaluationReport JSON roundtrip."""
        result = TestCaseResult(
            test_case_id="TEST-001",
            name="Test Case",
            channel=IntakeChannel.FORM,
            device_type=DeviceType.DIAGNOSTIC,
            severity="malfunction",
        )
        report = generate_report([result])

        json_data = report.model_dump_json()
        restored = EvaluationReport.model_validate_json(json_data)

        assert restored.overall.count == report.overall.count
        assert len(restored.results) == len(report.results)
