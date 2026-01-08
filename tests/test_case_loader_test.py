"""Tests for test case loader."""

from src.models import IntakeChannel
from src.models.enums import DeviceType
from src.testing import (
    ComplaintTestCase,
    load_all_test_cases,
    load_test_case,
    load_test_cases_by_channel,
)
from src.testing.test_case_loader import get_test_case_summary


class TestLoadTestCase:
    """Tests for loading individual test cases."""

    def test_load_existing_test_case(self) -> None:
        """Test loading a test case that exists."""
        test_case = load_test_case("FORM-001")
        assert test_case is not None
        assert test_case.test_case_id == "FORM-001"
        assert test_case.name == "Pacemaker Failure Leading to Death"
        assert test_case.channel == IntakeChannel.FORM

    def test_load_nonexistent_test_case(self) -> None:
        """Test loading a test case that doesn't exist."""
        test_case = load_test_case("NONEXISTENT-999")
        assert test_case is None

    def test_test_case_has_ground_truth(self) -> None:
        """Test that loaded test cases include ground truth."""
        test_case = load_test_case("FORM-001")
        assert test_case is not None
        assert test_case.ground_truth is not None
        assert len(test_case.ground_truth.expected_imdrf_codes) > 0
        assert test_case.ground_truth.requires_mdr is True

    def test_test_case_has_expected_complaint(self) -> None:
        """Test that loaded test cases include expected complaint."""
        test_case = load_test_case("FORM-002")
        assert test_case is not None
        assert test_case.expected_complaint is not None
        assert test_case.expected_complaint.device_info.device_name is not None


class TestLoadTestCasesByChannel:
    """Tests for loading test cases by channel."""

    def test_load_form_test_cases(self) -> None:
        """Test loading all form test cases."""
        test_cases = load_test_cases_by_channel(IntakeChannel.FORM)
        assert len(test_cases) == 7

        for case in test_cases:
            assert case.channel == IntakeChannel.FORM

    def test_load_nonexistent_channel(self) -> None:
        """Test loading test cases for channel with no data."""
        test_cases = load_test_cases_by_channel(IntakeChannel.EMAIL)
        # Will be empty until we create email test cases
        assert isinstance(test_cases, list)


class TestLoadAllTestCases:
    """Tests for loading all test cases with filters."""

    def test_load_all_without_filters(self) -> None:
        """Test loading all test cases."""
        test_cases = load_all_test_cases()
        assert len(test_cases) >= 7  # At least the 7 form cases

    def test_filter_by_device_type(self) -> None:
        """Test filtering by device type."""
        implantable_cases = load_all_test_cases(device_type=DeviceType.IMPLANTABLE)
        assert len(implantable_cases) == 2

        for case in implantable_cases:
            assert case.device_type == DeviceType.IMPLANTABLE

    def test_filter_by_severity(self) -> None:
        """Test filtering by severity."""
        death_cases = load_all_test_cases(severity="death")
        assert len(death_cases) == 1
        assert death_cases[0].test_case_id == "FORM-001"

    def test_filter_by_difficulty(self) -> None:
        """Test filtering by difficulty."""
        easy_cases = load_all_test_cases(difficulty="easy")
        assert len(easy_cases) >= 2

        for case in easy_cases:
            assert case.difficulty == "easy"

    def test_multiple_filters(self) -> None:
        """Test applying multiple filters."""
        cases = load_all_test_cases(
            device_type=DeviceType.DIAGNOSTIC, difficulty="medium"
        )

        for case in cases:
            assert case.device_type == DeviceType.DIAGNOSTIC
            assert case.difficulty == "medium"


class TestTestCaseSummary:
    """Tests for test case summary function."""

    def test_get_summary(self) -> None:
        """Test getting summary of test cases."""
        summary = get_test_case_summary()

        assert summary["total"] >= 7
        assert "by_channel" in summary
        assert "by_device_type" in summary
        assert "by_severity" in summary
        assert "by_difficulty" in summary

    def test_summary_channel_counts(self) -> None:
        """Test that channel counts are accurate."""
        summary = get_test_case_summary()
        assert summary["by_channel"].get("form", 0) == 7


class TestTestCaseContent:
    """Tests for test case content validation."""

    def test_all_form_cases_have_required_fields(self) -> None:
        """Test that all form test cases have required fields."""
        test_cases = load_test_cases_by_channel(IntakeChannel.FORM)

        for case in test_cases:
            # Basic fields
            assert case.test_case_id is not None
            assert case.name is not None
            assert case.raw_input is not None

            # Ground truth
            assert case.ground_truth.expected_imdrf_codes is not None
            assert isinstance(case.ground_truth.requires_mdr, bool)
            assert case.ground_truth.coding_rationale is not None

            # Expected complaint
            assert case.expected_complaint.device_info is not None
            assert case.expected_complaint.event_info is not None
            assert case.expected_complaint.narrative is not None

    def test_mdr_cases_have_criteria(self) -> None:
        """Test that MDR-required cases have criteria listed."""
        test_cases = load_all_test_cases()

        for case in test_cases:
            if case.ground_truth.requires_mdr:
                assert len(case.ground_truth.mdr_criteria) > 0, (
                    f"Test case {case.test_case_id} requires MDR but has no criteria"
                )

    def test_imdrf_codes_are_valid_format(self) -> None:
        """Test that IMDRF codes follow expected format."""
        test_cases = load_all_test_cases()

        for case in test_cases:
            for code in case.ground_truth.expected_imdrf_codes:
                # Codes should start with A, B, C, or D followed by digits
                assert code[0] in "ABCD", f"Invalid code prefix in {code}"
                assert code[1:].replace("0", "").isdigit() or code[1:].isdigit(), (
                    f"Invalid code format: {code}"
                )


class TestComplaintTestCaseModel:
    """Tests for ComplaintTestCase model."""

    def test_test_case_json_roundtrip(self) -> None:
        """Test that test cases can be serialized and deserialized."""
        test_case = load_test_case("FORM-001")
        assert test_case is not None

        json_str = test_case.model_dump_json()
        loaded = ComplaintTestCase.model_validate_json(json_str)

        assert loaded.test_case_id == test_case.test_case_id
        assert loaded.ground_truth.requires_mdr == test_case.ground_truth.requires_mdr
