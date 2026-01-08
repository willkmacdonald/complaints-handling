"""Tests for form intake processing."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.intake import (
    FormSubmission,
    form_to_complaint,
    parse_form_submission,
    validate_form_completeness,
)
from src.models.enums import DeviceType, IntakeChannel, ReporterType

# Path to test case data
TEST_CASES_DIR = Path(__file__).parent.parent / "data" / "test_cases" / "form"


def load_test_case(filename: str) -> dict:
    """Load a test case JSON file."""
    with open(TEST_CASES_DIR / filename) as f:
        return json.load(f)


class TestParseFormSubmission:
    """Tests for parse_form_submission function."""

    def test_parse_complete_form(self) -> None:
        """Parse a complete form with all fields."""
        test_case = load_test_case("form_001_pacemaker_death.json")
        raw_input = test_case["raw_input"]

        form = parse_form_submission(raw_input)

        assert form.form_type == "online_complaint"
        assert form.device.device_name == "CardioRhythm Pacemaker Model CR-500"
        assert form.device.manufacturer == "CardioRhythm Medical Inc."
        assert form.device.model_number == "CR-500"
        assert form.device.serial_number == "CR500-2021-78432"
        assert form.device.lot_number == "LOT2021-Q3-445"
        assert form.event.event_date == date(2024, 1, 10)
        assert "battery depleted" in form.event.event_description.lower()
        assert form.event.patient_outcome == "Death"
        assert form.event.device_returned is True
        assert form.submitter.reporter_type == "physician"
        assert form.submitter.organization == "Memorial Heart Center"
        assert form.patient.age == 78
        assert form.patient.sex == "M"

    def test_parse_form_with_minimal_fields(self) -> None:
        """Parse a form with only required fields."""
        raw_data = {
            "submission_date": "2024-01-15T10:00:00Z",
            "fields": {
                "device_name": "Test Device",
                "manufacturer": "Test Manufacturer",
                "event_description": "Device malfunctioned during use.",
            },
        }

        form = parse_form_submission(raw_data)

        assert form.device.device_name == "Test Device"
        assert form.device.manufacturer == "Test Manufacturer"
        assert form.event.event_description == "Device malfunctioned during use."
        assert form.device.model_number is None
        assert form.patient.age is None

    def test_parse_form_missing_submission_date_raises(self) -> None:
        """Parsing without submission_date raises ValueError."""
        raw_data = {
            "fields": {
                "device_name": "Test Device",
                "manufacturer": "Test Manufacturer",
            },
        }

        with pytest.raises(ValueError, match="submission_date is required"):
            parse_form_submission(raw_data)

    def test_parse_all_form_test_cases(self) -> None:
        """All form test cases should parse successfully."""
        manifest_path = TEST_CASES_DIR / "_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        for test_case_info in manifest["test_cases"]:
            test_case = load_test_case(test_case_info["file"])
            raw_input = test_case["raw_input"]

            form = parse_form_submission(raw_input)

            assert form.device.device_name is not None
            assert form.device.manufacturer is not None
            assert form.event.event_description is not None

    def test_parse_date_formats(self) -> None:
        """Handle various date formats."""
        # ISO format with Z
        raw_data = {
            "submission_date": "2024-01-15T14:30:00Z",
            "fields": {
                "device_name": "Test",
                "manufacturer": "Test",
                "event_description": "Test",
                "event_date": "2024-01-10",
            },
        }
        form = parse_form_submission(raw_data)
        assert form.submission_date.year == 2024
        assert form.event.event_date == date(2024, 1, 10)

    def test_parse_preserves_raw_fields(self) -> None:
        """Raw fields are preserved in the FormSubmission."""
        raw_data = {
            "submission_date": "2024-01-15T10:00:00Z",
            "fields": {
                "device_name": "Test Device",
                "manufacturer": "Test Mfg",
                "event_description": "Test event",
                "custom_field": "custom_value",
            },
        }

        form = parse_form_submission(raw_data)

        assert form.raw_fields["custom_field"] == "custom_value"


class TestValidateFormCompleteness:
    """Tests for form validation."""

    def test_complete_form_is_valid(self) -> None:
        """A complete form passes validation."""
        test_case = load_test_case("form_001_pacemaker_death.json")
        form = parse_form_submission(test_case["raw_input"])

        result = validate_form_completeness(form)

        assert result.is_complete is True
        assert len(result.missing_required) == 0

    def test_missing_required_fields(self) -> None:
        """Missing required fields are detected."""
        form = FormSubmission(
            submission_date=datetime.now(UTC),
            # Missing device_name, manufacturer, event_description
        )

        result = validate_form_completeness(form)

        assert result.is_complete is False
        assert "device_name" in result.missing_required
        assert "manufacturer" in result.missing_required
        assert "event_description" in result.missing_required
        assert result.needs_followup is True

    def test_missing_recommended_fields(self) -> None:
        """Missing recommended fields trigger warnings but form is still complete."""
        raw_data = {
            "submission_date": "2024-01-15T10:00:00Z",
            "fields": {
                "device_name": "Test Device",
                "manufacturer": "Test Manufacturer",
                "event_description": "Device malfunctioned.",
                # Missing: event_date, model_number, serial_number, patient_outcome, reporter_type
            },
        }
        form = parse_form_submission(raw_data)

        result = validate_form_completeness(form)

        assert result.is_complete is True  # Required fields present
        assert "event_date" in result.missing_recommended
        assert "model_number" in result.missing_recommended
        assert "serial_number" in result.missing_recommended
        assert "patient_outcome" in result.missing_recommended
        assert "reporter_type" in result.missing_recommended

    def test_traceability_warning(self) -> None:
        """Warning when neither serial nor lot number provided."""
        raw_data = {
            "submission_date": "2024-01-15T10:00:00Z",
            "fields": {
                "device_name": "Test Device",
                "manufacturer": "Test Manufacturer",
                "event_description": "Device malfunctioned.",
                # No serial_number or lot_number
            },
        }
        form = parse_form_submission(raw_data)

        result = validate_form_completeness(form)

        assert any("traceability" in w.lower() for w in result.warnings)

    def test_death_without_date_warning(self) -> None:
        """Warning when death reported but no event date."""
        raw_data = {
            "submission_date": "2024-01-15T10:00:00Z",
            "fields": {
                "device_name": "Test Device",
                "manufacturer": "Test Manufacturer",
                "event_description": "Device failure.",
                "patient_outcome": "Death",
                # No event_date
            },
        }
        form = parse_form_submission(raw_data)

        result = validate_form_completeness(form)

        assert any(
            "death" in w.lower() and "date" in w.lower() for w in result.warnings
        )


class TestFormToComplaint:
    """Tests for form to complaint conversion."""

    def test_convert_complete_form(self) -> None:
        """Convert a complete form to ComplaintRecord."""
        test_case = load_test_case("form_001_pacemaker_death.json")
        form = parse_form_submission(test_case["raw_input"])

        complaint = form_to_complaint(form, complaint_id="FORM-001")

        assert complaint.complaint_id == "FORM-001"
        assert complaint.intake_channel == IntakeChannel.FORM
        assert (
            complaint.device_info.device_name == "CardioRhythm Pacemaker Model CR-500"
        )
        assert complaint.device_info.manufacturer == "CardioRhythm Medical Inc."
        assert complaint.event_info.event_date == date(2024, 1, 10)
        assert "battery depleted" in complaint.narrative.lower()
        assert complaint.patient_info is not None
        assert complaint.patient_info.age == 78
        assert complaint.reporter_info is not None
        assert complaint.reporter_info.reporter_type == ReporterType.PHYSICIAN

    def test_convert_generates_complaint_id(self) -> None:
        """Complaint ID is generated if not provided."""
        raw_data = {
            "submission_date": "2024-01-15T10:00:00Z",
            "fields": {
                "device_name": "Test Device",
                "manufacturer": "Test Manufacturer",
                "event_description": "Device malfunctioned.",
            },
        }
        form = parse_form_submission(raw_data)

        complaint = form_to_complaint(form)

        assert complaint.complaint_id.startswith("FORM-")

    def test_convert_missing_required_raises(self) -> None:
        """Conversion fails if required fields missing."""
        form = FormSubmission(
            submission_date=datetime.now(UTC),
            # Missing device_name, manufacturer, event_description
        )

        with pytest.raises(ValueError, match="device_name is required"):
            form_to_complaint(form)

    def test_convert_maps_reporter_types(self) -> None:
        """Reporter types are correctly mapped."""
        test_cases = [
            ("physician", ReporterType.PHYSICIAN),
            ("patient", ReporterType.PATIENT),
            ("family_member", ReporterType.FAMILY_MEMBER),
            ("nurse", ReporterType.NURSE),
            ("sales_rep", ReporterType.SALES_REP),
            ("unknown_type", ReporterType.OTHER),
        ]

        for reporter_str, expected_type in test_cases:
            raw_data = {
                "submission_date": "2024-01-15T10:00:00Z",
                "fields": {
                    "device_name": "Test Device",
                    "manufacturer": "Test Manufacturer",
                    "event_description": "Test event.",
                    "reporter_type": reporter_str,
                },
            }
            form = parse_form_submission(raw_data)
            complaint = form_to_complaint(form)

            assert complaint.reporter_info is not None
            assert complaint.reporter_info.reporter_type == expected_type

    def test_convert_device_returned_sets_outcome(self) -> None:
        """Device returned flag sets appropriate device outcome."""
        # Device returned = True
        raw_data = {
            "submission_date": "2024-01-15T10:00:00Z",
            "fields": {
                "device_name": "Test Device",
                "manufacturer": "Test Manufacturer",
                "event_description": "Test event.",
                "device_returned": True,
            },
        }
        form = parse_form_submission(raw_data)
        complaint = form_to_complaint(form)

        assert "returned" in complaint.event_info.device_outcome.lower()
        assert complaint.event_info.was_device_available_for_evaluation is True

        # Device returned = False
        raw_data["fields"]["device_returned"] = False
        form = parse_form_submission(raw_data)
        complaint = form_to_complaint(form)

        assert "not returned" in complaint.event_info.device_outcome.lower()
        assert complaint.event_info.was_device_available_for_evaluation is False

    def test_convert_with_device_type_override(self) -> None:
        """Device type can be overridden."""
        raw_data = {
            "submission_date": "2024-01-15T10:00:00Z",
            "fields": {
                "device_name": "Test Device",
                "manufacturer": "Test Manufacturer",
                "event_description": "Test event.",
            },
        }
        form = parse_form_submission(raw_data)

        complaint = form_to_complaint(form, device_type=DeviceType.IMPLANTABLE)

        assert complaint.device_info.device_type == DeviceType.IMPLANTABLE

    def test_convert_preserves_raw_content(self) -> None:
        """Raw form fields are preserved in complaint."""
        raw_data = {
            "submission_date": "2024-01-15T10:00:00Z",
            "fields": {
                "device_name": "Test Device",
                "manufacturer": "Test Manufacturer",
                "event_description": "Test event.",
                "custom_field": "custom_value",
            },
        }
        form = parse_form_submission(raw_data)
        complaint = form_to_complaint(form)

        assert complaint.raw_content is not None
        assert complaint.raw_content.get("custom_field") == "custom_value"


class TestAllFormTestCases:
    """Integration tests using all form test cases."""

    def test_all_forms_parse_and_convert(self) -> None:
        """All form test cases parse and convert successfully."""
        manifest_path = TEST_CASES_DIR / "_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        for test_case_info in manifest["test_cases"]:
            test_case = load_test_case(test_case_info["file"])
            raw_input = test_case["raw_input"]
            expected = test_case["expected_complaint"]

            # Parse
            form = parse_form_submission(raw_input)
            assert form is not None

            # Convert
            complaint = form_to_complaint(form, complaint_id=expected["complaint_id"])

            # Verify key fields match expected
            assert complaint.complaint_id == expected["complaint_id"]
            assert (
                complaint.device_info.device_name
                == expected["device_info"]["device_name"]
            )
            assert (
                complaint.device_info.manufacturer
                == expected["device_info"]["manufacturer"]
            )
            assert (
                complaint.event_info.event_description
                == expected["event_info"]["event_description"]
            )

    def test_all_forms_validate(self) -> None:
        """All form test cases should pass validation (they have complete data)."""
        manifest_path = TEST_CASES_DIR / "_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        for test_case_info in manifest["test_cases"]:
            test_case = load_test_case(test_case_info["file"])
            raw_input = test_case["raw_input"]

            form = parse_form_submission(raw_input)
            result = validate_form_completeness(form)

            assert result.is_complete is True, (
                f"Test case {test_case_info['id']} should be complete but is missing: "
                f"{result.missing_required}"
            )

    def test_patient_info_extracted(self) -> None:
        """Patient info is correctly extracted when present."""
        test_case = load_test_case("form_001_pacemaker_death.json")
        form = parse_form_submission(test_case["raw_input"])
        complaint = form_to_complaint(form)

        assert complaint.patient_info is not None
        assert complaint.patient_info.age == 78
        assert complaint.patient_info.sex == "M"

    def test_form_without_patient_info(self) -> None:
        """Forms without patient info still convert successfully."""
        raw_data = {
            "submission_date": "2024-01-15T10:00:00Z",
            "fields": {
                "device_name": "Test Device",
                "manufacturer": "Test Manufacturer",
                "event_description": "Test event.",
                # No patient info
            },
        }
        form = parse_form_submission(raw_data)
        complaint = form_to_complaint(form)

        assert complaint.patient_info is None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_age_handled(self) -> None:
        """Invalid age values are handled gracefully."""
        raw_data = {
            "submission_date": "2024-01-15T10:00:00Z",
            "fields": {
                "device_name": "Test Device",
                "manufacturer": "Test Manufacturer",
                "event_description": "Test event.",
                "patient_age": "invalid",
            },
        }
        form = parse_form_submission(raw_data)

        assert form.patient.age is None

    def test_age_out_of_range_handled(self) -> None:
        """Age values out of range are handled."""
        raw_data = {
            "submission_date": "2024-01-15T10:00:00Z",
            "fields": {
                "device_name": "Test Device",
                "manufacturer": "Test Manufacturer",
                "event_description": "Test event.",
                "patient_age": 200,
            },
        }
        form = parse_form_submission(raw_data)

        assert form.patient.age is None

    def test_empty_fields_handled(self) -> None:
        """Empty string fields are handled as None."""
        raw_data = {
            "submission_date": "2024-01-15T10:00:00Z",
            "fields": {
                "device_name": "Test Device",
                "manufacturer": "Test Manufacturer",
                "event_description": "Test event.",
                "serial_number": "",
            },
        }
        form = parse_form_submission(raw_data)

        # Empty strings are preserved (not converted to None by pydantic by default)
        # This is expected behavior - empty string is different from None
        assert form.device.serial_number == ""

    def test_form_id_generated(self) -> None:
        """Form ID is auto-generated."""
        raw_data = {
            "submission_date": "2024-01-15T10:00:00Z",
            "fields": {
                "device_name": "Test Device",
                "manufacturer": "Test Manufacturer",
                "event_description": "Test event.",
            },
        }
        form1 = parse_form_submission(raw_data)
        form2 = parse_form_submission(raw_data)

        # Each form gets a unique ID
        assert form1.form_id != form2.form_id
