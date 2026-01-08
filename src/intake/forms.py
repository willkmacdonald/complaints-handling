"""Form submission intake processing."""

import logging
import uuid
from datetime import UTC, date, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.models.complaint import (
    ComplaintRecord,
    DeviceInfo,
    EventInfo,
    PatientInfo,
    ReporterInfo,
)
from src.models.enums import ComplaintStatus, DeviceType, IntakeChannel, ReporterType

logger = logging.getLogger(__name__)


class SubmitterInfo(BaseModel):
    """Information about the person submitting the form."""

    name: str | None = Field(default=None, description="Submitter's name")
    email: str | None = Field(default=None, description="Submitter's email")
    phone: str | None = Field(default=None, description="Submitter's phone")
    relationship: str | None = Field(
        default=None, description="Relationship to patient (self, family, HCP, etc.)"
    )
    reporter_type: str | None = Field(
        default=None, description="Type of reporter (patient, physician, etc.)"
    )
    organization: str | None = Field(
        default=None, description="Reporter's organization"
    )


class FormDeviceInfo(BaseModel):
    """Device information from form submission."""

    device_name: str | None = Field(default=None, description="Device name")
    manufacturer: str | None = Field(default=None, description="Manufacturer name")
    model_number: str | None = Field(default=None, description="Model number")
    serial_number: str | None = Field(default=None, description="Serial number")
    lot_number: str | None = Field(default=None, description="Lot/batch number")
    udi: str | None = Field(default=None, description="Unique Device Identifier")
    device_type: str | None = Field(
        default=None, description="Type of device (implantable, diagnostic, etc.)"
    )


class FormEventInfo(BaseModel):
    """Event information from form submission."""

    event_date: date | None = Field(default=None, description="Date of event")
    event_description: str | None = Field(
        default=None, description="Description of what happened"
    )
    patient_outcome: str | None = Field(
        default=None, description="What happened to the patient"
    )
    device_outcome: str | None = Field(
        default=None, description="What happened to the device"
    )
    location: str | None = Field(default=None, description="Where the event occurred")
    device_returned: bool | None = Field(
        default=None, description="Whether device was returned for evaluation"
    )


class FormPatientInfo(BaseModel):
    """Patient information from form submission."""

    age: int | None = Field(default=None, description="Patient age")
    sex: str | None = Field(default=None, description="Patient sex")
    weight_kg: float | None = Field(default=None, description="Patient weight in kg")
    relevant_conditions: list[str] = Field(
        default_factory=list, description="Relevant medical conditions"
    )

    @field_validator("age", mode="before")
    @classmethod
    def validate_age(cls, v: Any) -> int | None:
        """Convert age to int, handling string inputs."""
        if v is None:
            return None
        if isinstance(v, int):
            return v if 0 <= v <= 150 else None
        if isinstance(v, str):
            try:
                age = int(v)
                return age if 0 <= age <= 150 else None
            except ValueError:
                return None
        return None


class AttachmentInfo(BaseModel):
    """Metadata about an attachment."""

    filename: str = Field(..., description="Name of the file")
    content_type: str | None = Field(default=None, description="MIME type")
    size_bytes: int | None = Field(default=None, description="File size in bytes")
    description: str | None = Field(default=None, description="Description of contents")


class FormSubmission(BaseModel):
    """Represents a raw online form submission before conversion to ComplaintRecord."""

    form_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this submission",
    )
    form_type: str = Field(
        default="online_complaint", description="Type of form submitted"
    )
    submission_date: datetime = Field(..., description="When the form was submitted")

    # Form field groups
    submitter: SubmitterInfo = Field(
        default_factory=SubmitterInfo, description="Submitter information"
    )
    device: FormDeviceInfo = Field(
        default_factory=FormDeviceInfo, description="Device information"
    )
    event: FormEventInfo = Field(
        default_factory=FormEventInfo, description="Event information"
    )
    patient: FormPatientInfo = Field(
        default_factory=FormPatientInfo, description="Patient information"
    )

    # Attachments
    attachments: list[AttachmentInfo] = Field(
        default_factory=list, description="File attachments"
    )

    # Raw data preservation
    raw_fields: dict[str, Any] = Field(
        default_factory=dict, description="Original form fields as submitted"
    )

    @field_validator("submission_date", mode="before")
    @classmethod
    def parse_submission_date(cls, v: Any) -> datetime:
        """Parse submission date from various formats."""
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            # Try ISO format first
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                pass
            # Try common formats
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                try:
                    return datetime.strptime(v, fmt).replace(tzinfo=UTC)
                except ValueError:
                    continue
        raise ValueError(f"Cannot parse submission date: {v}")


class FormValidationResult(BaseModel):
    """Result of form validation with any issues found."""

    is_complete: bool = Field(
        ..., description="Whether all required fields are present"
    )
    missing_required: list[str] = Field(
        default_factory=list, description="List of missing required fields"
    )
    missing_recommended: list[str] = Field(
        default_factory=list, description="List of missing recommended fields"
    )
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")
    needs_followup: bool = Field(
        default=False, description="Whether follow-up is needed"
    )


# Required fields for a valid complaint
REQUIRED_FIELDS = [
    "device_name",
    "manufacturer",
    "event_description",
]

# Recommended fields that should trigger warnings if missing
RECOMMENDED_FIELDS = [
    "event_date",
    "model_number",
    "serial_number",
    "patient_outcome",
    "reporter_type",
]


def parse_form_submission(raw_data: dict[str, Any]) -> FormSubmission:
    """Parse raw form data into a structured FormSubmission.

    Args:
        raw_data: Dictionary containing form submission data.
            Expected structure:
            {
                "form_type": "online_complaint",
                "submission_date": "2024-01-15T14:30:00Z",
                "fields": {
                    "device_name": "...",
                    "manufacturer": "...",
                    ...
                }
            }

    Returns:
        Parsed FormSubmission object.

    Raises:
        ValueError: If required data is missing or invalid.
    """
    # Extract submission metadata
    form_type = raw_data.get("form_type", "online_complaint")
    submission_date = raw_data.get("submission_date")

    if not submission_date:
        raise ValueError("submission_date is required")

    # Get the fields dict (may be nested or flat)
    fields = raw_data.get("fields", raw_data)

    # Build submitter info
    submitter = SubmitterInfo(
        name=fields.get("submitter_name"),
        email=fields.get("submitter_email"),
        phone=fields.get("submitter_phone"),
        relationship=fields.get("relationship"),
        reporter_type=fields.get("reporter_type"),
        organization=fields.get("reporter_organization"),
    )

    # Build device info
    device = FormDeviceInfo(
        device_name=fields.get("device_name"),
        manufacturer=fields.get("manufacturer"),
        model_number=fields.get("model_number"),
        serial_number=fields.get("serial_number"),
        lot_number=fields.get("lot_number"),
        udi=fields.get("udi"),
        device_type=fields.get("device_type"),
    )

    # Build event info
    event_date = fields.get("event_date")
    if isinstance(event_date, str):
        try:
            event_date = date.fromisoformat(event_date)
        except ValueError:
            logger.warning("Could not parse event_date: %s", event_date)
            event_date = None

    event = FormEventInfo(
        event_date=event_date,
        event_description=fields.get("event_description"),
        patient_outcome=fields.get("patient_outcome"),
        device_outcome=fields.get("device_outcome"),
        location=fields.get("location"),
        device_returned=fields.get("device_returned"),
    )

    # Build patient info
    patient = FormPatientInfo(
        age=fields.get("patient_age"),
        sex=fields.get("patient_sex"),
        weight_kg=fields.get("patient_weight_kg"),
        relevant_conditions=fields.get("relevant_conditions", []),
    )

    # Parse attachments if present
    attachments = []
    raw_attachments = fields.get("attachments", [])
    for att in raw_attachments:
        if isinstance(att, dict):
            attachments.append(
                AttachmentInfo(
                    filename=att.get("filename", "unknown"),
                    content_type=att.get("content_type"),
                    size_bytes=att.get("size_bytes"),
                    description=att.get("description"),
                )
            )

    return FormSubmission(
        form_type=form_type,
        submission_date=submission_date,
        submitter=submitter,
        device=device,
        event=event,
        patient=patient,
        attachments=attachments,
        raw_fields=fields,
    )


def validate_form_completeness(form: FormSubmission) -> FormValidationResult:
    """Validate form submission completeness.

    Args:
        form: The parsed form submission to validate.

    Returns:
        FormValidationResult with validation status and any issues.
    """
    missing_required: list[str] = []
    missing_recommended: list[str] = []
    warnings: list[str] = []

    # Check required fields
    if not form.device.device_name:
        missing_required.append("device_name")
    if not form.device.manufacturer:
        missing_required.append("manufacturer")
    if not form.event.event_description:
        missing_required.append("event_description")

    # Check recommended fields
    if not form.event.event_date:
        missing_recommended.append("event_date")
    if not form.device.model_number:
        missing_recommended.append("model_number")
    if not form.device.serial_number:
        missing_recommended.append("serial_number")
    if not form.event.patient_outcome:
        missing_recommended.append("patient_outcome")
    if not form.submitter.reporter_type:
        missing_recommended.append("reporter_type")

    # Generate warnings
    if not form.device.serial_number and not form.device.lot_number:
        warnings.append(
            "Neither serial number nor lot number provided - "
            "device traceability may be limited"
        )

    if (
        form.event.patient_outcome
        and "death" in form.event.patient_outcome.lower()
        and not form.event.event_date
    ):
        warnings.append(
            "Death reported but event date not provided - "
            "this is critical for MDR timeline"
        )

    is_complete = len(missing_required) == 0
    needs_followup = not is_complete or len(warnings) > 0

    return FormValidationResult(
        is_complete=is_complete,
        missing_required=missing_required,
        missing_recommended=missing_recommended,
        warnings=warnings,
        needs_followup=needs_followup,
    )


def _map_reporter_type(reporter_type_str: str | None) -> ReporterType:
    """Map string reporter type to ReporterType enum."""
    if not reporter_type_str:
        return ReporterType.OTHER

    type_lower = reporter_type_str.lower().strip()
    mapping = {
        "patient": ReporterType.PATIENT,
        "family_member": ReporterType.FAMILY_MEMBER,
        "family": ReporterType.FAMILY_MEMBER,
        "physician": ReporterType.PHYSICIAN,
        "doctor": ReporterType.PHYSICIAN,
        "nurse": ReporterType.NURSE,
        "clinical_staff": ReporterType.CLINICAL_STAFF,
        "clinical": ReporterType.CLINICAL_STAFF,
        "sales_rep": ReporterType.SALES_REP,
        "sales": ReporterType.SALES_REP,
        "distributor": ReporterType.DISTRIBUTOR,
    }
    return mapping.get(type_lower, ReporterType.OTHER)


def _map_device_type(device_type_str: str | None) -> DeviceType:
    """Map string device type to DeviceType enum."""
    if not device_type_str:
        return DeviceType.OTHER

    type_lower = device_type_str.lower().strip()
    mapping = {
        "implantable": DeviceType.IMPLANTABLE,
        "implant": DeviceType.IMPLANTABLE,
        "diagnostic": DeviceType.DIAGNOSTIC,
        "consumable": DeviceType.CONSUMABLE,
        "samd": DeviceType.SAMD,
        "software": DeviceType.SAMD,
    }
    return mapping.get(type_lower, DeviceType.OTHER)


def form_to_complaint(
    form: FormSubmission,
    complaint_id: str | None = None,
    device_type: DeviceType | None = None,
) -> ComplaintRecord:
    """Convert a FormSubmission to a ComplaintRecord.

    Args:
        form: The parsed form submission.
        complaint_id: Optional complaint ID (generated if not provided).
        device_type: Optional device type override.

    Returns:
        ComplaintRecord ready for processing.

    Raises:
        ValueError: If required fields are missing.
    """
    # Validate required fields
    if not form.device.device_name:
        raise ValueError("device_name is required")
    if not form.device.manufacturer:
        raise ValueError("manufacturer is required")
    if not form.event.event_description:
        raise ValueError("event_description is required")

    # Generate complaint ID if not provided
    if not complaint_id:
        complaint_id = f"FORM-{form.form_id[:8].upper()}"

    # Build DeviceInfo
    resolved_device_type = device_type or _map_device_type(form.device.device_type)
    device_info = DeviceInfo(
        device_name=form.device.device_name,
        manufacturer=form.device.manufacturer,
        model_number=form.device.model_number,
        serial_number=form.device.serial_number,
        lot_number=form.device.lot_number,
        device_type=resolved_device_type,
        udi=form.device.udi,
    )

    # Build EventInfo
    device_outcome = form.event.device_outcome
    if form.event.device_returned is True and not device_outcome:
        device_outcome = "Device returned for evaluation"
    elif form.event.device_returned is False and not device_outcome:
        device_outcome = "Device not returned"

    event_info = EventInfo(
        event_date=form.event.event_date,
        event_description=form.event.event_description,
        patient_outcome=form.event.patient_outcome,
        device_outcome=device_outcome,
        location=form.event.location,
        was_device_available_for_evaluation=form.event.device_returned,
    )

    # Build PatientInfo (optional)
    patient_info = None
    if (
        form.patient.age is not None
        or form.patient.sex
        or form.patient.relevant_conditions
    ):
        patient_info = PatientInfo(
            age=form.patient.age,
            sex=form.patient.sex,
            weight_kg=form.patient.weight_kg,
            relevant_conditions=form.patient.relevant_conditions,
        )

    # Build ReporterInfo (optional)
    reporter_info = None
    if form.submitter.reporter_type or form.submitter.organization:
        reporter_info = ReporterInfo(
            reporter_type=_map_reporter_type(form.submitter.reporter_type),
            organization=form.submitter.organization,
        )

    # Build narrative from event description
    narrative = form.event.event_description

    return ComplaintRecord(
        complaint_id=complaint_id,
        intake_channel=IntakeChannel.FORM,
        received_date=form.submission_date,
        status=ComplaintStatus.NEW,
        device_info=device_info,
        event_info=event_info,
        patient_info=patient_info,
        reporter_info=reporter_info,
        narrative=narrative,
        raw_content=form.raw_fields,
    )
