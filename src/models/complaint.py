"""Core complaint data models."""

from datetime import UTC, date, datetime
from typing import Any

from pydantic import BaseModel, Field

from src.models.enums import ComplaintStatus, DeviceType, IntakeChannel, ReporterType


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(UTC)


class DeviceInfo(BaseModel):
    """Information about the medical device involved in the complaint."""

    device_name: str = Field(..., description="Name or description of the device")
    manufacturer: str = Field(..., description="Device manufacturer name")
    model_number: str | None = Field(default=None, description="Device model number")
    serial_number: str | None = Field(default=None, description="Device serial number")
    lot_number: str | None = Field(default=None, description="Manufacturing lot number")
    device_type: DeviceType = Field(
        default=DeviceType.OTHER, description="Category of medical device"
    )
    udi: str | None = Field(default=None, description="Unique Device Identifier")


class PatientInfo(BaseModel):
    """Anonymized patient information (no PII stored)."""

    age: int | None = Field(default=None, ge=0, le=150, description="Patient age")
    sex: str | None = Field(default=None, description="Patient sex (M/F/Other/Unknown)")
    weight_kg: float | None = Field(
        default=None, ge=0, description="Patient weight in kg"
    )
    relevant_conditions: list[str] = Field(
        default_factory=list, description="Relevant medical conditions"
    )


class ReporterInfo(BaseModel):
    """Information about who reported the complaint."""

    reporter_type: ReporterType = Field(..., description="Type of reporter")
    organization: str | None = Field(
        default=None, description="Reporter's organization (hospital, clinic, etc.)"
    )
    contact_reference: str | None = Field(
        default=None,
        description="Reference ID for contacting reporter (not actual contact info)",
    )


class EventInfo(BaseModel):
    """Information about the complaint event."""

    event_date: date | None = Field(default=None, description="Date the event occurred")
    event_description: str = Field(..., description="Description of what happened")
    patient_outcome: str | None = Field(
        default=None, description="Outcome for the patient"
    )
    device_outcome: str | None = Field(
        default=None, description="What happened to the device"
    )
    location: str | None = Field(
        default=None, description="Where the event occurred (hospital, home, etc.)"
    )
    was_device_available_for_evaluation: bool | None = Field(
        default=None, description="Whether the device was returned for evaluation"
    )


class ComplaintRecord(BaseModel):
    """Complete complaint record combining all information."""

    complaint_id: str = Field(..., description="Unique complaint identifier")
    intake_channel: IntakeChannel = Field(
        ..., description="Channel through which complaint was received"
    )
    received_date: datetime = Field(
        ..., description="Date and time complaint was received"
    )
    status: ComplaintStatus = Field(
        default=ComplaintStatus.NEW, description="Current processing status"
    )

    # Core information
    device_info: DeviceInfo = Field(..., description="Device information")
    event_info: EventInfo = Field(..., description="Event information")
    patient_info: PatientInfo | None = Field(
        default=None, description="Patient information (if available)"
    )
    reporter_info: ReporterInfo | None = Field(
        default=None, description="Reporter information (if available)"
    )

    # Raw content
    narrative: str = Field(..., description="Raw complaint narrative text")
    raw_content: dict[str, Any] | None = Field(
        default=None, description="Original raw content (form fields, email, etc.)"
    )

    # Extraction metadata
    extracted_fields: dict[str, Any] = Field(
        default_factory=dict, description="Fields extracted by AI"
    )
    extraction_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Overall extraction confidence"
    )

    # Audit fields
    created_at: datetime = Field(
        default_factory=_utc_now, description="Record creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=_utc_now, description="Last update timestamp"
    )
