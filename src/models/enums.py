"""Enumerations for complaint handling system."""

from enum import Enum


class IntakeChannel(str, Enum):
    """Channel through which complaint was received."""

    FORM = "form"
    EMAIL = "email"
    CALL = "call"
    LETTER = "letter"
    SALES_REP = "sales_rep"


class DeviceType(str, Enum):
    """Category of medical device."""

    IMPLANTABLE = "implantable"
    DIAGNOSTIC = "diagnostic"
    CONSUMABLE = "consumable"
    SAMD = "samd"  # Software as Medical Device
    OTHER = "other"


class ComplaintStatus(str, Enum):
    """Processing status of a complaint."""

    NEW = "new"
    EXTRACTED = "extracted"
    CODED = "coded"
    REVIEWED = "reviewed"
    CLOSED = "closed"


class ReporterType(str, Enum):
    """Type of person reporting the complaint."""

    PATIENT = "patient"
    FAMILY_MEMBER = "family_member"
    PHYSICIAN = "physician"
    NURSE = "nurse"
    CLINICAL_STAFF = "clinical_staff"
    SALES_REP = "sales_rep"
    DISTRIBUTOR = "distributor"
    OTHER = "other"


class IMDRFCodeType(str, Enum):
    """Type of IMDRF code."""

    DEVICE_PROBLEM = "device_problem"  # Annex A
    COMPONENT = "component"  # Annex B
    PATIENT_PROBLEM = "patient_problem"  # Annex C
    EVALUATION = "evaluation"  # Annex D
