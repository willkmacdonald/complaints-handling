"""Intake module for processing complaints from various channels."""

from src.intake.forms import (
    FormSubmission,
    FormValidationResult,
    form_to_complaint,
    parse_form_submission,
    validate_form_completeness,
)

__all__ = [
    "FormSubmission",
    "FormValidationResult",
    "form_to_complaint",
    "parse_form_submission",
    "validate_form_completeness",
]
