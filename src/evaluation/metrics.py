"""Evaluation metrics calculation functions."""

from typing import Any

from src.evaluation.models import CodingMetrics, ExtractionMetrics, MDRMetrics
from src.models import ComplaintRecord
from src.models.mdr import MDRDetermination


def _normalize_value(value: Any) -> Any:
    """Normalize a value for comparison."""
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, list):
        return sorted([_normalize_value(v) for v in value])
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in value.items()}
    return value


def _compare_values(predicted: Any, expected: Any) -> tuple[bool, float]:
    """Compare two values and return (exact_match, similarity_score).

    Returns:
        Tuple of (is_exact_match, similarity_score from 0.0 to 1.0).
    """
    if predicted is None and expected is None:
        return True, 1.0
    if predicted is None or expected is None:
        return False, 0.0

    pred_norm = _normalize_value(predicted)
    exp_norm = _normalize_value(expected)

    if pred_norm == exp_norm:
        return True, 1.0

    # String similarity for partial matches
    if isinstance(predicted, str) and isinstance(expected, str):
        pred_lower = predicted.lower()
        exp_lower = expected.lower()
        if pred_lower in exp_lower or exp_lower in pred_lower:
            return False, 0.5
        # Check for significant overlap
        pred_words = set(pred_lower.split())
        exp_words = set(exp_lower.split())
        if pred_words and exp_words:
            overlap = len(pred_words & exp_words) / len(pred_words | exp_words)
            if overlap > 0.5:
                return False, overlap

    return False, 0.0


def _extract_comparable_fields(record: ComplaintRecord) -> dict[str, Any]:
    """Extract fields from ComplaintRecord for comparison.

    Returns a flat dictionary of field values.
    """
    fields: dict[str, Any] = {}

    # Device info fields
    if record.device_info:
        fields["device_name"] = record.device_info.device_name
        fields["manufacturer"] = record.device_info.manufacturer
        fields["model_number"] = record.device_info.model_number
        fields["serial_number"] = record.device_info.serial_number
        fields["lot_number"] = record.device_info.lot_number
        fields["device_type"] = (
            record.device_info.device_type.value
            if record.device_info.device_type
            else None
        )

    # Event info fields
    if record.event_info:
        fields["event_date"] = (
            str(record.event_info.event_date) if record.event_info.event_date else None
        )
        fields["event_description"] = record.event_info.event_description
        fields["patient_outcome"] = record.event_info.patient_outcome
        fields["device_outcome"] = record.event_info.device_outcome

    # Patient info fields
    if record.patient_info:
        fields["patient_age"] = record.patient_info.age
        fields["patient_sex"] = record.patient_info.sex

    # Reporter info fields
    if record.reporter_info:
        fields["reporter_type"] = (
            record.reporter_info.reporter_type.value
            if record.reporter_info.reporter_type
            else None
        )
        fields["reporter_organization"] = record.reporter_info.organization

    return fields


def evaluate_extraction(
    predicted: ComplaintRecord,
    expected: ComplaintRecord,
) -> ExtractionMetrics:
    """Evaluate extraction accuracy by comparing predicted and expected records.

    Args:
        predicted: The ComplaintRecord produced by the extraction system.
        expected: The ground truth ComplaintRecord.

    Returns:
        ExtractionMetrics with detailed field-level analysis.
    """
    pred_fields = _extract_comparable_fields(predicted)
    exp_fields = _extract_comparable_fields(expected)

    all_fields = set(pred_fields.keys()) | set(exp_fields.keys())
    field_scores: dict[str, float] = {}

    correct = 0
    missing = 0
    extra = 0
    partial = 0

    for field_name in all_fields:
        pred_val = pred_fields.get(field_name)
        exp_val = exp_fields.get(field_name)

        if field_name not in exp_fields or exp_val is None:
            # Field is extra (not expected)
            if pred_val is not None:
                extra += 1
                field_scores[field_name] = 0.0
            continue

        if field_name not in pred_fields or pred_val is None:
            # Field is missing
            missing += 1
            field_scores[field_name] = 0.0
            continue

        exact_match, similarity = _compare_values(pred_val, exp_val)
        field_scores[field_name] = similarity

        if exact_match:
            correct += 1
        elif similarity > 0:
            partial += 1
        else:
            missing += 1

    total_expected = len([f for f in exp_fields.values() if f is not None])

    return ExtractionMetrics(
        total_fields=total_expected,
        correct_fields=correct,
        missing_fields=missing,
        extra_fields=extra,
        partial_matches=partial,
        field_scores=field_scores,
    )


def evaluate_coding(
    predicted_codes: list[str],
    expected_codes: list[str],
    alternative_codes: list[str] | None = None,
) -> CodingMetrics:
    """Evaluate IMDRF coding accuracy.

    Args:
        predicted_codes: Codes predicted by the system.
        expected_codes: Ground truth codes.
        alternative_codes: Optional alternative acceptable codes.

    Returns:
        CodingMetrics with precision, recall, and F1 score.
    """
    alternative_codes = alternative_codes or []

    # Combine expected and alternative codes for matching
    acceptable_codes = set(expected_codes) | set(alternative_codes)
    predicted_set = set(predicted_codes)
    expected_set = set(expected_codes)

    # True positives: predicted codes that match expected or alternatives
    # For TP, we count matches against expected codes only (alternatives don't add to TP)
    # but alternatives DO prevent false positives
    true_positives = len(predicted_set & expected_set)

    # Also count alternative code matches as correct predictions
    alternative_matches = predicted_set & set(alternative_codes)
    true_positives += len(alternative_matches)

    # False positives: predicted codes that are neither expected nor alternatives
    false_positives = len(predicted_set - acceptable_codes)

    # False negatives: expected codes that weren't predicted
    # (alternatives don't count as false negatives if not predicted)
    false_negatives = len(expected_set - predicted_set)

    return CodingMetrics(
        predicted_codes=predicted_codes,
        expected_codes=expected_codes,
        alternative_codes=alternative_codes,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def evaluate_mdr(
    predicted: MDRDetermination,
    expected_requires_mdr: bool,
    expected_criteria: list[str] | None = None,
) -> MDRMetrics:
    """Evaluate MDR determination accuracy.

    Args:
        predicted: MDR determination from the system.
        expected_requires_mdr: Ground truth MDR requirement.
        expected_criteria: Ground truth criteria that apply.

    Returns:
        MDRMetrics with determination accuracy and criteria evaluation.
    """
    expected_criteria = expected_criteria or []

    # Extract criteria values from MDRCriteria enums
    predicted_criteria_values = [c.value for c in predicted.mdr_criteria_met]

    return MDRMetrics(
        predicted_requires_mdr=predicted.requires_mdr,
        expected_requires_mdr=expected_requires_mdr,
        predicted_criteria=predicted_criteria_values,
        expected_criteria=expected_criteria,
    )


# Re-export models for convenience
__all__ = [
    "CodingMetrics",
    "ExtractionMetrics",
    "MDRMetrics",
    "evaluate_coding",
    "evaluate_extraction",
    "evaluate_mdr",
]
