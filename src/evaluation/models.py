"""Evaluation metrics models."""

from pydantic import BaseModel, Field


class ExtractionMetrics(BaseModel):
    """Metrics for evaluating field extraction accuracy."""

    total_fields: int = Field(..., description="Total number of fields evaluated")
    correct_fields: int = Field(..., description="Number of correctly extracted fields")
    missing_fields: int = Field(
        ..., description="Number of fields that should exist but don't"
    )
    extra_fields: int = Field(
        ..., description="Number of fields extracted but not expected"
    )
    partial_matches: int = Field(
        default=0, description="Number of fields with partial matches"
    )

    field_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-field accuracy scores (0.0-1.0)",
    )

    @property
    def accuracy(self) -> float:
        """Overall extraction accuracy."""
        if self.total_fields == 0:
            return 1.0
        return self.correct_fields / self.total_fields

    @property
    def completeness(self) -> float:
        """Measure of how many expected fields were extracted."""
        expected = self.total_fields - self.extra_fields
        if expected == 0:
            return 1.0
        extracted = expected - self.missing_fields
        return extracted / expected

    @property
    def precision(self) -> float:
        """Precision of extracted fields (correct / total extracted)."""
        total_extracted = self.correct_fields + self.partial_matches + self.extra_fields
        if total_extracted == 0:
            return 1.0
        return self.correct_fields / total_extracted


class CodingMetrics(BaseModel):
    """Metrics for evaluating IMDRF code suggestions."""

    predicted_codes: list[str] = Field(..., description="Codes predicted by the system")
    expected_codes: list[str] = Field(..., description="Ground truth codes")
    alternative_codes: list[str] = Field(
        default_factory=list, description="Alternative acceptable codes"
    )

    true_positives: int = Field(..., description="Correctly predicted codes")
    false_positives: int = Field(..., description="Incorrectly predicted codes")
    false_negatives: int = Field(..., description="Missed codes")

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)."""
        total_predicted = self.true_positives + self.false_positives
        if total_predicted == 0:
            return 1.0
        return self.true_positives / total_predicted

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)."""
        total_expected = self.true_positives + self.false_negatives
        if total_expected == 0:
            return 1.0
        return self.true_positives / total_expected

    @property
    def f1_score(self) -> float:
        """F1 = 2 * (precision * recall) / (precision + recall)."""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    @property
    def exact_match(self) -> bool:
        """Whether predicted codes exactly match expected codes."""
        return set(self.predicted_codes) == set(self.expected_codes)


class MDRMetrics(BaseModel):
    """Metrics for evaluating MDR determination accuracy."""

    predicted_requires_mdr: bool = Field(..., description="System's MDR determination")
    expected_requires_mdr: bool = Field(..., description="Ground truth MDR requirement")
    predicted_criteria: list[str] = Field(
        default_factory=list, description="Criteria predicted by system"
    )
    expected_criteria: list[str] = Field(
        default_factory=list, description="Ground truth criteria"
    )

    @property
    def is_correct(self) -> bool:
        """Whether the MDR determination is correct."""
        return self.predicted_requires_mdr == self.expected_requires_mdr

    @property
    def is_false_negative(self) -> bool:
        """Whether this is a false negative (missed MDR requirement).

        This is critical - we must not miss cases that require MDR.
        """
        return self.expected_requires_mdr and not self.predicted_requires_mdr

    @property
    def is_false_positive(self) -> bool:
        """Whether this is a false positive (incorrectly flagged for MDR)."""
        return not self.expected_requires_mdr and self.predicted_requires_mdr

    @property
    def criteria_precision(self) -> float:
        """Precision on criteria identification."""
        if not self.predicted_criteria:
            return 1.0 if not self.expected_criteria else 0.0
        correct = len(set(self.predicted_criteria) & set(self.expected_criteria))
        return correct / len(self.predicted_criteria)

    @property
    def criteria_recall(self) -> float:
        """Recall on criteria identification."""
        if not self.expected_criteria:
            return 1.0
        correct = len(set(self.predicted_criteria) & set(self.expected_criteria))
        return correct / len(self.expected_criteria)
