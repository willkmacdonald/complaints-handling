"""Evaluation metrics models."""

from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

from src.models.enums import DeviceType, IntakeChannel


class PromptStrategy(str, Enum):
    """Prompt strategy for IMDRF code suggestions."""

    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT_COT = "few_shot_cot"


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


class EvaluationFilters(BaseModel):
    """Filters applied to test cases during evaluation."""

    device_type: DeviceType | None = Field(
        default=None, description="Filter by device type"
    )
    channel: IntakeChannel | None = Field(
        default=None, description="Filter by intake channel"
    )
    severity: str | None = Field(default=None, description="Filter by severity level")
    difficulty: str | None = Field(
        default=None, description="Filter by difficulty level"
    )


class TokenStats(BaseModel):
    """Token usage statistics for an evaluation run."""

    total_prompt_tokens: int = Field(default=0, description="Total prompt tokens used")
    total_completion_tokens: int = Field(
        default=0, description="Total completion tokens used"
    )
    total_tokens: int = Field(default=0, description="Total tokens used")

    @property
    def avg_tokens_per_case(self) -> float:
        """Average tokens per test case (requires case count from parent)."""
        return 0.0  # Calculated at runner level


class EvaluationRunMetadata(BaseModel):
    """Metadata for an evaluation run."""

    run_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique identifier for this run",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the evaluation started",
    )
    strategy: PromptStrategy = Field(
        default=PromptStrategy.ZERO_SHOT,
        description="Prompt strategy used",
    )
    model: str = Field(default="", description="LLM model used for evaluation")
    filters: EvaluationFilters = Field(
        default_factory=EvaluationFilters,
        description="Filters applied to test cases",
    )
    token_stats: TokenStats = Field(
        default_factory=TokenStats,
        description="Token usage statistics",
    )
    total_duration_ms: float = Field(
        default=0.0, description="Total evaluation duration in milliseconds"
    )
    test_case_count: int = Field(
        default=0, description="Number of test cases evaluated"
    )


class TestCaseEvaluationResult(BaseModel):
    """Result of evaluating a single test case."""

    test_case_id: str = Field(..., description="Test case identifier")
    name: str = Field(..., description="Human-readable test case name")
    channel: IntakeChannel = Field(..., description="Intake channel")
    device_type: DeviceType = Field(..., description="Device type")
    severity: str = Field(..., description="Severity level")
    difficulty: str = Field(default="medium", description="Difficulty level")

    # Predicted outputs
    predicted_codes: list[str] = Field(
        default_factory=list, description="Codes predicted by the system"
    )
    predicted_confidences: dict[str, float] = Field(
        default_factory=dict, description="Confidence scores for each predicted code"
    )

    # Ground truth
    expected_codes: list[str] = Field(
        default_factory=list, description="Expected IMDRF codes"
    )
    alternative_codes: list[str] = Field(
        default_factory=list, description="Alternative acceptable codes"
    )

    # Metrics
    coding_metrics: CodingMetrics | None = Field(
        default=None, description="Coding accuracy metrics"
    )

    # Execution details
    tokens_used: int = Field(default=0, description="Tokens used for this case")
    latency_ms: float = Field(default=0.0, description="Request latency in ms")
    error: str | None = Field(default=None, description="Error message if failed")

    @property
    def is_success(self) -> bool:
        """Check if evaluation was successful."""
        return self.error is None


class EvaluationRunResult(BaseModel):
    """Complete result of an evaluation run."""

    metadata: EvaluationRunMetadata = Field(
        ..., description="Run metadata and configuration"
    )
    results: list[TestCaseEvaluationResult] = Field(
        default_factory=list, description="Per-test-case results"
    )

    # Aggregate metrics (calculated from results)
    overall_precision: float | None = Field(
        default=None, description="Overall precision across all cases"
    )
    overall_recall: float | None = Field(
        default=None, description="Overall recall across all cases"
    )
    overall_f1: float | None = Field(
        default=None, description="Overall F1 score across all cases"
    )
    exact_match_rate: float | None = Field(
        default=None, description="Rate of exact code matches"
    )

    # Breakdowns
    by_difficulty: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Metrics by difficulty level"
    )
    by_device_type: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Metrics by device type"
    )
    by_channel: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Metrics by intake channel"
    )

    def calculate_aggregates(self) -> None:
        """Calculate aggregate metrics from individual results."""
        successful_results = [
            r for r in self.results if r.is_success and r.coding_metrics
        ]

        if not successful_results:
            return

        # Calculate overall metrics
        precisions = [
            r.coding_metrics.precision for r in successful_results if r.coding_metrics
        ]
        recalls = [
            r.coding_metrics.recall for r in successful_results if r.coding_metrics
        ]
        f1_scores = [
            r.coding_metrics.f1_score for r in successful_results if r.coding_metrics
        ]
        exact_matches = sum(
            1
            for r in successful_results
            if r.coding_metrics and r.coding_metrics.exact_match
        )

        self.overall_precision = (
            sum(precisions) / len(precisions) if precisions else None
        )
        self.overall_recall = sum(recalls) / len(recalls) if recalls else None
        self.overall_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else None
        self.exact_match_rate = (
            exact_matches / len(successful_results) if successful_results else None
        )

        # Calculate breakdowns
        self.by_difficulty = self._calculate_breakdown(successful_results, "difficulty")
        self.by_device_type = self._calculate_breakdown(
            successful_results, "device_type"
        )
        self.by_channel = self._calculate_breakdown(successful_results, "channel")

    def _calculate_breakdown(
        self, results: list[TestCaseEvaluationResult], group_by: str
    ) -> dict[str, dict[str, float]]:
        """Calculate metrics breakdown by a specific attribute."""
        groups: dict[str, list[TestCaseEvaluationResult]] = {}

        for r in results:
            key = getattr(r, group_by)
            if hasattr(key, "value"):
                key = key.value
            if key not in groups:
                groups[key] = []
            groups[key].append(r)

        breakdown: dict[str, dict[str, float]] = {}
        for key, group_results in groups.items():
            metrics_list = [r.coding_metrics for r in group_results if r.coding_metrics]
            if metrics_list:
                breakdown[key] = {
                    "count": len(group_results),
                    "precision": sum(m.precision for m in metrics_list)
                    / len(metrics_list),
                    "recall": sum(m.recall for m in metrics_list) / len(metrics_list),
                    "f1": sum(m.f1_score for m in metrics_list) / len(metrics_list),
                }

        return breakdown
