"""Confidence calibration analysis for IMDRF code suggestions.

This module provides tools for analyzing confidence-accuracy correlation
and finding optimal confidence thresholds.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from src.evaluation.models import EvaluationRunResult

logger = logging.getLogger(__name__)

# Default calibration config path
DEFAULT_CALIBRATION_CONFIG_PATH = Path("data/calibration_config.json")


class SuggestionOutcome(BaseModel):
    """Outcome of a single code suggestion for calibration analysis."""

    code_id: str = Field(..., description="The suggested IMDRF code")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    is_correct: bool = Field(..., description="Whether the suggestion was correct")
    test_case_id: str = Field(..., description="Source test case")


class CalibrationBin(BaseModel):
    """A bin for calibration analysis."""

    bin_start: float = Field(..., ge=0.0, le=1.0, description="Lower bound (inclusive)")
    bin_end: float = Field(..., ge=0.0, le=1.0, description="Upper bound (exclusive)")
    count: int = Field(default=0, description="Number of suggestions in this bin")
    correct_count: int = Field(default=0, description="Number of correct suggestions")
    avg_confidence: float = Field(
        default=0.0, description="Average confidence in this bin"
    )
    accuracy: float = Field(default=0.0, description="Actual accuracy in this bin")
    calibration_error: float = Field(
        default=0.0, description="Absolute difference between confidence and accuracy"
    )


class CalibrationAnalysis(BaseModel):
    """Complete calibration analysis results."""

    run_id: str = Field(..., description="Evaluation run this analysis is based on")
    total_suggestions: int = Field(
        default=0, description="Total number of suggestions analyzed"
    )
    total_correct: int = Field(default=0, description="Total correct suggestions")
    overall_accuracy: float = Field(
        default=0.0, description="Overall accuracy across all suggestions"
    )
    avg_confidence: float = Field(
        default=0.0, description="Average confidence across all suggestions"
    )

    # Calibration metrics
    expected_calibration_error: float = Field(
        default=0.0,
        description="ECE: Weighted average of per-bin calibration errors",
    )
    maximum_calibration_error: float = Field(
        default=0.0,
        description="MCE: Maximum per-bin calibration error",
    )
    brier_score: float = Field(
        default=0.0,
        description="Mean squared error between confidence and correctness",
    )

    # Bin details
    bins: list[CalibrationBin] = Field(
        default_factory=list, description="Per-bin calibration details"
    )

    # Threshold analysis
    optimal_threshold: float = Field(
        default=0.5, description="Optimal confidence threshold for best F1"
    )
    threshold_analysis: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Metrics at different threshold values",
    )


class CalibrationConfig(BaseModel):
    """Configuration for confidence thresholds based on calibration analysis."""

    optimal_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Optimal confidence threshold for filtering suggestions",
    )
    high_confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Threshold for high-confidence suggestions",
    )
    low_confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Below this, suggestions need human review",
    )
    source_run_id: str = Field(
        default="",
        description="Evaluation run ID this config was derived from",
    )
    ece: float = Field(
        default=0.0,
        description="Expected Calibration Error at time of config creation",
    )


@dataclass
class ThresholdMetrics:
    """Metrics computed at a specific confidence threshold."""

    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def precision(self) -> float:
        """Precision at this threshold."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 1.0

    @property
    def recall(self) -> float:
        """Recall at this threshold."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 1.0

    @property
    def f1_score(self) -> float:
        """F1 score at this threshold."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def extract_suggestion_outcomes(
    result: EvaluationRunResult,
) -> list[SuggestionOutcome]:
    """Extract individual suggestion outcomes from an evaluation run.

    Args:
        result: Evaluation run result with per-case predictions.

    Returns:
        List of SuggestionOutcome for each predicted code.
    """
    outcomes: list[SuggestionOutcome] = []

    for case_result in result.results:
        if not case_result.is_success:
            continue

        # Build set of correct codes (expected + alternatives)
        correct_codes = set(case_result.expected_codes) | set(
            case_result.alternative_codes
        )

        # Create outcome for each prediction
        for code_id, confidence in case_result.predicted_confidences.items():
            outcomes.append(
                SuggestionOutcome(
                    code_id=code_id,
                    confidence=confidence,
                    is_correct=code_id in correct_codes,
                    test_case_id=case_result.test_case_id,
                )
            )

    return outcomes


def calculate_calibration_error(
    outcomes: list[SuggestionOutcome],
    num_bins: int = 10,
) -> CalibrationAnalysis:
    """Calculate calibration error metrics from suggestion outcomes.

    Expected Calibration Error (ECE) measures how well confidence scores
    predict actual accuracy. A perfectly calibrated model has ECE = 0.

    Maximum Calibration Error (MCE) is the worst-case per-bin error.

    Args:
        outcomes: List of suggestion outcomes with confidence and correctness.
        num_bins: Number of bins for calibration analysis.

    Returns:
        CalibrationAnalysis with ECE, MCE, and per-bin details.
    """
    if not outcomes:
        return CalibrationAnalysis(
            run_id="",
            total_suggestions=0,
            bins=[],
        )

    # Initialize bins
    bin_width = 1.0 / num_bins
    bins: list[CalibrationBin] = []
    bin_outcomes: list[list[SuggestionOutcome]] = [[] for _ in range(num_bins)]

    # Assign outcomes to bins
    for outcome in outcomes:
        bin_idx = min(int(outcome.confidence / bin_width), num_bins - 1)
        bin_outcomes[bin_idx].append(outcome)

    # Calculate per-bin metrics
    for i in range(num_bins):
        bin_start = i * bin_width
        bin_end = (i + 1) * bin_width
        bin_data = bin_outcomes[i]

        if bin_data:
            count = len(bin_data)
            correct_count = sum(1 for o in bin_data if o.is_correct)
            avg_conf = sum(o.confidence for o in bin_data) / count
            accuracy = correct_count / count
            cal_error = abs(avg_conf - accuracy)
        else:
            count = 0
            correct_count = 0
            avg_conf = 0.0
            accuracy = 0.0
            cal_error = 0.0

        bins.append(
            CalibrationBin(
                bin_start=bin_start,
                bin_end=bin_end,
                count=count,
                correct_count=correct_count,
                avg_confidence=avg_conf,
                accuracy=accuracy,
                calibration_error=cal_error,
            )
        )

    # Calculate overall metrics
    total = len(outcomes)
    total_correct = sum(1 for o in outcomes if o.is_correct)
    overall_accuracy = total_correct / total if total > 0 else 0.0
    avg_confidence = sum(o.confidence for o in outcomes) / total if total > 0 else 0.0

    # Calculate ECE (weighted average of per-bin calibration errors)
    ece = sum(b.count * b.calibration_error for b in bins) / total if total > 0 else 0.0

    # Calculate MCE (maximum per-bin calibration error)
    mce = max((b.calibration_error for b in bins if b.count > 0), default=0.0)

    # Calculate Brier score (mean squared error)
    brier = (
        sum((o.confidence - (1.0 if o.is_correct else 0.0)) ** 2 for o in outcomes)
        / total
        if total > 0
        else 0.0
    )

    return CalibrationAnalysis(
        run_id="",  # Will be set by caller
        total_suggestions=total,
        total_correct=total_correct,
        overall_accuracy=overall_accuracy,
        avg_confidence=avg_confidence,
        expected_calibration_error=ece,
        maximum_calibration_error=mce,
        brier_score=brier,
        bins=bins,
    )


def _compute_threshold_metrics(
    outcomes: list[SuggestionOutcome],
    threshold: float,
    expected_codes_by_case: dict[str, set[str]],
) -> ThresholdMetrics:
    """Compute precision/recall metrics at a given threshold.

    At threshold T:
    - TP: correct predictions with confidence >= T
    - FP: incorrect predictions with confidence >= T
    - FN: expected codes not predicted with confidence >= T
    - TN: incorrect predictions filtered out (confidence < T)

    Args:
        outcomes: All suggestion outcomes.
        threshold: Confidence threshold.
        expected_codes_by_case: Expected codes for each test case.

    Returns:
        ThresholdMetrics at the given threshold.
    """
    tp = fp = fn = tn = 0

    # Group outcomes by test case
    outcomes_by_case: dict[str, list[SuggestionOutcome]] = {}
    for o in outcomes:
        if o.test_case_id not in outcomes_by_case:
            outcomes_by_case[o.test_case_id] = []
        outcomes_by_case[o.test_case_id].append(o)

    for case_id, case_outcomes in outcomes_by_case.items():
        expected = expected_codes_by_case.get(case_id, set())

        # Predictions above threshold
        above_threshold = [o for o in case_outcomes if o.confidence >= threshold]
        predicted_codes = {o.code_id for o in above_threshold}

        # Calculate TP, FP, FN
        for o in above_threshold:
            if o.is_correct:
                tp += 1
            else:
                fp += 1

        # FN: expected codes not predicted above threshold
        fn += len(expected - predicted_codes)

        # TN: incorrect predictions filtered out
        below_threshold = [o for o in case_outcomes if o.confidence < threshold]
        tn += sum(1 for o in below_threshold if not o.is_correct)

    return ThresholdMetrics(
        threshold=threshold,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
    )


def find_optimal_threshold(
    outcomes: list[SuggestionOutcome],
    result: EvaluationRunResult,
    threshold_range: tuple[float, float] = (0.1, 0.9),
    step: float = 0.05,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Find the optimal confidence threshold that maximizes F1 score.

    Args:
        outcomes: Suggestion outcomes to analyze.
        result: Original evaluation result (for expected codes).
        threshold_range: Range of thresholds to test (min, max).
        step: Step size for threshold search.

    Returns:
        Tuple of (optimal_threshold, threshold_analysis dict).
    """
    # Build expected codes lookup
    expected_codes_by_case: dict[str, set[str]] = {}
    for case_result in result.results:
        if case_result.is_success:
            expected_codes_by_case[case_result.test_case_id] = set(
                case_result.expected_codes
            ) | set(case_result.alternative_codes)

    # Test thresholds
    threshold_analysis: dict[str, dict[str, float]] = {}
    best_threshold = 0.5
    best_f1 = 0.0

    threshold = threshold_range[0]
    while threshold <= threshold_range[1]:
        metrics = _compute_threshold_metrics(
            outcomes, threshold, expected_codes_by_case
        )
        threshold_analysis[f"{threshold:.2f}"] = {
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1_score,
            "tp": float(metrics.true_positives),
            "fp": float(metrics.false_positives),
            "fn": float(metrics.false_negatives),
        }

        if metrics.f1_score > best_f1:
            best_f1 = metrics.f1_score
            best_threshold = threshold

        threshold += step

    return best_threshold, threshold_analysis


def analyze_calibration(
    result: EvaluationRunResult,
    num_bins: int = 10,
) -> CalibrationAnalysis:
    """Perform full calibration analysis on an evaluation run.

    Args:
        result: Evaluation run result to analyze.
        num_bins: Number of bins for calibration analysis.

    Returns:
        Complete CalibrationAnalysis with all metrics.
    """
    # Extract outcomes
    outcomes = extract_suggestion_outcomes(result)

    if not outcomes:
        logger.warning("No suggestion outcomes found in evaluation run")
        return CalibrationAnalysis(
            run_id=result.metadata.run_id,
            total_suggestions=0,
        )

    # Calculate calibration metrics
    analysis = calculate_calibration_error(outcomes, num_bins=num_bins)
    analysis.run_id = result.metadata.run_id

    # Find optimal threshold
    optimal_threshold, threshold_analysis = find_optimal_threshold(outcomes, result)
    analysis.optimal_threshold = optimal_threshold
    analysis.threshold_analysis = threshold_analysis

    return analysis


def create_calibration_config(
    analysis: CalibrationAnalysis,
) -> CalibrationConfig:
    """Create a calibration config from analysis results.

    Args:
        analysis: Completed calibration analysis.

    Returns:
        CalibrationConfig with recommended thresholds.
    """
    # Find thresholds where accuracy drops significantly
    # High confidence: bins where accuracy >= 90%
    # Low confidence: bins where accuracy < 70%

    high_threshold = 0.8
    low_threshold = 0.3

    for bin_data in reversed(analysis.bins):
        if bin_data.count >= 3 and bin_data.accuracy >= 0.9:
            high_threshold = bin_data.bin_start
            break

    for bin_data in analysis.bins:
        if bin_data.count >= 3 and bin_data.accuracy < 0.7:
            low_threshold = bin_data.bin_end
            break

    return CalibrationConfig(
        optimal_threshold=analysis.optimal_threshold,
        high_confidence_threshold=high_threshold,
        low_confidence_threshold=low_threshold,
        source_run_id=analysis.run_id,
        ece=analysis.expected_calibration_error,
    )


def save_calibration_config(
    config: CalibrationConfig,
    path: Path | None = None,
) -> Path:
    """Save calibration config to JSON file.

    Args:
        config: Calibration config to save.
        path: Path to save to. Defaults to data/calibration_config.json.

    Returns:
        Path where config was saved.
    """
    import json

    save_path = path or DEFAULT_CALIBRATION_CONFIG_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(config.model_dump(), f, indent=2)

    logger.info("Saved calibration config to %s", save_path)
    return save_path


def load_calibration_config(
    path: Path | None = None,
) -> CalibrationConfig | None:
    """Load calibration config from JSON file.

    Args:
        path: Path to load from. Defaults to data/calibration_config.json.

    Returns:
        CalibrationConfig if found, None otherwise.
    """
    import json

    load_path = path or DEFAULT_CALIBRATION_CONFIG_PATH

    if not load_path.exists():
        return None

    try:
        with open(load_path) as f:
            data = json.load(f)
        return CalibrationConfig.model_validate(data)
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Failed to load calibration config: %s", e)
        return None
