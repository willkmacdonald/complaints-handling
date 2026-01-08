"""Evaluation report generation."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from src.evaluation.models import CodingMetrics, ExtractionMetrics, MDRMetrics
from src.models.enums import DeviceType, IntakeChannel

logger = logging.getLogger(__name__)


class TestCaseResult(BaseModel):
    """Evaluation result for a single test case."""

    test_case_id: str = Field(..., description="Test case identifier")
    name: str = Field(..., description="Test case name")
    channel: IntakeChannel = Field(..., description="Intake channel")
    device_type: DeviceType = Field(..., description="Device type")
    severity: str = Field(..., description="Severity level")

    extraction_metrics: ExtractionMetrics | None = Field(
        default=None, description="Extraction evaluation results"
    )
    coding_metrics: CodingMetrics | None = Field(
        default=None, description="Coding evaluation results"
    )
    mdr_metrics: MDRMetrics | None = Field(
        default=None, description="MDR evaluation results"
    )


class AggregateMetrics(BaseModel):
    """Aggregated metrics across multiple test cases."""

    count: int = Field(..., description="Number of test cases")

    # Extraction metrics
    avg_extraction_accuracy: float | None = Field(
        default=None, description="Average extraction accuracy"
    )
    avg_extraction_completeness: float | None = Field(
        default=None, description="Average extraction completeness"
    )

    # Coding metrics
    avg_coding_precision: float | None = Field(
        default=None, description="Average coding precision"
    )
    avg_coding_recall: float | None = Field(
        default=None, description="Average coding recall"
    )
    avg_coding_f1: float | None = Field(
        default=None, description="Average coding F1 score"
    )
    exact_match_rate: float | None = Field(
        default=None, description="Percentage of exact code matches"
    )

    # MDR metrics
    mdr_accuracy: float | None = Field(
        default=None, description="MDR determination accuracy"
    )
    mdr_sensitivity: float | None = Field(
        default=None, description="MDR sensitivity (recall for positive cases)"
    )
    mdr_specificity: float | None = Field(
        default=None, description="MDR specificity (correct negatives)"
    )
    mdr_false_negative_count: int = Field(
        default=0, description="Number of missed MDR cases (critical)"
    )


class EvaluationReport(BaseModel):
    """Complete evaluation report."""

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Report generation timestamp",
    )

    # Test case results
    results: list[TestCaseResult] = Field(
        default_factory=list, description="Per-test-case results"
    )

    # Overall metrics
    overall: AggregateMetrics = Field(..., description="Overall aggregate metrics")

    # Breakdowns
    by_channel: dict[str, AggregateMetrics] = Field(
        default_factory=dict, description="Metrics by intake channel"
    )
    by_device_type: dict[str, AggregateMetrics] = Field(
        default_factory=dict, description="Metrics by device type"
    )
    by_severity: dict[str, AggregateMetrics] = Field(
        default_factory=dict, description="Metrics by severity level"
    )


def _calculate_aggregate_metrics(results: list[TestCaseResult]) -> AggregateMetrics:
    """Calculate aggregate metrics from a list of test case results."""
    if not results:
        return AggregateMetrics(count=0)

    count = len(results)

    # Extraction metrics
    extraction_results = [r for r in results if r.extraction_metrics is not None]
    avg_extraction_accuracy = None
    avg_extraction_completeness = None

    if extraction_results:
        avg_extraction_accuracy = sum(
            r.extraction_metrics.accuracy
            for r in extraction_results  # type: ignore[union-attr]
        ) / len(extraction_results)
        avg_extraction_completeness = sum(
            r.extraction_metrics.completeness
            for r in extraction_results  # type: ignore[union-attr]
        ) / len(extraction_results)

    # Coding metrics
    coding_results = [r for r in results if r.coding_metrics is not None]
    avg_coding_precision = None
    avg_coding_recall = None
    avg_coding_f1 = None
    exact_match_rate = None

    if coding_results:
        avg_coding_precision = sum(
            r.coding_metrics.precision
            for r in coding_results  # type: ignore[union-attr]
        ) / len(coding_results)
        avg_coding_recall = sum(
            r.coding_metrics.recall
            for r in coding_results  # type: ignore[union-attr]
        ) / len(coding_results)
        avg_coding_f1 = sum(
            r.coding_metrics.f1_score
            for r in coding_results  # type: ignore[union-attr]
        ) / len(coding_results)
        exact_matches = sum(
            1
            for r in coding_results
            if r.coding_metrics.exact_match  # type: ignore[union-attr]
        )
        exact_match_rate = exact_matches / len(coding_results)

    # MDR metrics
    mdr_results = [r for r in results if r.mdr_metrics is not None]
    mdr_accuracy = None
    mdr_sensitivity = None
    mdr_specificity = None
    mdr_false_negative_count = 0

    if mdr_results:
        correct = sum(1 for r in mdr_results if r.mdr_metrics.is_correct)  # type: ignore[union-attr]
        mdr_accuracy = correct / len(mdr_results)

        # Sensitivity: TP / (TP + FN) for cases where MDR is required
        positive_cases = [r for r in mdr_results if r.mdr_metrics.expected_requires_mdr]  # type: ignore[union-attr]
        if positive_cases:
            true_positives = sum(
                1
                for r in positive_cases
                if r.mdr_metrics.predicted_requires_mdr  # type: ignore[union-attr]
            )
            mdr_sensitivity = true_positives / len(positive_cases)

        # Specificity: TN / (TN + FP) for cases where MDR is NOT required
        negative_cases = [
            r
            for r in mdr_results
            if not r.mdr_metrics.expected_requires_mdr  # type: ignore[union-attr]
        ]
        if negative_cases:
            true_negatives = sum(
                1
                for r in negative_cases
                if not r.mdr_metrics.predicted_requires_mdr  # type: ignore[union-attr]
            )
            mdr_specificity = true_negatives / len(negative_cases)

        # Count false negatives (critical failures)
        mdr_false_negative_count = sum(
            1
            for r in mdr_results
            if r.mdr_metrics.is_false_negative  # type: ignore[union-attr]
        )

    return AggregateMetrics(
        count=count,
        avg_extraction_accuracy=avg_extraction_accuracy,
        avg_extraction_completeness=avg_extraction_completeness,
        avg_coding_precision=avg_coding_precision,
        avg_coding_recall=avg_coding_recall,
        avg_coding_f1=avg_coding_f1,
        exact_match_rate=exact_match_rate,
        mdr_accuracy=mdr_accuracy,
        mdr_sensitivity=mdr_sensitivity,
        mdr_specificity=mdr_specificity,
        mdr_false_negative_count=mdr_false_negative_count,
    )


def generate_report(results: list[TestCaseResult]) -> EvaluationReport:
    """Generate a comprehensive evaluation report from test case results.

    Args:
        results: List of TestCaseResult objects from evaluation runs.

    Returns:
        EvaluationReport with overall and breakdown metrics.
    """
    # Calculate overall metrics
    overall = _calculate_aggregate_metrics(results)

    # Group by channel
    by_channel: dict[str, AggregateMetrics] = {}
    channel_groups: dict[str, list[TestCaseResult]] = {}
    for r in results:
        channel_key = r.channel.value
        if channel_key not in channel_groups:
            channel_groups[channel_key] = []
        channel_groups[channel_key].append(r)
    for channel, group in channel_groups.items():
        by_channel[channel] = _calculate_aggregate_metrics(group)

    # Group by device type
    by_device_type: dict[str, AggregateMetrics] = {}
    device_groups: dict[str, list[TestCaseResult]] = {}
    for r in results:
        device_key = r.device_type.value
        if device_key not in device_groups:
            device_groups[device_key] = []
        device_groups[device_key].append(r)
    for device, group in device_groups.items():
        by_device_type[device] = _calculate_aggregate_metrics(group)

    # Group by severity
    by_severity: dict[str, AggregateMetrics] = {}
    severity_groups: dict[str, list[TestCaseResult]] = {}
    for r in results:
        if r.severity not in severity_groups:
            severity_groups[r.severity] = []
        severity_groups[r.severity].append(r)
    for severity, group in severity_groups.items():
        by_severity[severity] = _calculate_aggregate_metrics(group)

    return EvaluationReport(
        results=results,
        overall=overall,
        by_channel=by_channel,
        by_device_type=by_device_type,
        by_severity=by_severity,
    )


def export_report_json(report: EvaluationReport, output_path: Path) -> None:
    """Export evaluation report to JSON file.

    Args:
        report: The evaluation report to export.
        output_path: Path where JSON file will be written.

    Raises:
        OSError: If the file cannot be written.
    """
    try:
        with open(output_path, "w") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2, default=str)
    except OSError as e:
        logger.error("Failed to export report to JSON at %s: %s", output_path, e)
        raise


def export_report_markdown(report: EvaluationReport, output_path: Path) -> None:
    """Export evaluation report to Markdown file.

    Args:
        report: The evaluation report to export.
        output_path: Path where Markdown file will be written.

    Raises:
        OSError: If the file cannot be written.
    """
    lines = [
        "# Evaluation Report",
        "",
        f"Generated: {report.generated_at.isoformat()}",
        "",
        "## Overall Metrics",
        "",
        f"**Total Test Cases:** {report.overall.count}",
        "",
    ]

    # Extraction metrics
    if report.overall.avg_extraction_accuracy is not None:
        lines.extend(
            [
                "### Extraction",
                "",
                f"- Average Accuracy: {report.overall.avg_extraction_accuracy:.1%}",
                f"- Average Completeness: {report.overall.avg_extraction_completeness:.1%}",
                "",
            ]
        )

    # Coding metrics
    if report.overall.avg_coding_f1 is not None:
        lines.extend(
            [
                "### Coding",
                "",
                f"- Precision: {report.overall.avg_coding_precision:.1%}",
                f"- Recall: {report.overall.avg_coding_recall:.1%}",
                f"- F1 Score: {report.overall.avg_coding_f1:.1%}",
                f"- Exact Match Rate: {report.overall.exact_match_rate:.1%}",
                "",
            ]
        )

    # MDR metrics
    if report.overall.mdr_accuracy is not None:
        lines.extend(
            [
                "### MDR Determination",
                "",
                f"- Accuracy: {report.overall.mdr_accuracy:.1%}",
                f"- Sensitivity: {_format_pct(report.overall.mdr_sensitivity)}",
                f"- Specificity: {_format_pct(report.overall.mdr_specificity)}",
                f"- **False Negatives: {report.overall.mdr_false_negative_count}** (CRITICAL)",
                "",
            ]
        )

    # Breakdown by channel
    if report.by_channel:
        lines.extend(["## By Channel", ""])
        for channel, metrics in report.by_channel.items():
            lines.append(f"### {channel.title()}")
            lines.append("")
            lines.append(f"Count: {metrics.count}")
            if metrics.avg_coding_f1 is not None:
                lines.append(f"- Coding F1: {metrics.avg_coding_f1:.1%}")
            if metrics.mdr_accuracy is not None:
                lines.append(f"- MDR Accuracy: {metrics.mdr_accuracy:.1%}")
            lines.append("")

    # Breakdown by severity
    if report.by_severity:
        lines.extend(["## By Severity", ""])
        for severity, metrics in report.by_severity.items():
            lines.append(f"### {severity.replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"Count: {metrics.count}")
            if metrics.mdr_false_negative_count > 0:
                lines.append(
                    f"- **MDR False Negatives: {metrics.mdr_false_negative_count}**"
                )
            lines.append("")

    try:
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
    except OSError as e:
        logger.error("Failed to export report to Markdown at %s: %s", output_path, e)
        raise


def _format_pct(value: float | None) -> str:
    """Format a percentage value, handling None."""
    if value is None:
        return "N/A"
    return f"{value:.1%}"


def format_report_summary(report: EvaluationReport) -> str:
    """Format a brief summary of the evaluation report.

    Args:
        report: The evaluation report.

    Returns:
        A formatted string summary.
    """
    lines = [
        f"Evaluation Summary ({report.overall.count} test cases)",
        "=" * 50,
    ]

    if report.overall.avg_coding_f1 is not None:
        lines.append(f"Coding F1: {report.overall.avg_coding_f1:.1%}")

    if report.overall.mdr_accuracy is not None:
        lines.append(f"MDR Accuracy: {report.overall.mdr_accuracy:.1%}")
        if report.overall.mdr_false_negative_count > 0:
            lines.append(
                f"WARNING: {report.overall.mdr_false_negative_count} "
                f"MDR false negative(s)!"
            )

    return "\n".join(lines)


# Re-export models and functions
__all__ = [
    "AggregateMetrics",
    "EvaluationReport",
    "TestCaseResult",
    "export_report_json",
    "export_report_markdown",
    "format_report_summary",
    "generate_report",
]
