"""Ablation testing framework for comparing prompt strategies.

This module provides tools for running ablation tests across different
prompt strategies and computing statistical significance of improvements.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from src.evaluation.models import (
    EvaluationFilters,
    EvaluationRunResult,
    PromptStrategy,
)
from src.evaluation.runner import run_evaluation
from src.evaluation.storage import EvaluationStorage, get_default_storage
from src.llm import LLMClient

logger = logging.getLogger(__name__)


class StrategyMetrics(BaseModel):
    """Metrics summary for a single prompt strategy."""

    strategy: PromptStrategy = Field(..., description="Prompt strategy")
    run_id: str = Field(..., description="Evaluation run ID")
    test_case_count: int = Field(..., description="Number of test cases")
    precision: float = Field(..., description="Average precision")
    recall: float = Field(..., description="Average recall")
    f1_score: float = Field(..., description="Average F1 score")
    exact_match_rate: float = Field(..., description="Rate of exact matches")
    total_tokens: int = Field(..., description="Total tokens used")
    duration_ms: float = Field(..., description="Total duration in milliseconds")

    @property
    def tokens_per_case(self) -> float:
        """Average tokens per test case."""
        return self.total_tokens / self.test_case_count if self.test_case_count else 0

    @property
    def f1_per_1k_tokens(self) -> float:
        """F1 score per 1000 tokens (cost-effectiveness metric)."""
        if self.total_tokens == 0:
            return 0.0
        return self.f1_score / (self.total_tokens / 1000)


class StrategyComparison(BaseModel):
    """Comparison between two prompt strategies."""

    baseline_strategy: PromptStrategy = Field(..., description="Baseline strategy")
    comparison_strategy: PromptStrategy = Field(..., description="Strategy to compare")
    baseline_metrics: StrategyMetrics = Field(..., description="Baseline metrics")
    comparison_metrics: StrategyMetrics = Field(..., description="Comparison metrics")

    # Deltas
    f1_delta: float = Field(..., description="F1 score difference")
    precision_delta: float = Field(..., description="Precision difference")
    recall_delta: float = Field(..., description="Recall difference")
    token_delta: int = Field(..., description="Token usage difference")

    # Statistical significance
    is_significant: bool = Field(
        default=False,
        description="Whether the F1 difference is statistically significant",
    )
    p_value: float | None = Field(
        default=None,
        description="P-value from paired t-test (if computed)",
    )
    t_statistic: float | None = Field(
        default=None,
        description="T-statistic from paired t-test (if computed)",
    )

    # Cost-effectiveness
    f1_per_token_baseline: float = Field(
        default=0.0, description="Baseline F1 per 1K tokens"
    )
    f1_per_token_comparison: float = Field(
        default=0.0, description="Comparison F1 per 1K tokens"
    )

    @property
    def is_improvement(self) -> bool:
        """Whether the comparison strategy is better than baseline."""
        return self.f1_delta > 0


class AblationReport(BaseModel):
    """Report from running ablation tests across multiple strategies."""

    report_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique identifier for this report",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the ablation test was run",
    )
    strategies_tested: list[PromptStrategy] = Field(
        ..., description="Strategies included in the test"
    )
    test_case_count: int = Field(..., description="Number of test cases")
    filters: EvaluationFilters = Field(
        default_factory=EvaluationFilters,
        description="Filters applied to test cases",
    )

    # Per-strategy results
    strategy_metrics: list[StrategyMetrics] = Field(
        default_factory=list, description="Metrics for each strategy"
    )

    # Comparisons (all vs baseline)
    baseline_strategy: PromptStrategy = Field(
        default=PromptStrategy.ZERO_SHOT,
        description="Baseline strategy for comparisons",
    )
    comparisons: list[StrategyComparison] = Field(
        default_factory=list, description="Comparisons vs baseline"
    )

    # Summary
    best_strategy: PromptStrategy | None = Field(
        default=None, description="Best performing strategy by F1"
    )
    best_cost_effective_strategy: PromptStrategy | None = Field(
        default=None, description="Best strategy by F1/token"
    )

    # Total stats
    total_tokens: int = Field(default=0, description="Total tokens across all runs")
    total_duration_ms: float = Field(
        default=0.0, description="Total duration in milliseconds"
    )


@dataclass
class PerCaseF1:
    """F1 scores for a single test case across strategies."""

    test_case_id: str
    f1_scores: dict[PromptStrategy, float]


def run_ablation_test(
    client: LLMClient,
    strategies: list[PromptStrategy] | None = None,
    filters: EvaluationFilters | None = None,
    baseline: PromptStrategy = PromptStrategy.ZERO_SHOT,
    storage: EvaluationStorage | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> AblationReport:
    """Run ablation test comparing multiple prompt strategies.

    Runs the evaluation pipeline with each strategy on the same test cases
    and computes comparative metrics.

    Args:
        client: LLM client for making requests.
        strategies: List of strategies to test. Defaults to all strategies.
        filters: Optional filters for test cases.
        baseline: Baseline strategy for comparisons.
        storage: Storage for saving results. Defaults to file storage.
        progress_callback: Called with (strategy_name, current_idx, total).

    Returns:
        AblationReport with all results and comparisons.
    """
    strategies = strategies or list(PromptStrategy)
    filters = filters or EvaluationFilters()
    storage = storage or get_default_storage()

    # Ensure baseline is in strategies
    if baseline not in strategies:
        strategies.insert(0, baseline)

    start_time = time.perf_counter()
    strategy_metrics: list[StrategyMetrics] = []
    run_results: dict[PromptStrategy, EvaluationRunResult] = {}

    # Run evaluation for each strategy
    for idx, strategy in enumerate(strategies):
        if progress_callback:
            progress_callback(strategy.value, idx + 1, len(strategies))

        logger.info("Running evaluation with strategy: %s", strategy.value)

        result = run_evaluation(
            client=client,
            strategy=strategy,
            filters=filters,
        )

        # Save result
        storage.save_run(result)
        run_results[strategy] = result

        # Extract metrics
        metrics = StrategyMetrics(
            strategy=strategy,
            run_id=result.metadata.run_id,
            test_case_count=result.metadata.test_case_count,
            precision=result.overall_precision or 0.0,
            recall=result.overall_recall or 0.0,
            f1_score=result.overall_f1 or 0.0,
            exact_match_rate=result.exact_match_rate or 0.0,
            total_tokens=result.metadata.token_stats.total_tokens,
            duration_ms=result.metadata.total_duration_ms,
        )
        strategy_metrics.append(metrics)

    end_time = time.perf_counter()
    total_duration_ms = (end_time - start_time) * 1000

    # Find best strategies
    best_by_f1 = max(strategy_metrics, key=lambda m: m.f1_score)
    best_by_cost = max(strategy_metrics, key=lambda m: m.f1_per_1k_tokens)

    # Compute comparisons vs baseline
    baseline_result = run_results[baseline]
    baseline_metrics = next(m for m in strategy_metrics if m.strategy == baseline)
    comparisons: list[StrategyComparison] = []

    for strategy in strategies:
        if strategy == baseline:
            continue

        comparison_metrics = next(m for m in strategy_metrics if m.strategy == strategy)
        comparison_result = run_results[strategy]

        comparison = compare_strategies(
            baseline_result=baseline_result,
            baseline_metrics=baseline_metrics,
            comparison_result=comparison_result,
            comparison_metrics=comparison_metrics,
        )
        comparisons.append(comparison)

    # Create report
    test_case_count = strategy_metrics[0].test_case_count if strategy_metrics else 0

    return AblationReport(
        strategies_tested=strategies,
        test_case_count=test_case_count,
        filters=filters,
        strategy_metrics=strategy_metrics,
        baseline_strategy=baseline,
        comparisons=comparisons,
        best_strategy=best_by_f1.strategy,
        best_cost_effective_strategy=best_by_cost.strategy,
        total_tokens=sum(m.total_tokens for m in strategy_metrics),
        total_duration_ms=total_duration_ms,
    )


def compare_strategies(
    baseline_result: EvaluationRunResult,
    baseline_metrics: StrategyMetrics,
    comparison_result: EvaluationRunResult,
    comparison_metrics: StrategyMetrics,
) -> StrategyComparison:
    """Compare two strategies with optional statistical testing.

    Args:
        baseline_result: Full evaluation result for baseline.
        baseline_metrics: Summary metrics for baseline.
        comparison_result: Full evaluation result for comparison.
        comparison_metrics: Summary metrics for comparison.

    Returns:
        StrategyComparison with deltas and significance.
    """
    # Calculate deltas
    f1_delta = comparison_metrics.f1_score - baseline_metrics.f1_score
    precision_delta = comparison_metrics.precision - baseline_metrics.precision
    recall_delta = comparison_metrics.recall - baseline_metrics.recall
    token_delta = comparison_metrics.total_tokens - baseline_metrics.total_tokens

    # Try to compute statistical significance
    p_value = None
    t_statistic = None
    is_significant = False

    try:
        p_value, t_statistic = compute_paired_ttest(baseline_result, comparison_result)
        is_significant = p_value is not None and p_value < 0.05
    except ImportError:
        logger.warning("scipy not available; skipping significance test")
    except ValueError as e:
        logger.warning("Could not compute significance: %s", e)

    return StrategyComparison(
        baseline_strategy=baseline_metrics.strategy,
        comparison_strategy=comparison_metrics.strategy,
        baseline_metrics=baseline_metrics,
        comparison_metrics=comparison_metrics,
        f1_delta=f1_delta,
        precision_delta=precision_delta,
        recall_delta=recall_delta,
        token_delta=token_delta,
        is_significant=is_significant,
        p_value=p_value,
        t_statistic=t_statistic,
        f1_per_token_baseline=baseline_metrics.f1_per_1k_tokens,
        f1_per_token_comparison=comparison_metrics.f1_per_1k_tokens,
    )


def compute_paired_ttest(
    result_a: EvaluationRunResult,
    result_b: EvaluationRunResult,
) -> tuple[float | None, float | None]:
    """Compute paired t-test for F1 scores between two runs.

    The test determines if the difference in F1 scores is statistically
    significant across test cases.

    Args:
        result_a: First evaluation result.
        result_b: Second evaluation result.

    Returns:
        Tuple of (p_value, t_statistic), or (None, None) if cannot compute.

    Raises:
        ImportError: If scipy is not available.
        ValueError: If results cannot be paired.
    """
    from scipy import stats

    # Build F1 score lists for paired test cases
    f1_a: list[float] = []
    f1_b: list[float] = []

    # Create lookup for result_b
    b_by_id = {r.test_case_id: r for r in result_b.results}

    for result in result_a.results:
        if not result.is_success or not result.coding_metrics:
            continue

        if result.test_case_id not in b_by_id:
            continue

        b_result = b_by_id[result.test_case_id]
        if not b_result.is_success or not b_result.coding_metrics:
            continue

        f1_a.append(result.coding_metrics.f1_score)
        f1_b.append(b_result.coding_metrics.f1_score)

    if len(f1_a) < 3:
        raise ValueError("Need at least 3 paired samples for t-test")

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(f1_b, f1_a)  # Compare B to A

    return float(p_value), float(t_stat)


def format_ablation_summary(report: AblationReport) -> str:
    """Format ablation report as a text summary.

    Args:
        report: Ablation report to format.

    Returns:
        Formatted string summary.
    """
    lines = [
        f"Ablation Test Report: {report.report_id}",
        f"Timestamp: {report.timestamp.isoformat()}",
        f"Test Cases: {report.test_case_count}",
        "",
        "Strategy Performance:",
        "-" * 60,
    ]

    for metrics in sorted(report.strategy_metrics, key=lambda m: -m.f1_score):
        best_marker = " *" if metrics.strategy == report.best_strategy else ""
        cost_marker = (
            " $" if metrics.strategy == report.best_cost_effective_strategy else ""
        )
        lines.append(
            f"  {metrics.strategy.value:15s} | "
            f"F1: {metrics.f1_score:.1%} | "
            f"P: {metrics.precision:.1%} | "
            f"R: {metrics.recall:.1%} | "
            f"Tokens: {metrics.total_tokens:,}{best_marker}{cost_marker}"
        )

    lines.extend(
        [
            "",
            "* = Best F1 Score",
            "$ = Most Cost-Effective (F1/token)",
            "",
            f"Comparisons vs {report.baseline_strategy.value}:",
            "-" * 60,
        ]
    )

    for comparison in report.comparisons:
        sig_marker = " **" if comparison.is_significant else ""
        direction = "+" if comparison.f1_delta >= 0 else ""
        lines.append(
            f"  {comparison.comparison_strategy.value:15s} | "
            f"F1 delta: {direction}{comparison.f1_delta:.1%}{sig_marker} | "
            f"Token delta: {comparison.token_delta:+,}"
        )

    lines.extend(
        [
            "",
            "** = Statistically significant (p < 0.05)",
        ]
    )

    return "\n".join(lines)


def save_ablation_report(
    report: AblationReport,
) -> str:
    """Save ablation report to file storage.

    Args:
        report: Ablation report to save.

    Returns:
        Report ID.
    """
    import json
    from pathlib import Path

    storage_path = Path("data/ablation_reports")
    storage_path.mkdir(parents=True, exist_ok=True)

    file_path = storage_path / f"{report.report_id}.json"

    with open(file_path, "w") as f:
        json.dump(report.model_dump(mode="json"), f, indent=2, default=str)

    logger.info("Saved ablation report to %s", file_path)
    return report.report_id


def load_ablation_report(report_id: str) -> AblationReport | None:
    """Load ablation report from storage.

    Args:
        report_id: Report ID to load.

    Returns:
        AblationReport if found, None otherwise.
    """
    import json
    from pathlib import Path

    file_path = Path("data/ablation_reports") / f"{report_id}.json"

    if not file_path.exists():
        return None

    try:
        with open(file_path) as f:
            data = json.load(f)
        return AblationReport.model_validate(data)
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Failed to load ablation report: %s", e)
        return None


def list_ablation_reports(limit: int = 20) -> list[dict[str, str]]:
    """List available ablation reports.

    Args:
        limit: Maximum number of reports to return.

    Returns:
        List of report summaries.
    """
    import json
    from pathlib import Path

    storage_path = Path("data/ablation_reports")
    if not storage_path.exists():
        return []

    reports: list[dict[str, str]] = []

    json_files = sorted(
        storage_path.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for file_path in json_files[:limit]:
        try:
            with open(file_path) as f:
                data = json.load(f)

            reports.append(
                {
                    "report_id": data.get("report_id", file_path.stem),
                    "timestamp": data.get("timestamp", ""),
                    "strategies": ", ".join(data.get("strategies_tested", [])),
                    "test_cases": str(data.get("test_case_count", 0)),
                    "best_strategy": data.get("best_strategy", "unknown"),
                }
            )
        except (OSError, json.JSONDecodeError):
            continue

    return reports
