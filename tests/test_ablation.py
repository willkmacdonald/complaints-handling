"""Tests for ablation testing framework."""

import pytest

from src.evaluation.ablation import (
    AblationReport,
    StrategyComparison,
    StrategyMetrics,
    compare_strategies,
    compute_paired_ttest,
    format_ablation_summary,
    list_ablation_reports,
    load_ablation_report,
)
from src.evaluation.models import (
    CodingMetrics,
    EvaluationRunMetadata,
    EvaluationRunResult,
    PromptStrategy,
    TestCaseEvaluationResult,
    TokenStats,
)
from src.models.enums import DeviceType, IntakeChannel


class TestStrategyMetrics:
    """Tests for StrategyMetrics model."""

    def test_create_metrics(self):
        """Test creating strategy metrics."""
        metrics = StrategyMetrics(
            strategy=PromptStrategy.FEW_SHOT,
            run_id="test_run",
            test_case_count=10,
            precision=0.85,
            recall=0.80,
            f1_score=0.82,
            exact_match_rate=0.60,
            total_tokens=5000,
            duration_ms=10000.0,
        )

        assert metrics.strategy == PromptStrategy.FEW_SHOT
        assert metrics.f1_score == 0.82
        assert metrics.total_tokens == 5000

    def test_tokens_per_case(self):
        """Test tokens per case calculation."""
        metrics = StrategyMetrics(
            strategy=PromptStrategy.ZERO_SHOT,
            run_id="test",
            test_case_count=10,
            precision=0.8,
            recall=0.8,
            f1_score=0.8,
            exact_match_rate=0.5,
            total_tokens=5000,
            duration_ms=1000.0,
        )

        assert metrics.tokens_per_case == 500.0

    def test_f1_per_1k_tokens(self):
        """Test F1 per 1K tokens calculation."""
        metrics = StrategyMetrics(
            strategy=PromptStrategy.ZERO_SHOT,
            run_id="test",
            test_case_count=10,
            precision=0.8,
            recall=0.8,
            f1_score=0.8,
            exact_match_rate=0.5,
            total_tokens=4000,  # 4K tokens
            duration_ms=1000.0,
        )

        # F1=0.8, tokens=4K, so F1/1K = 0.8/4 = 0.2
        assert metrics.f1_per_1k_tokens == pytest.approx(0.2, abs=0.001)

    def test_f1_per_1k_tokens_zero_tokens(self):
        """Test F1 per 1K tokens with zero tokens."""
        metrics = StrategyMetrics(
            strategy=PromptStrategy.ZERO_SHOT,
            run_id="test",
            test_case_count=10,
            precision=0.8,
            recall=0.8,
            f1_score=0.8,
            exact_match_rate=0.5,
            total_tokens=0,
            duration_ms=1000.0,
        )

        assert metrics.f1_per_1k_tokens == 0.0


class TestStrategyComparison:
    """Tests for StrategyComparison model."""

    def test_create_comparison(self):
        """Test creating strategy comparison."""
        baseline = StrategyMetrics(
            strategy=PromptStrategy.ZERO_SHOT,
            run_id="baseline",
            test_case_count=10,
            precision=0.75,
            recall=0.70,
            f1_score=0.72,
            exact_match_rate=0.50,
            total_tokens=3000,
            duration_ms=5000.0,
        )

        comparison_metrics = StrategyMetrics(
            strategy=PromptStrategy.FEW_SHOT,
            run_id="comparison",
            test_case_count=10,
            precision=0.85,
            recall=0.80,
            f1_score=0.82,
            exact_match_rate=0.60,
            total_tokens=5000,
            duration_ms=8000.0,
        )

        comparison = StrategyComparison(
            baseline_strategy=PromptStrategy.ZERO_SHOT,
            comparison_strategy=PromptStrategy.FEW_SHOT,
            baseline_metrics=baseline,
            comparison_metrics=comparison_metrics,
            f1_delta=0.10,  # 0.82 - 0.72
            precision_delta=0.10,
            recall_delta=0.10,
            token_delta=2000,
            is_significant=True,
            p_value=0.03,
            t_statistic=2.5,
        )

        assert comparison.is_improvement is True
        assert comparison.f1_delta == 0.10

    def test_is_improvement_negative(self):
        """Test is_improvement when comparison is worse."""
        baseline = StrategyMetrics(
            strategy=PromptStrategy.ZERO_SHOT,
            run_id="baseline",
            test_case_count=10,
            precision=0.85,
            recall=0.80,
            f1_score=0.82,
            exact_match_rate=0.60,
            total_tokens=3000,
            duration_ms=5000.0,
        )

        comparison_metrics = StrategyMetrics(
            strategy=PromptStrategy.FEW_SHOT,
            run_id="comparison",
            test_case_count=10,
            precision=0.75,
            recall=0.70,
            f1_score=0.72,
            exact_match_rate=0.50,
            total_tokens=5000,
            duration_ms=8000.0,
        )

        comparison = StrategyComparison(
            baseline_strategy=PromptStrategy.ZERO_SHOT,
            comparison_strategy=PromptStrategy.FEW_SHOT,
            baseline_metrics=baseline,
            comparison_metrics=comparison_metrics,
            f1_delta=-0.10,  # Worse than baseline
            precision_delta=-0.10,
            recall_delta=-0.10,
            token_delta=2000,
        )

        assert comparison.is_improvement is False


class TestAblationReport:
    """Tests for AblationReport model."""

    def test_create_report(self):
        """Test creating ablation report."""
        metrics = [
            StrategyMetrics(
                strategy=PromptStrategy.ZERO_SHOT,
                run_id="run1",
                test_case_count=10,
                precision=0.75,
                recall=0.70,
                f1_score=0.72,
                exact_match_rate=0.50,
                total_tokens=3000,
                duration_ms=5000.0,
            ),
            StrategyMetrics(
                strategy=PromptStrategy.FEW_SHOT,
                run_id="run2",
                test_case_count=10,
                precision=0.85,
                recall=0.80,
                f1_score=0.82,
                exact_match_rate=0.60,
                total_tokens=5000,
                duration_ms=8000.0,
            ),
        ]

        report = AblationReport(
            strategies_tested=[PromptStrategy.ZERO_SHOT, PromptStrategy.FEW_SHOT],
            test_case_count=10,
            strategy_metrics=metrics,
            best_strategy=PromptStrategy.FEW_SHOT,
            best_cost_effective_strategy=PromptStrategy.ZERO_SHOT,
            total_tokens=8000,
            total_duration_ms=13000.0,
        )

        assert len(report.strategies_tested) == 2
        assert report.best_strategy == PromptStrategy.FEW_SHOT
        assert report.total_tokens == 8000

    def test_report_has_uuid(self):
        """Test report has auto-generated ID."""
        report = AblationReport(
            strategies_tested=[PromptStrategy.ZERO_SHOT],
            test_case_count=5,
        )

        assert report.report_id is not None
        assert len(report.report_id) == 8


class TestCompareStrategies:
    """Tests for compare_strategies function."""

    def _create_mock_result(
        self,
        strategy: PromptStrategy,
        f1_scores: list[float],
    ) -> EvaluationRunResult:
        """Create mock evaluation result."""
        results = []
        for i, f1 in enumerate(f1_scores):
            # Back-calculate precision/recall from F1 (assume equal)
            pr = f1  # Simplified: assume precision = recall = F1

            results.append(
                TestCaseEvaluationResult(
                    test_case_id=f"test_{i:03d}",
                    name=f"Test Case {i}",
                    channel=IntakeChannel.FORM,
                    device_type=DeviceType.IMPLANTABLE,
                    severity="serious",
                    difficulty="medium",
                    predicted_codes=["A01"],
                    predicted_confidences={"A01": 0.9},
                    expected_codes=["A01"],
                    coding_metrics=CodingMetrics(
                        predicted_codes=["A01"],
                        expected_codes=["A01"],
                        alternative_codes=[],
                        true_positives=1,
                        false_positives=0,
                        false_negatives=0,
                    ),
                )
            )
            # Override the F1 score for testing
            results[-1].coding_metrics = CodingMetrics(
                predicted_codes=["A01"],
                expected_codes=["A01"],
                alternative_codes=[],
                true_positives=int(pr * 10),
                false_positives=int((1 - pr) * 5),
                false_negatives=int((1 - pr) * 5),
            )

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

        result = EvaluationRunResult(
            metadata=EvaluationRunMetadata(
                run_id=f"{strategy.value}_run",
                strategy=strategy,
                test_case_count=len(f1_scores),
                token_stats=TokenStats(total_tokens=1000 * len(f1_scores)),
                total_duration_ms=500 * len(f1_scores),
            ),
            results=results,
        )
        result.overall_f1 = avg_f1
        result.overall_precision = avg_f1
        result.overall_recall = avg_f1
        result.exact_match_rate = avg_f1 * 0.8

        return result

    def test_compare_strategies_calculates_deltas(self):
        """Test comparison calculates correct deltas."""
        baseline_result = self._create_mock_result(
            PromptStrategy.ZERO_SHOT,
            [0.7, 0.7, 0.7, 0.7, 0.7],
        )
        comparison_result = self._create_mock_result(
            PromptStrategy.FEW_SHOT,
            [0.8, 0.8, 0.8, 0.8, 0.8],
        )

        baseline_metrics = StrategyMetrics(
            strategy=PromptStrategy.ZERO_SHOT,
            run_id="baseline",
            test_case_count=5,
            precision=0.7,
            recall=0.7,
            f1_score=0.7,
            exact_match_rate=0.5,
            total_tokens=5000,
            duration_ms=2500.0,
        )

        comparison_metrics = StrategyMetrics(
            strategy=PromptStrategy.FEW_SHOT,
            run_id="comparison",
            test_case_count=5,
            precision=0.8,
            recall=0.8,
            f1_score=0.8,
            exact_match_rate=0.6,
            total_tokens=7000,
            duration_ms=3500.0,
        )

        comparison = compare_strategies(
            baseline_result,
            baseline_metrics,
            comparison_result,
            comparison_metrics,
        )

        assert comparison.f1_delta == pytest.approx(0.1, abs=0.001)
        assert comparison.token_delta == 2000


class TestComputePairedTtest:
    """Tests for paired t-test computation."""

    def _create_mock_result(
        self,
        strategy: PromptStrategy,
        f1_scores: list[float],
    ) -> EvaluationRunResult:
        """Create mock evaluation result with specific F1 scores per case."""
        results = []
        for i, f1 in enumerate(f1_scores):
            results.append(
                TestCaseEvaluationResult(
                    test_case_id=f"test_{i:03d}",
                    name=f"Test Case {i}",
                    channel=IntakeChannel.FORM,
                    device_type=DeviceType.IMPLANTABLE,
                    severity="serious",
                    difficulty="medium",
                    predicted_codes=["A01"],
                    predicted_confidences={"A01": 0.9},
                    expected_codes=["A01"],
                    coding_metrics=CodingMetrics(
                        predicted_codes=["A01"],
                        expected_codes=["A01"],
                        alternative_codes=[],
                        # Set TP/FP/FN to produce desired F1
                        true_positives=int(f1 * 10),
                        false_positives=int((1 - f1) * 5),
                        false_negatives=int((1 - f1) * 5),
                    ),
                )
            )

        return EvaluationRunResult(
            metadata=EvaluationRunMetadata(
                run_id=f"{strategy.value}_run",
                strategy=strategy,
                test_case_count=len(f1_scores),
            ),
            results=results,
        )

    def test_significant_difference(self):
        """Test detects significant difference in F1 scores."""
        pytest.importorskip("scipy")

        # Consistently better scores in B
        result_a = self._create_mock_result(
            PromptStrategy.ZERO_SHOT,
            [0.5, 0.5, 0.5, 0.5, 0.5],
        )
        result_b = self._create_mock_result(
            PromptStrategy.FEW_SHOT,
            [0.9, 0.9, 0.9, 0.9, 0.9],
        )

        p_value, t_stat = compute_paired_ttest(result_a, result_b)

        assert p_value is not None
        assert p_value < 0.05  # Should be significant
        assert t_stat is not None
        assert t_stat > 0  # B should be better

    def test_no_significant_difference(self):
        """Test when no significant difference exists."""
        pytest.importorskip("scipy")

        # Nearly identical scores
        result_a = self._create_mock_result(
            PromptStrategy.ZERO_SHOT,
            [0.7, 0.7, 0.7, 0.7, 0.7],
        )
        result_b = self._create_mock_result(
            PromptStrategy.FEW_SHOT,
            [0.71, 0.69, 0.7, 0.71, 0.69],
        )

        p_value, t_stat = compute_paired_ttest(result_a, result_b)

        assert p_value is not None
        assert p_value > 0.05  # Should not be significant

    def test_insufficient_samples(self):
        """Test raises error with insufficient samples."""
        pytest.importorskip("scipy")

        result_a = self._create_mock_result(
            PromptStrategy.ZERO_SHOT,
            [0.7, 0.7],  # Only 2 samples
        )
        result_b = self._create_mock_result(
            PromptStrategy.FEW_SHOT,
            [0.8, 0.8],
        )

        with pytest.raises(ValueError, match="at least 3"):
            compute_paired_ttest(result_a, result_b)


class TestFormatAblationSummary:
    """Tests for formatting ablation summary."""

    def test_format_summary(self):
        """Test formatting ablation report as text."""
        metrics = [
            StrategyMetrics(
                strategy=PromptStrategy.ZERO_SHOT,
                run_id="run1",
                test_case_count=10,
                precision=0.75,
                recall=0.70,
                f1_score=0.72,
                exact_match_rate=0.50,
                total_tokens=3000,
                duration_ms=5000.0,
            ),
            StrategyMetrics(
                strategy=PromptStrategy.FEW_SHOT,
                run_id="run2",
                test_case_count=10,
                precision=0.85,
                recall=0.80,
                f1_score=0.82,
                exact_match_rate=0.60,
                total_tokens=5000,
                duration_ms=8000.0,
            ),
        ]

        report = AblationReport(
            report_id="test123",
            strategies_tested=[PromptStrategy.ZERO_SHOT, PromptStrategy.FEW_SHOT],
            test_case_count=10,
            strategy_metrics=metrics,
            best_strategy=PromptStrategy.FEW_SHOT,
            best_cost_effective_strategy=PromptStrategy.ZERO_SHOT,
            total_tokens=8000,
            total_duration_ms=13000.0,
        )

        summary = format_ablation_summary(report)

        assert "test123" in summary
        assert "zero_shot" in summary
        assert "few_shot" in summary
        assert "Best F1 Score" in summary


class TestAblationReportStorage:
    """Tests for saving and loading ablation reports."""

    def test_save_and_load_report_serialization(self):
        """Test ablation report serialization/deserialization."""
        metrics = [
            StrategyMetrics(
                strategy=PromptStrategy.ZERO_SHOT,
                run_id="run1",
                test_case_count=10,
                precision=0.75,
                recall=0.70,
                f1_score=0.72,
                exact_match_rate=0.50,
                total_tokens=3000,
                duration_ms=5000.0,
            ),
        ]

        report = AblationReport(
            report_id="test_save",
            strategies_tested=[PromptStrategy.ZERO_SHOT],
            test_case_count=10,
            strategy_metrics=metrics,
            best_strategy=PromptStrategy.ZERO_SHOT,
            total_tokens=3000,
            total_duration_ms=5000.0,
        )

        # Test model serialization
        json_data = report.model_dump(mode="json")

        assert json_data["report_id"] == "test_save"
        assert json_data["test_case_count"] == 10

        # Test deserialization
        loaded = AblationReport.model_validate(json_data)
        assert loaded.report_id == report.report_id
        assert loaded.best_strategy == report.best_strategy

    def test_load_nonexistent_report(self):
        """Test loading report that doesn't exist returns None."""
        result = load_ablation_report("definitely_nonexistent_12345678")
        assert result is None

    def test_list_ablation_reports_when_dir_missing(self):
        """Test listing reports when directory doesn't exist."""
        # If the data/ablation_reports directory doesn't exist,
        # list_ablation_reports should return empty list
        from pathlib import Path as RealPath

        ablation_dir = RealPath("data/ablation_reports")
        if not ablation_dir.exists():
            reports = list_ablation_reports()
            assert reports == []
