"""Tests for confidence calibration module."""

import tempfile
from pathlib import Path

import pytest

from src.evaluation.calibration import (
    CalibrationAnalysis,
    CalibrationBin,
    CalibrationConfig,
    SuggestionOutcome,
    analyze_calibration,
    calculate_calibration_error,
    create_calibration_config,
    extract_suggestion_outcomes,
    find_optimal_threshold,
    load_calibration_config,
    save_calibration_config,
)
from src.evaluation.models import (
    CodingMetrics,
    EvaluationRunMetadata,
    EvaluationRunResult,
    PromptStrategy,
    TestCaseEvaluationResult,
)
from src.models.enums import DeviceType, IntakeChannel


class TestSuggestionOutcome:
    """Tests for SuggestionOutcome model."""

    def test_create_outcome(self):
        """Test creating a suggestion outcome."""
        outcome = SuggestionOutcome(
            code_id="A01",
            confidence=0.85,
            is_correct=True,
            test_case_id="test_001",
        )

        assert outcome.code_id == "A01"
        assert outcome.confidence == 0.85
        assert outcome.is_correct is True
        assert outcome.test_case_id == "test_001"

    def test_confidence_validation(self):
        """Test confidence must be between 0 and 1."""
        with pytest.raises(ValueError):
            SuggestionOutcome(
                code_id="A01",
                confidence=1.5,
                is_correct=True,
                test_case_id="test_001",
            )

        with pytest.raises(ValueError):
            SuggestionOutcome(
                code_id="A01",
                confidence=-0.1,
                is_correct=True,
                test_case_id="test_001",
            )


class TestCalibrationBin:
    """Tests for CalibrationBin model."""

    def test_create_bin(self):
        """Test creating a calibration bin."""
        bin_data = CalibrationBin(
            bin_start=0.8,
            bin_end=0.9,
            count=10,
            correct_count=8,
            avg_confidence=0.85,
            accuracy=0.8,
            calibration_error=0.05,
        )

        assert bin_data.bin_start == 0.8
        assert bin_data.bin_end == 0.9
        assert bin_data.count == 10
        assert bin_data.accuracy == 0.8

    def test_empty_bin_defaults(self):
        """Test empty bin has zero defaults."""
        bin_data = CalibrationBin(bin_start=0.0, bin_end=0.1)

        assert bin_data.count == 0
        assert bin_data.correct_count == 0
        assert bin_data.avg_confidence == 0.0


class TestExtractSuggestionOutcomes:
    """Tests for extracting outcomes from evaluation results."""

    def _create_mock_result(
        self,
        predicted_codes: list[str],
        predicted_confidences: dict[str, float],
        expected_codes: list[str],
        alternative_codes: list[str] | None = None,
    ) -> EvaluationRunResult:
        """Create a mock evaluation result."""
        case_result = TestCaseEvaluationResult(
            test_case_id="test_001",
            name="Test Case 1",
            channel=IntakeChannel.FORM,
            device_type=DeviceType.IMPLANTABLE,
            severity="serious",
            difficulty="medium",
            predicted_codes=predicted_codes,
            predicted_confidences=predicted_confidences,
            expected_codes=expected_codes,
            alternative_codes=alternative_codes or [],
            coding_metrics=CodingMetrics(
                predicted_codes=predicted_codes,
                expected_codes=expected_codes,
                alternative_codes=alternative_codes or [],
                true_positives=len(set(predicted_codes) & set(expected_codes)),
                false_positives=len(set(predicted_codes) - set(expected_codes)),
                false_negatives=len(set(expected_codes) - set(predicted_codes)),
            ),
        )

        return EvaluationRunResult(
            metadata=EvaluationRunMetadata(
                run_id="test_run",
                strategy=PromptStrategy.ZERO_SHOT,
                test_case_count=1,
            ),
            results=[case_result],
        )

    def test_extract_correct_outcomes(self):
        """Test extracting outcomes where predictions match expected."""
        result = self._create_mock_result(
            predicted_codes=["A01", "B02"],
            predicted_confidences={"A01": 0.9, "B02": 0.7},
            expected_codes=["A01", "B02"],
        )

        outcomes = extract_suggestion_outcomes(result)

        assert len(outcomes) == 2
        assert all(o.is_correct for o in outcomes)

    def test_extract_incorrect_outcomes(self):
        """Test extracting outcomes where predictions don't match."""
        result = self._create_mock_result(
            predicted_codes=["A01", "C03"],
            predicted_confidences={"A01": 0.9, "C03": 0.7},
            expected_codes=["A01", "B02"],
        )

        outcomes = extract_suggestion_outcomes(result)

        assert len(outcomes) == 2
        correct_outcomes = [o for o in outcomes if o.is_correct]
        incorrect_outcomes = [o for o in outcomes if not o.is_correct]

        assert len(correct_outcomes) == 1
        assert len(incorrect_outcomes) == 1
        assert correct_outcomes[0].code_id == "A01"
        assert incorrect_outcomes[0].code_id == "C03"

    def test_alternative_codes_counted_as_correct(self):
        """Test that alternative codes are counted as correct."""
        result = self._create_mock_result(
            predicted_codes=["A01", "ALT1"],
            predicted_confidences={"A01": 0.9, "ALT1": 0.8},
            expected_codes=["A01"],
            alternative_codes=["ALT1"],
        )

        outcomes = extract_suggestion_outcomes(result)

        assert len(outcomes) == 2
        assert all(o.is_correct for o in outcomes)


class TestCalculateCalibrationError:
    """Tests for calibration error calculation."""

    def test_perfectly_calibrated(self):
        """Test perfectly calibrated model has ECE = 0."""
        # 100% accurate predictions with 100% confidence
        outcomes = [
            SuggestionOutcome(
                code_id=f"A{i:02d}",
                confidence=1.0,
                is_correct=True,
                test_case_id="test",
            )
            for i in range(10)
        ]

        analysis = calculate_calibration_error(outcomes, num_bins=10)

        assert analysis.expected_calibration_error == pytest.approx(0.0, abs=0.01)
        assert analysis.overall_accuracy == 1.0

    def test_completely_miscalibrated(self):
        """Test completely miscalibrated model."""
        # High confidence but 0% accuracy
        outcomes = [
            SuggestionOutcome(
                code_id=f"A{i:02d}",
                confidence=0.95,  # High confidence
                is_correct=False,  # All wrong
                test_case_id="test",
            )
            for i in range(10)
        ]

        analysis = calculate_calibration_error(outcomes, num_bins=10)

        # ECE should be high (confidence ~0.95, accuracy 0.0)
        assert analysis.expected_calibration_error > 0.8
        assert analysis.overall_accuracy == 0.0

    def test_brier_score_calculation(self):
        """Test Brier score is calculated correctly."""
        # Half correct at 0.8 confidence, half wrong at 0.2 confidence
        outcomes = [
            SuggestionOutcome(
                code_id=f"A{i:02d}",
                confidence=0.8,
                is_correct=True,
                test_case_id="test",
            )
            for i in range(5)
        ] + [
            SuggestionOutcome(
                code_id=f"B{i:02d}",
                confidence=0.2,
                is_correct=False,
                test_case_id="test",
            )
            for i in range(5)
        ]

        analysis = calculate_calibration_error(outcomes, num_bins=10)

        # Brier score: mean of (0.8-1)^2 for correct + (0.2-0)^2 for incorrect
        # = (5 * 0.04 + 5 * 0.04) / 10 = 0.04
        assert analysis.brier_score == pytest.approx(0.04, abs=0.001)

    def test_empty_outcomes(self):
        """Test handling of empty outcomes."""
        analysis = calculate_calibration_error([], num_bins=10)

        assert analysis.total_suggestions == 0
        assert analysis.expected_calibration_error == 0.0

    def test_bin_distribution(self):
        """Test outcomes are distributed across bins correctly."""
        # Create outcomes evenly distributed across confidence range
        outcomes = [
            SuggestionOutcome(
                code_id=f"A{i:02d}",
                confidence=i * 0.1,
                is_correct=True,
                test_case_id="test",
            )
            for i in range(10)
        ]

        analysis = calculate_calibration_error(outcomes, num_bins=10)

        # Each bin should have 1 outcome (except bin 0 which gets conf 0.0)
        non_empty_bins = [b for b in analysis.bins if b.count > 0]
        assert len(non_empty_bins) == 10


class TestFindOptimalThreshold:
    """Tests for optimal threshold finding."""

    def _create_result_with_outcomes(
        self, outcomes: list[SuggestionOutcome]
    ) -> EvaluationRunResult:
        """Create a result that matches the given outcomes."""
        # Group outcomes by test case
        cases_data: dict[str, list[SuggestionOutcome]] = {}
        for o in outcomes:
            if o.test_case_id not in cases_data:
                cases_data[o.test_case_id] = []
            cases_data[o.test_case_id].append(o)

        results: list[TestCaseEvaluationResult] = []
        for case_id, case_outcomes in cases_data.items():
            predicted_codes = [o.code_id for o in case_outcomes]
            predicted_confidences = {o.code_id: o.confidence for o in case_outcomes}
            expected_codes = [o.code_id for o in case_outcomes if o.is_correct]

            results.append(
                TestCaseEvaluationResult(
                    test_case_id=case_id,
                    name=f"Test {case_id}",
                    channel=IntakeChannel.FORM,
                    device_type=DeviceType.IMPLANTABLE,
                    severity="serious",
                    difficulty="medium",
                    predicted_codes=predicted_codes,
                    predicted_confidences=predicted_confidences,
                    expected_codes=expected_codes,
                    coding_metrics=CodingMetrics(
                        predicted_codes=predicted_codes,
                        expected_codes=expected_codes,
                        alternative_codes=[],
                        true_positives=len(expected_codes),
                        false_positives=len(
                            [o for o in case_outcomes if not o.is_correct]
                        ),
                        false_negatives=0,
                    ),
                )
            )

        return EvaluationRunResult(
            metadata=EvaluationRunMetadata(
                run_id="test_run",
                strategy=PromptStrategy.ZERO_SHOT,
                test_case_count=len(results),
            ),
            results=results,
        )

    def test_finds_optimal_threshold(self):
        """Test finding optimal threshold."""
        # High confidence correct predictions
        high_conf_correct = [
            SuggestionOutcome(
                code_id=f"A{i:02d}",
                confidence=0.9,
                is_correct=True,
                test_case_id=f"test_{i}",
            )
            for i in range(5)
        ]
        # Low confidence incorrect predictions
        low_conf_incorrect = [
            SuggestionOutcome(
                code_id=f"B{i:02d}",
                confidence=0.2,
                is_correct=False,
                test_case_id=f"test_{i}",
            )
            for i in range(5)
        ]

        outcomes = high_conf_correct + low_conf_incorrect
        result = self._create_result_with_outcomes(outcomes)

        optimal_threshold, analysis = find_optimal_threshold(outcomes, result)

        # Optimal threshold should be somewhere between 0.2 and 0.9
        # to filter out low confidence wrong predictions
        assert 0.2 < optimal_threshold < 0.9

    def test_threshold_analysis_structure(self):
        """Test threshold analysis dict structure."""
        outcomes = [
            SuggestionOutcome(
                code_id="A01",
                confidence=0.8,
                is_correct=True,
                test_case_id="test_1",
            )
        ]
        result = self._create_result_with_outcomes(outcomes)

        _, analysis = find_optimal_threshold(outcomes, result)

        # Should have entries for each threshold tested
        assert len(analysis) > 0

        # Each entry should have the required metrics
        for _threshold, metrics in analysis.items():
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1" in metrics
            assert "tp" in metrics
            assert "fp" in metrics
            assert "fn" in metrics


class TestCalibrationConfig:
    """Tests for calibration config model."""

    def test_create_config(self):
        """Test creating calibration config."""
        config = CalibrationConfig(
            optimal_threshold=0.6,
            high_confidence_threshold=0.85,
            low_confidence_threshold=0.25,
            source_run_id="abc123",
            ece=0.08,
        )

        assert config.optimal_threshold == 0.6
        assert config.high_confidence_threshold == 0.85
        assert config.low_confidence_threshold == 0.25

    def test_threshold_validation(self):
        """Test threshold values are validated."""
        with pytest.raises(ValueError):
            CalibrationConfig(optimal_threshold=1.5)

        with pytest.raises(ValueError):
            CalibrationConfig(optimal_threshold=-0.1)

    def test_save_and_load_config(self):
        """Test saving and loading calibration config."""
        config = CalibrationConfig(
            optimal_threshold=0.6,
            high_confidence_threshold=0.85,
            low_confidence_threshold=0.25,
            source_run_id="abc123",
            ece=0.08,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration_config.json"
            saved_path = save_calibration_config(config, path)

            assert saved_path == path
            assert path.exists()

            loaded = load_calibration_config(path)

            assert loaded is not None
            assert loaded.optimal_threshold == config.optimal_threshold
            assert loaded.source_run_id == config.source_run_id

    def test_load_nonexistent_config(self):
        """Test loading config that doesn't exist returns None."""
        result = load_calibration_config(Path("/nonexistent/path.json"))
        assert result is None


class TestCreateCalibrationConfig:
    """Tests for creating config from analysis."""

    def test_creates_config_from_analysis(self):
        """Test creating config from calibration analysis."""
        analysis = CalibrationAnalysis(
            run_id="test_run",
            total_suggestions=100,
            total_correct=80,
            overall_accuracy=0.8,
            avg_confidence=0.75,
            expected_calibration_error=0.08,
            maximum_calibration_error=0.15,
            brier_score=0.05,
            optimal_threshold=0.55,
            bins=[
                CalibrationBin(
                    bin_start=0.8,
                    bin_end=0.9,
                    count=20,
                    correct_count=18,
                    avg_confidence=0.85,
                    accuracy=0.9,
                    calibration_error=0.05,
                ),
                CalibrationBin(
                    bin_start=0.2,
                    bin_end=0.3,
                    count=10,
                    correct_count=5,
                    avg_confidence=0.25,
                    accuracy=0.5,
                    calibration_error=0.25,
                ),
            ],
        )

        config = create_calibration_config(analysis)

        assert config.optimal_threshold == 0.55
        assert config.source_run_id == "test_run"
        assert config.ece == 0.08


class TestAnalyzeCalibration:
    """Tests for full calibration analysis."""

    def _create_mock_result(self) -> EvaluationRunResult:
        """Create a mock evaluation result for testing."""
        results = []

        # Create test cases with varying confidence and correctness
        for i in range(5):
            predicted_codes = [f"A{i:02d}", f"B{i:02d}"]
            expected_codes = [f"A{i:02d}"]  # B codes are false positives

            results.append(
                TestCaseEvaluationResult(
                    test_case_id=f"test_{i:03d}",
                    name=f"Test Case {i}",
                    channel=IntakeChannel.FORM,
                    device_type=DeviceType.IMPLANTABLE,
                    severity="serious",
                    difficulty="medium",
                    predicted_codes=predicted_codes,
                    predicted_confidences={
                        f"A{i:02d}": 0.9 - i * 0.1,  # Varying confidence
                        f"B{i:02d}": 0.3 + i * 0.1,
                    },
                    expected_codes=expected_codes,
                    coding_metrics=CodingMetrics(
                        predicted_codes=predicted_codes,
                        expected_codes=expected_codes,
                        alternative_codes=[],
                        true_positives=1,
                        false_positives=1,
                        false_negatives=0,
                    ),
                )
            )

        return EvaluationRunResult(
            metadata=EvaluationRunMetadata(
                run_id="analysis_test",
                strategy=PromptStrategy.ZERO_SHOT,
                test_case_count=5,
            ),
            results=results,
        )

    def test_full_analysis(self):
        """Test full calibration analysis pipeline."""
        result = self._create_mock_result()

        analysis = analyze_calibration(result, num_bins=10)

        assert analysis.run_id == "analysis_test"
        assert analysis.total_suggestions == 10  # 2 per test case, 5 cases
        assert len(analysis.bins) == 10
        assert analysis.optimal_threshold is not None
        assert len(analysis.threshold_analysis) > 0

    def test_analysis_with_empty_result(self):
        """Test analysis handles empty results."""
        result = EvaluationRunResult(
            metadata=EvaluationRunMetadata(
                run_id="empty_test",
                strategy=PromptStrategy.ZERO_SHOT,
                test_case_count=0,
            ),
            results=[],
        )

        analysis = analyze_calibration(result)

        assert analysis.total_suggestions == 0
        assert analysis.expected_calibration_error == 0.0
