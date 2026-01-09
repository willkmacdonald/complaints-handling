"""Tests for evaluation runner and storage."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.models import (
    CodingMetrics,
    EvaluationFilters,
    EvaluationRunMetadata,
    EvaluationRunResult,
    PromptStrategy,
    TestCaseEvaluationResult,
    TokenStats,
)
from src.evaluation.runner import _evaluate_test_case, format_run_summary, run_evaluation
from src.evaluation.storage import FileEvaluationStorage, get_default_storage
from src.models.enums import DeviceType, IntakeChannel


class TestEvaluationModels:
    """Tests for evaluation data models."""

    def test_prompt_strategy_values(self) -> None:
        """Test PromptStrategy enum values."""
        assert PromptStrategy.ZERO_SHOT.value == "zero_shot"
        assert PromptStrategy.FEW_SHOT.value == "few_shot"
        assert PromptStrategy.CHAIN_OF_THOUGHT.value == "chain_of_thought"
        assert PromptStrategy.FEW_SHOT_COT.value == "few_shot_cot"

    def test_evaluation_filters_defaults(self) -> None:
        """Test EvaluationFilters default values."""
        filters = EvaluationFilters()
        assert filters.device_type is None
        assert filters.channel is None
        assert filters.severity is None
        assert filters.difficulty is None

    def test_evaluation_filters_with_values(self) -> None:
        """Test EvaluationFilters with specified values."""
        filters = EvaluationFilters(
            device_type=DeviceType.IMPLANTABLE,
            channel=IntakeChannel.FORM,
            severity="death",
            difficulty="hard",
        )
        assert filters.device_type == DeviceType.IMPLANTABLE
        assert filters.channel == IntakeChannel.FORM
        assert filters.severity == "death"
        assert filters.difficulty == "hard"

    def test_token_stats_defaults(self) -> None:
        """Test TokenStats default values."""
        stats = TokenStats()
        assert stats.total_prompt_tokens == 0
        assert stats.total_completion_tokens == 0
        assert stats.total_tokens == 0

    def test_evaluation_run_metadata_defaults(self) -> None:
        """Test EvaluationRunMetadata generates defaults."""
        metadata = EvaluationRunMetadata()
        assert metadata.run_id is not None
        assert len(metadata.run_id) == 8
        assert metadata.timestamp is not None
        assert metadata.strategy == PromptStrategy.ZERO_SHOT
        assert metadata.model == ""
        assert metadata.test_case_count == 0

    def test_test_case_evaluation_result_success(self) -> None:
        """Test TestCaseEvaluationResult for successful evaluation."""
        result = TestCaseEvaluationResult(
            test_case_id="test-001",
            name="Test Case 1",
            channel=IntakeChannel.FORM,
            device_type=DeviceType.IMPLANTABLE,
            severity="death",
            predicted_codes=["A01", "C01"],
            expected_codes=["A01", "C02"],
            tokens_used=100,
            latency_ms=500.0,
        )
        assert result.is_success is True
        assert result.error is None

    def test_test_case_evaluation_result_failure(self) -> None:
        """Test TestCaseEvaluationResult for failed evaluation."""
        result = TestCaseEvaluationResult(
            test_case_id="test-001",
            name="Test Case 1",
            channel=IntakeChannel.FORM,
            device_type=DeviceType.IMPLANTABLE,
            severity="death",
            error="LLM connection failed",
        )
        assert result.is_success is False
        assert result.error == "LLM connection failed"

    def test_evaluation_run_result_calculate_aggregates(self) -> None:
        """Test EvaluationRunResult aggregate calculation."""
        # Create mock results with metrics
        results = [
            TestCaseEvaluationResult(
                test_case_id="test-001",
                name="Test 1",
                channel=IntakeChannel.FORM,
                device_type=DeviceType.IMPLANTABLE,
                severity="death",
                difficulty="easy",
                predicted_codes=["A01"],
                expected_codes=["A01"],
                coding_metrics=CodingMetrics(
                    predicted_codes=["A01"],
                    expected_codes=["A01"],
                    true_positives=1,
                    false_positives=0,
                    false_negatives=0,
                ),
            ),
            TestCaseEvaluationResult(
                test_case_id="test-002",
                name="Test 2",
                channel=IntakeChannel.EMAIL,
                device_type=DeviceType.DIAGNOSTIC,
                severity="malfunction",
                difficulty="medium",
                predicted_codes=["A02", "C01"],
                expected_codes=["A02"],
                coding_metrics=CodingMetrics(
                    predicted_codes=["A02", "C01"],
                    expected_codes=["A02"],
                    true_positives=1,
                    false_positives=1,
                    false_negatives=0,
                ),
            ),
        ]

        run_result = EvaluationRunResult(
            metadata=EvaluationRunMetadata(),
            results=results,
        )
        run_result.calculate_aggregates()

        # Check overall metrics
        assert run_result.overall_precision is not None
        assert run_result.overall_recall is not None
        assert run_result.overall_f1 is not None
        assert run_result.exact_match_rate is not None

        # Check breakdowns exist
        assert "easy" in run_result.by_difficulty
        assert "medium" in run_result.by_difficulty
        assert "form" in run_result.by_channel
        assert "email" in run_result.by_channel

    def test_evaluation_run_result_empty_results(self) -> None:
        """Test EvaluationRunResult with no results."""
        run_result = EvaluationRunResult(
            metadata=EvaluationRunMetadata(),
            results=[],
        )
        run_result.calculate_aggregates()

        assert run_result.overall_f1 is None
        assert run_result.by_difficulty == {}


class TestFileEvaluationStorage:
    """Tests for file-based evaluation storage."""

    @pytest.fixture
    def temp_storage_dir(self, tmp_path: Path) -> Path:
        """Create temporary storage directory."""
        storage_dir = tmp_path / "evaluation_runs"
        storage_dir.mkdir()
        return storage_dir

    @pytest.fixture
    def storage(self, temp_storage_dir: Path) -> FileEvaluationStorage:
        """Create storage instance with temp directory."""
        return FileEvaluationStorage(base_path=temp_storage_dir)

    @pytest.fixture
    def sample_run_result(self) -> EvaluationRunResult:
        """Create a sample evaluation run result."""
        return EvaluationRunResult(
            metadata=EvaluationRunMetadata(
                run_id="abc123",
                strategy=PromptStrategy.ZERO_SHOT,
                model="gpt-4o",
                test_case_count=2,
            ),
            results=[
                TestCaseEvaluationResult(
                    test_case_id="test-001",
                    name="Test 1",
                    channel=IntakeChannel.FORM,
                    device_type=DeviceType.IMPLANTABLE,
                    severity="death",
                    predicted_codes=["A01"],
                    expected_codes=["A01"],
                ),
            ],
            overall_f1=0.85,
        )

    def test_save_and_load_run(
        self, storage: FileEvaluationStorage, sample_run_result: EvaluationRunResult
    ) -> None:
        """Test saving and loading an evaluation run."""
        # Save
        run_id = storage.save_run(sample_run_result)
        assert run_id == "abc123"

        # Load
        loaded = storage.load_run(run_id)
        assert loaded is not None
        assert loaded.metadata.run_id == "abc123"
        assert loaded.metadata.strategy == PromptStrategy.ZERO_SHOT
        assert loaded.overall_f1 == 0.85
        assert len(loaded.results) == 1

    def test_load_nonexistent_run(self, storage: FileEvaluationStorage) -> None:
        """Test loading a run that doesn't exist."""
        result = storage.load_run("nonexistent")
        assert result is None

    def test_list_runs_empty(self, storage: FileEvaluationStorage) -> None:
        """Test listing runs when none exist."""
        runs = storage.list_runs()
        assert runs == []

    def test_list_runs_multiple(
        self,
        storage: FileEvaluationStorage,
        temp_storage_dir: Path,
    ) -> None:
        """Test listing multiple runs."""
        # Create a few run files
        for i in range(3):
            run_result = EvaluationRunResult(
                metadata=EvaluationRunMetadata(
                    run_id=f"run{i:03d}",
                    strategy=PromptStrategy.ZERO_SHOT,
                    test_case_count=i + 1,
                ),
                results=[],
                overall_f1=0.7 + i * 0.05,
            )
            storage.save_run(run_result)

        runs = storage.list_runs()
        assert len(runs) == 3
        assert all("run_id" in run for run in runs)
        assert all("f1_score" in run for run in runs)

    def test_list_runs_with_limit(
        self,
        storage: FileEvaluationStorage,
    ) -> None:
        """Test listing runs with a limit."""
        # Create 5 runs
        for i in range(5):
            run_result = EvaluationRunResult(
                metadata=EvaluationRunMetadata(
                    run_id=f"run{i:03d}",
                    strategy=PromptStrategy.ZERO_SHOT,
                ),
                results=[],
            )
            storage.save_run(run_result)

        runs = storage.list_runs(limit=3)
        assert len(runs) == 3

    def test_delete_run(
        self, storage: FileEvaluationStorage, sample_run_result: EvaluationRunResult
    ) -> None:
        """Test deleting an evaluation run."""
        storage.save_run(sample_run_result)

        # Verify it exists
        assert storage.load_run("abc123") is not None

        # Delete
        deleted = storage.delete_run("abc123")
        assert deleted is True

        # Verify it's gone
        assert storage.load_run("abc123") is None

    def test_delete_nonexistent_run(self, storage: FileEvaluationStorage) -> None:
        """Test deleting a run that doesn't exist."""
        deleted = storage.delete_run("nonexistent")
        assert deleted is False


class TestEvaluationRunner:
    """Tests for evaluation runner functions."""

    def test_format_run_summary(self) -> None:
        """Test formatting run summary."""
        result = EvaluationRunResult(
            metadata=EvaluationRunMetadata(
                run_id="test123",
                strategy=PromptStrategy.FEW_SHOT,
                test_case_count=10,
                total_duration_ms=5000,
                token_stats=TokenStats(total_tokens=1000),
            ),
            results=[],
            overall_precision=0.85,
            overall_recall=0.80,
            overall_f1=0.82,
        )

        summary = format_run_summary(result)
        assert "test123" in summary
        assert "few_shot" in summary
        assert "10" in summary
        assert "5000" in summary

    @patch("src.evaluation.runner.load_all_test_cases")
    def test_run_evaluation_no_test_cases(
        self, mock_load: MagicMock
    ) -> None:
        """Test run_evaluation when no test cases match filters."""
        mock_load.return_value = []

        mock_client = MagicMock()
        mock_client.config.deployment_name = "gpt-4o"

        result = run_evaluation(
            client=mock_client,
            strategy=PromptStrategy.ZERO_SHOT,
            filters=EvaluationFilters(severity="nonexistent"),
        )

        assert result.metadata.test_case_count == 0
        assert len(result.results) == 0


class TestGetDefaultStorage:
    """Tests for default storage factory."""

    def test_get_default_storage_creates_directory(self, tmp_path: Path) -> None:
        """Test that get_default_storage creates the storage directory."""
        with patch(
            "src.evaluation.storage.DEFAULT_EVALUATION_DIR",
            tmp_path / "evaluation_runs",
        ):
            storage = get_default_storage()
            assert storage.base_path.exists()
