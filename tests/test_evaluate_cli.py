"""Tests for evaluate CLI commands."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.cli.evaluate import app
from src.evaluation.models import (
    EvaluationRunMetadata,
    EvaluationRunResult,
    PromptStrategy,
    TestCaseEvaluationResult,
)
from src.models.enums import DeviceType, IntakeChannel

runner = CliRunner()


class TestEvaluateHistoryCommand:
    """Tests for evaluate history command."""

    def test_history_empty(self, tmp_path: Path) -> None:
        """Test history command with no runs."""
        with patch("src.cli.evaluate.get_default_storage") as mock_storage:
            mock_instance = MagicMock()
            mock_instance.list_runs.return_value = []
            mock_storage.return_value = mock_instance

            result = runner.invoke(app, ["history"])
            assert result.exit_code == 0
            assert "No evaluation runs found" in result.stdout

    def test_history_with_runs(self, tmp_path: Path) -> None:
        """Test history command with existing runs."""
        with patch("src.cli.evaluate.get_default_storage") as mock_storage:
            mock_instance = MagicMock()
            mock_instance.list_runs.return_value = [
                {
                    "run_id": "abc123",
                    "timestamp": "2024-01-01T12:00:00",
                    "strategy": "zero_shot",
                    "test_case_count": "10",
                    "f1_score": "85.0%",
                },
                {
                    "run_id": "def456",
                    "timestamp": "2024-01-02T12:00:00",
                    "strategy": "few_shot",
                    "test_case_count": "10",
                    "f1_score": "90.0%",
                },
            ]
            mock_storage.return_value = mock_instance

            result = runner.invoke(app, ["history"])
            assert result.exit_code == 0
            assert "abc123" in result.stdout
            assert "def456" in result.stdout
            assert "zero_shot" in result.stdout
            assert "few_shot" in result.stdout

    def test_history_with_limit(self, tmp_path: Path) -> None:
        """Test history command with custom limit."""
        with patch("src.cli.evaluate.get_default_storage") as mock_storage:
            mock_instance = MagicMock()
            mock_instance.list_runs.return_value = []
            mock_storage.return_value = mock_instance

            result = runner.invoke(app, ["history", "--limit", "5"])
            assert result.exit_code == 0
            mock_instance.list_runs.assert_called_once_with(limit=5)


class TestEvaluateReportCommand:
    """Tests for evaluate report command."""

    def test_report_not_found(self) -> None:
        """Test report command with nonexistent run."""
        with patch("src.cli.evaluate.get_default_storage") as mock_storage:
            mock_instance = MagicMock()
            mock_instance.load_run.return_value = None
            mock_storage.return_value = mock_instance

            result = runner.invoke(app, ["report", "nonexistent"])
            assert result.exit_code == 1
            assert "not found" in result.stdout

    def test_report_found(self) -> None:
        """Test report command with existing run."""
        mock_result = EvaluationRunResult(
            metadata=EvaluationRunMetadata(
                run_id="abc123",
                strategy=PromptStrategy.ZERO_SHOT,
                test_case_count=5,
                total_duration_ms=3000,
            ),
            results=[
                TestCaseEvaluationResult(
                    test_case_id="test-001",
                    name="Test 1",
                    channel=IntakeChannel.FORM,
                    device_type=DeviceType.IMPLANTABLE,
                    severity="death",
                ),
            ],
            overall_f1=0.85,
            overall_precision=0.90,
            overall_recall=0.80,
        )

        with patch("src.cli.evaluate.get_default_storage") as mock_storage:
            mock_instance = MagicMock()
            mock_instance.load_run.return_value = mock_result
            mock_storage.return_value = mock_instance

            result = runner.invoke(app, ["report", "abc123"])
            assert result.exit_code == 0
            assert "abc123" in result.stdout
            assert "zero_shot" in result.stdout


class TestEvaluateRunCommand:
    """Tests for evaluate run command."""

    def test_run_missing_config(self) -> None:
        """Test run command with missing Azure config."""
        with patch("src.cli.evaluate.LLMConfig.from_env") as mock_config:
            mock_config.side_effect = ValueError("Missing AZURE_OPENAI_ENDPOINT")

            result = runner.invoke(app, ["run"])
            assert result.exit_code == 1
            assert "Configuration error" in result.stdout or "Error" in result.stdout

    def test_run_with_strategy(self) -> None:
        """Test run command with strategy option."""
        # Create mock client and result
        mock_client = MagicMock()
        mock_result = EvaluationRunResult(
            metadata=EvaluationRunMetadata(
                run_id="test123",
                strategy=PromptStrategy.FEW_SHOT,
                test_case_count=0,
            ),
            results=[],
        )

        with (
            patch("src.cli.evaluate.LLMConfig.from_env") as mock_config,
            patch("src.cli.evaluate.LLMClient") as mock_client_class,
            patch("src.cli.evaluate.run_evaluation") as mock_run,
            patch("src.cli.evaluate.get_default_storage") as mock_storage,
        ):
            mock_config.return_value = MagicMock()
            mock_client_class.return_value = mock_client
            mock_run.return_value = mock_result
            mock_storage_instance = MagicMock()
            mock_storage_instance.save_run.return_value = "test123"
            mock_storage.return_value = mock_storage_instance

            result = runner.invoke(app, ["run", "--strategy", "few_shot"])
            assert result.exit_code == 0

            # Verify run_evaluation was called with the right strategy
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["strategy"] == PromptStrategy.FEW_SHOT

    def test_run_with_no_save(self) -> None:
        """Test run command with --no-save flag."""
        mock_client = MagicMock()
        mock_result = EvaluationRunResult(
            metadata=EvaluationRunMetadata(
                run_id="test123",
                strategy=PromptStrategy.ZERO_SHOT,
                test_case_count=0,
            ),
            results=[],
        )

        with (
            patch("src.cli.evaluate.LLMConfig.from_env") as mock_config,
            patch("src.cli.evaluate.LLMClient") as mock_client_class,
            patch("src.cli.evaluate.run_evaluation") as mock_run,
            patch("src.cli.evaluate.get_default_storage") as mock_storage,
        ):
            mock_config.return_value = MagicMock()
            mock_client_class.return_value = mock_client
            mock_run.return_value = mock_result
            mock_storage_instance = MagicMock()
            mock_storage.return_value = mock_storage_instance

            result = runner.invoke(app, ["run", "--no-save"])
            assert result.exit_code == 0

            # Verify save was not called
            mock_storage_instance.save_run.assert_not_called()

    def test_run_with_filters(self) -> None:
        """Test run command with filter options."""
        mock_client = MagicMock()
        mock_result = EvaluationRunResult(
            metadata=EvaluationRunMetadata(
                run_id="test123",
                strategy=PromptStrategy.ZERO_SHOT,
                test_case_count=0,
            ),
            results=[],
        )

        with (
            patch("src.cli.evaluate.LLMConfig.from_env") as mock_config,
            patch("src.cli.evaluate.LLMClient") as mock_client_class,
            patch("src.cli.evaluate.run_evaluation") as mock_run,
            patch("src.cli.evaluate.get_default_storage") as mock_storage,
        ):
            mock_config.return_value = MagicMock()
            mock_client_class.return_value = mock_client
            mock_run.return_value = mock_result
            mock_storage_instance = MagicMock()
            mock_storage_instance.save_run.return_value = "test123"
            mock_storage.return_value = mock_storage_instance

            result = runner.invoke(
                app,
                [
                    "run",
                    "--channel", "form",
                    "--device-type", "implantable",
                    "--difficulty", "easy",
                ],
            )
            assert result.exit_code == 0

            # Verify filters were passed
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            filters = call_kwargs["filters"]
            assert filters.channel == IntakeChannel.FORM
            assert filters.device_type == DeviceType.IMPLANTABLE
            assert filters.difficulty == "easy"
