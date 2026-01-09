"""Storage interface and implementations for evaluation runs."""

import json
import logging
from pathlib import Path
from typing import Protocol

from src.evaluation.models import EvaluationRunResult

logger = logging.getLogger(__name__)

# Default storage directory
DEFAULT_EVALUATION_DIR = Path("data/evaluation_runs")


class EvaluationStorage(Protocol):
    """Protocol for storing and retrieving evaluation runs.

    This abstraction allows swapping between file-based storage (current)
    and cloud storage (Azure Blob/Cosmos) in the future.
    """

    def save_run(self, result: EvaluationRunResult) -> str:
        """Save an evaluation run result.

        Args:
            result: Evaluation run result to save.

        Returns:
            The run_id of the saved result.
        """
        ...

    def load_run(self, run_id: str) -> EvaluationRunResult | None:
        """Load an evaluation run by ID.

        Args:
            run_id: The run identifier.

        Returns:
            EvaluationRunResult if found, None otherwise.
        """
        ...

    def list_runs(self, limit: int = 20) -> list[dict[str, str]]:
        """List recent evaluation runs.

        Args:
            limit: Maximum number of runs to return.

        Returns:
            List of dicts with run_id, timestamp, strategy, f1_score.
        """
        ...

    def delete_run(self, run_id: str) -> bool:
        """Delete an evaluation run.

        Args:
            run_id: The run identifier.

        Returns:
            True if deleted, False if not found.
        """
        ...


class FileEvaluationStorage:
    """File-based storage for evaluation runs.

    Stores each run as a JSON file in the configured directory.
    Files are named by run_id for easy lookup.
    """

    def __init__(self, base_path: Path | None = None):
        """Initialize file storage.

        Args:
            base_path: Directory for storing evaluation files.
                Defaults to data/evaluation_runs.
        """
        self.base_path = base_path or DEFAULT_EVALUATION_DIR
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_run_path(self, run_id: str) -> Path:
        """Get the file path for a run."""
        return self.base_path / f"{run_id}.json"

    def save_run(self, result: EvaluationRunResult) -> str:
        """Save an evaluation run result to JSON file.

        Args:
            result: Evaluation run result to save.

        Returns:
            The run_id of the saved result.
        """
        run_id = result.metadata.run_id
        file_path = self._get_run_path(run_id)

        try:
            with open(file_path, "w") as f:
                json.dump(result.model_dump(mode="json"), f, indent=2, default=str)
            logger.info("Saved evaluation run %s to %s", run_id, file_path)
            return run_id
        except OSError as e:
            logger.error("Failed to save evaluation run %s: %s", run_id, e)
            raise

    def load_run(self, run_id: str) -> EvaluationRunResult | None:
        """Load an evaluation run from JSON file.

        Args:
            run_id: The run identifier.

        Returns:
            EvaluationRunResult if found, None otherwise.
        """
        file_path = self._get_run_path(run_id)

        if not file_path.exists():
            logger.warning("Evaluation run %s not found at %s", run_id, file_path)
            return None

        try:
            with open(file_path) as f:
                data = json.load(f)
            return EvaluationRunResult.model_validate(data)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load evaluation run %s: %s", run_id, e)
            return None

    def list_runs(self, limit: int = 20) -> list[dict[str, str]]:
        """List recent evaluation runs.

        Args:
            limit: Maximum number of runs to return.

        Returns:
            List of dicts with run_id, timestamp, strategy, f1_score.
        """
        runs: list[dict[str, str]] = []

        if not self.base_path.exists():
            return runs

        # Get all JSON files sorted by modification time (newest first)
        json_files = sorted(
            self.base_path.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for file_path in json_files[:limit]:
            try:
                with open(file_path) as f:
                    data = json.load(f)

                metadata = data.get("metadata", {})
                runs.append(
                    {
                        "run_id": metadata.get("run_id", file_path.stem),
                        "timestamp": metadata.get("timestamp", ""),
                        "strategy": metadata.get("strategy", "unknown"),
                        "test_case_count": str(metadata.get("test_case_count", 0)),
                        "f1_score": f"{data.get('overall_f1', 0):.1%}"
                        if data.get("overall_f1")
                        else "N/A",
                    }
                )
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Failed to read run file %s: %s", file_path, e)
                continue

        return runs

    def delete_run(self, run_id: str) -> bool:
        """Delete an evaluation run.

        Args:
            run_id: The run identifier.

        Returns:
            True if deleted, False if not found.
        """
        file_path = self._get_run_path(run_id)

        if not file_path.exists():
            return False

        try:
            file_path.unlink()
            logger.info("Deleted evaluation run %s", run_id)
            return True
        except OSError as e:
            logger.error("Failed to delete evaluation run %s: %s", run_id, e)
            return False


def get_default_storage() -> FileEvaluationStorage:
    """Get the default file-based storage instance.

    Returns:
        FileEvaluationStorage configured with default path.
    """
    return FileEvaluationStorage()
