"""Evaluation framework for complaint handling system."""

from src.evaluation.metrics import (
    CodingMetrics,
    ExtractionMetrics,
    MDRMetrics,
    evaluate_coding,
    evaluate_extraction,
    evaluate_mdr,
)
from src.evaluation.models import (
    EvaluationFilters,
    EvaluationRunMetadata,
    EvaluationRunResult,
    PromptStrategy,
    TestCaseEvaluationResult,
    TokenStats,
)
from src.evaluation.reporter import (
    AggregateMetrics,
    EvaluationReport,
    TestCaseResult,
    export_report_json,
    export_report_markdown,
    format_report_summary,
    generate_report,
)
from src.evaluation.runner import format_run_summary, run_evaluation
from src.evaluation.storage import (
    EvaluationStorage,
    FileEvaluationStorage,
    get_default_storage,
)

__all__ = [
    "AggregateMetrics",
    "CodingMetrics",
    "EvaluationFilters",
    "EvaluationReport",
    "EvaluationRunMetadata",
    "EvaluationRunResult",
    "EvaluationStorage",
    "ExtractionMetrics",
    "FileEvaluationStorage",
    "MDRMetrics",
    "PromptStrategy",
    "TestCaseEvaluationResult",
    "TestCaseResult",
    "TokenStats",
    "evaluate_coding",
    "evaluate_extraction",
    "evaluate_mdr",
    "export_report_json",
    "export_report_markdown",
    "format_report_summary",
    "format_run_summary",
    "generate_report",
    "get_default_storage",
    "run_evaluation",
]
