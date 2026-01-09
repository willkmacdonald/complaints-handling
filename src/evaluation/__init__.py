"""Evaluation framework for complaint handling system."""

from src.evaluation.ablation import (
    AblationReport,
    StrategyComparison,
    StrategyMetrics,
    compare_strategies,
    format_ablation_summary,
    list_ablation_reports,
    load_ablation_report,
    run_ablation_test,
    save_ablation_report,
)
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
    # Ablation
    "AblationReport",
    "StrategyComparison",
    "StrategyMetrics",
    "compare_strategies",
    "format_ablation_summary",
    "list_ablation_reports",
    "load_ablation_report",
    "run_ablation_test",
    "save_ablation_report",
    # Calibration
    "CalibrationAnalysis",
    "CalibrationBin",
    "CalibrationConfig",
    "SuggestionOutcome",
    "analyze_calibration",
    "calculate_calibration_error",
    "create_calibration_config",
    "extract_suggestion_outcomes",
    "find_optimal_threshold",
    "load_calibration_config",
    "save_calibration_config",
    # Metrics
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
