"""Evaluation runner for running coding service against test cases."""

import logging
import time
from collections.abc import Callable

from src.coding.service import CodingResult, CodingService
from src.evaluation.metrics import evaluate_coding
from src.evaluation.models import (
    EvaluationFilters,
    EvaluationRunMetadata,
    EvaluationRunResult,
    PromptStrategy,
    TestCaseEvaluationResult,
    TokenStats,
)
from src.llm import LLMClient
from src.testing.test_case_loader import ComplaintTestCase, load_all_test_cases

logger = logging.getLogger(__name__)


def run_evaluation(
    client: LLMClient,
    strategy: PromptStrategy = PromptStrategy.ZERO_SHOT,
    filters: EvaluationFilters | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> EvaluationRunResult:
    """Run evaluation on test cases with the coding service.

    Args:
        client: LLM client for making requests.
        strategy: Prompt strategy to use.
        filters: Optional filters for test cases.
        progress_callback: Optional callback for progress updates.
            Called with (current, total, test_case_id).

    Returns:
        EvaluationRunResult with all metrics and results.
    """
    filters = filters or EvaluationFilters()

    # Load test cases with filters
    test_cases = load_all_test_cases(
        device_type=filters.device_type,
        severity=filters.severity,
        difficulty=filters.difficulty,
    )

    # Filter by channel if specified
    if filters.channel:
        test_cases = [tc for tc in test_cases if tc.channel == filters.channel]

    if not test_cases:
        logger.warning("No test cases match the specified filters")
        return EvaluationRunResult(
            metadata=EvaluationRunMetadata(
                strategy=strategy,
                filters=filters,
                test_case_count=0,
            ),
            results=[],
        )

    # Initialize metadata
    metadata = EvaluationRunMetadata(
        strategy=strategy,
        filters=filters,
        test_case_count=len(test_cases),
    )

    # Create coding service
    coding_service = CodingService(client=client)

    # Track overall stats
    results: list[TestCaseEvaluationResult] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    start_time = time.perf_counter()

    # Process each test case
    for idx, test_case in enumerate(test_cases):
        if progress_callback:
            progress_callback(idx + 1, len(test_cases), test_case.test_case_id)

        result = _evaluate_test_case(
            test_case=test_case,
            coding_service=coding_service,
        )
        results.append(result)

        # Accumulate token stats
        total_tokens += result.tokens_used
        # Note: We track total tokens; prompt/completion split available from CodingResult

    end_time = time.perf_counter()
    total_duration_ms = (end_time - start_time) * 1000

    # Update metadata
    metadata.total_duration_ms = total_duration_ms
    metadata.model = client.config.deployment_name
    metadata.token_stats = TokenStats(
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
    )

    # Create and populate result
    run_result = EvaluationRunResult(
        metadata=metadata,
        results=results,
    )
    run_result.calculate_aggregates()

    return run_result


def _evaluate_test_case(
    test_case: ComplaintTestCase,
    coding_service: CodingService,
) -> TestCaseEvaluationResult:
    """Evaluate a single test case.

    Args:
        test_case: Test case to evaluate.
        coding_service: Coding service instance.

    Returns:
        TestCaseEvaluationResult with predictions and metrics.
    """
    # Run coding service on the expected complaint
    coding_result: CodingResult = coding_service.suggest_codes(
        test_case.expected_complaint
    )

    if not coding_result.is_success:
        return TestCaseEvaluationResult(
            test_case_id=test_case.test_case_id,
            name=test_case.name,
            channel=test_case.channel,
            device_type=test_case.device_type,
            severity=test_case.severity,
            difficulty=test_case.difficulty,
            expected_codes=test_case.ground_truth.expected_imdrf_codes,
            alternative_codes=test_case.ground_truth.alternative_codes,
            error=coding_result.error,
            tokens_used=coding_result.tokens_used,
            latency_ms=coding_result.latency_ms,
        )

    # Extract predicted codes and confidences
    predicted_codes = [s.code_id for s in coding_result.suggestions]
    predicted_confidences = {s.code_id: s.confidence for s in coding_result.suggestions}

    # Calculate metrics
    coding_metrics = evaluate_coding(
        predicted_codes=predicted_codes,
        expected_codes=test_case.ground_truth.expected_imdrf_codes,
        alternative_codes=test_case.ground_truth.alternative_codes,
    )

    return TestCaseEvaluationResult(
        test_case_id=test_case.test_case_id,
        name=test_case.name,
        channel=test_case.channel,
        device_type=test_case.device_type,
        severity=test_case.severity,
        difficulty=test_case.difficulty,
        predicted_codes=predicted_codes,
        predicted_confidences=predicted_confidences,
        expected_codes=test_case.ground_truth.expected_imdrf_codes,
        alternative_codes=test_case.ground_truth.alternative_codes,
        coding_metrics=coding_metrics,
        tokens_used=coding_result.tokens_used,
        latency_ms=coding_result.latency_ms,
    )


def format_run_summary(result: EvaluationRunResult) -> str:
    """Format a summary of the evaluation run.

    Args:
        result: Evaluation run result.

    Returns:
        Formatted summary string.
    """
    lines = [
        f"Evaluation Run: {result.metadata.run_id}",
        f"Strategy: {result.metadata.strategy.value}",
        f"Test Cases: {result.metadata.test_case_count}",
        f"Duration: {result.metadata.total_duration_ms:.0f}ms",
        f"Total Tokens: {result.metadata.token_stats.total_tokens}",
        "",
    ]

    if result.overall_f1 is not None:
        lines.extend(
            [
                "Overall Metrics:",
                f"  Precision: {result.overall_precision:.1%}"
                if result.overall_precision
                else "",
                f"  Recall: {result.overall_recall:.1%}"
                if result.overall_recall
                else "",
                f"  F1 Score: {result.overall_f1:.1%}",
                f"  Exact Match: {result.exact_match_rate:.1%}"
                if result.exact_match_rate
                else "",
            ]
        )

    # Filter out empty lines from conditional formatting
    lines = [line for line in lines if line or line == ""]

    return "\n".join(lines)
