"""Form processing pipeline orchestrator.

This module provides the end-to-end pipeline for processing online form
submissions through the complete complaint handling workflow:

1. Parse form submission
2. Convert to ComplaintRecord
3. Suggest IMDRF codes
4. Determine MDR status
5. Log all events to audit
"""

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.audit.logger import AuditLogger, generate_event_id
from src.audit.models import (
    CodingSuggestedEvent,
    ComplaintCreatedEvent,
    MDRDeterminedEvent,
)
from src.coding.service import CodingResult, CodingService
from src.intake.forms import (
    form_to_complaint,
    parse_form_submission,
    validate_form_completeness,
)
from src.llm.client import LLMClient, LLMError, create_client
from src.models.coding import CodingDecision
from src.models.complaint import ComplaintRecord
from src.models.enums import ComplaintStatus
from src.pipeline.models import ProcessingResult, ProcessingStatus
from src.routing.mdr import determine_mdr

logger = logging.getLogger(__name__)

# Default directories for pipeline outputs
DEFAULT_COMPLAINTS_DIR = Path("data/complaints")
DEFAULT_DECISIONS_DIR = Path("data/decisions")
DEFAULT_AUDIT_DIR = Path("audit_logs")


def _generate_processing_id() -> str:
    """Generate a unique processing run ID.

    Format: PROC-{timestamp}-{uuid4_short}
    Example: PROC-20240115143052-a1b2c3d4
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"PROC-{timestamp}-{short_uuid}"


def _save_complaint(complaint: ComplaintRecord, output_dir: Path) -> None:
    """Save complaint record to disk.

    Args:
        complaint: The complaint record to save.
        output_dir: Directory for output files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    complaint_file = output_dir / f"{complaint.complaint_id}.json"
    with open(complaint_file, "w") as f:
        json.dump(complaint.model_dump(mode="json"), f, indent=2, default=str)
    logger.info("Saved complaint to %s", complaint_file)


def _save_decision(decision: CodingDecision, output_dir: Path) -> None:
    """Save coding decision to disk.

    Args:
        decision: The coding decision to save.
        output_dir: Directory for output files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_file = output_dir / f"{decision.complaint_id}_decision.json"
    with open(decision_file, "w") as f:
        json.dump(decision.model_dump(mode="json"), f, indent=2, default=str)
    logger.info("Saved coding decision to %s", decision_file)


def process_form(
    raw_data: dict[str, Any],
    client: LLMClient | None = None,
    audit_logger: AuditLogger | None = None,
    complaints_dir: Path | None = None,
    decisions_dir: Path | None = None,
    save_outputs: bool = True,
    use_llm_for_mdr: bool = True,
) -> ProcessingResult:
    """Process a form submission through the complete pipeline.

    This function orchestrates the end-to-end workflow:
    1. Parse form submission and validate
    2. Convert to ComplaintRecord
    3. Suggest IMDRF codes using LLM
    4. Determine MDR requirement
    5. Log all events to audit trail

    Args:
        raw_data: Raw form submission data (dictionary).
        client: Optional LLM client. If not provided, creates one from environment.
        audit_logger: Optional audit logger. If not provided, creates default.
        complaints_dir: Directory for complaint output files.
        decisions_dir: Directory for decision output files.
        save_outputs: Whether to save complaint and decision files to disk.
        use_llm_for_mdr: Whether to use LLM for MDR determination.

    Returns:
        ProcessingResult with all outputs and metadata.

    Raises:
        PipelineError: If a critical error prevents processing.
    """
    # Initialize result
    processing_id = _generate_processing_id()
    result = ProcessingResult(
        status=ProcessingStatus.FAILED,
        processing_id=processing_id,
    )

    # Set default directories
    complaints_dir = complaints_dir or DEFAULT_COMPLAINTS_DIR
    decisions_dir = decisions_dir or DEFAULT_DECISIONS_DIR

    # Initialize audit logger
    if audit_logger is None:
        audit_logger = AuditLogger(log_dir=DEFAULT_AUDIT_DIR)

    # Step 1: Parse form submission
    logger.info("Processing form submission (processing_id=%s)", processing_id)

    try:
        form = parse_form_submission(raw_data)
        validation = validate_form_completeness(form)
        result.validation_result = validation
        result.form_valid = validation.is_complete

        if not validation.is_complete:
            result.add_error(
                step="parse_form",
                message="Form validation failed: missing required fields",
                details={"missing_required": validation.missing_required},
            )
            logger.warning(
                "Form validation failed: missing %s", validation.missing_required
            )
            # Continue processing even with incomplete form for partial results
    except ValueError as e:
        result.add_error(
            step="parse_form",
            message=f"Failed to parse form submission: {e}",
        )
        logger.error("Failed to parse form submission: %s", e)
        result.completed_at = datetime.now(UTC)
        return result

    # Step 2: Convert to ComplaintRecord
    try:
        complaint = form_to_complaint(form)
        complaint.status = ComplaintStatus.EXTRACTED
        result.complaint = complaint
        result.form_valid = True  # Conversion succeeded even if some fields missing
        logger.info("Created complaint record: %s", complaint.complaint_id)

        # Log complaint creation to audit
        event_id = audit_logger.log_event(
            ComplaintCreatedEvent(
                event_id=generate_event_id(),
                resource_id=complaint.complaint_id,
                intake_channel=complaint.intake_channel.value,
                device_name=complaint.device_info.device_name,
                manufacturer=complaint.device_info.manufacturer,
                initial_status=complaint.status.value,
            )
        )
        result.audit_event_ids.append(event_id)

    except ValueError as e:
        result.add_error(
            step="convert_complaint",
            message=f"Failed to convert form to complaint: {e}",
        )
        logger.error("Failed to convert form to complaint: %s", e)
        result.completed_at = datetime.now(UTC)
        return result

    # Step 3: Suggest IMDRF codes
    coding_result: CodingResult | None = None
    try:
        if client is None:
            client = create_client()

        coding_service = CodingService(client=client)
        coding_result = coding_service.suggest_codes(complaint)
        result.coding_result = coding_result

        if coding_result.is_success:
            # Update complaint status
            complaint.status = ComplaintStatus.CODED

            # Log coding suggestions to audit
            suggested_codes_data = [
                {
                    "code_id": s.code_id,
                    "code_name": s.code_name,
                    "confidence": s.confidence,
                    "reasoning": s.reasoning,
                }
                for s in coding_result.suggestions
            ]
            event_id = audit_logger.log_event(
                CodingSuggestedEvent(
                    event_id=generate_event_id(),
                    resource_id=complaint.complaint_id,
                    suggested_codes=suggested_codes_data,
                    model_name=coding_result.model_used or "unknown",
                    total_tokens=coding_result.tokens_used,
                    latency_ms=coding_result.latency_ms,
                )
            )
            result.audit_event_ids.append(event_id)
            logger.info(
                "Generated %d IMDRF code suggestions for %s",
                len(coding_result.suggestions),
                complaint.complaint_id,
            )
        else:
            result.add_error(
                step="suggest_codes",
                message=f"IMDRF coding failed: {coding_result.error}",
            )
            logger.warning(
                "IMDRF coding failed for %s: %s",
                complaint.complaint_id,
                coding_result.error,
            )

    except LLMError as e:
        result.add_error(
            step="suggest_codes",
            message=f"LLM error during coding: {e}",
        )
        logger.error("LLM error during coding: %s", e)
        # Continue to MDR determination without LLM coding

    except ValueError as e:
        result.add_error(
            step="suggest_codes",
            message=f"Configuration error: {e}",
        )
        logger.error("Configuration error for LLM client: %s", e)

    # Step 4: Determine MDR requirement
    try:
        mdr_result = determine_mdr(
            complaint,
            client=client,
            use_llm=use_llm_for_mdr,
        )
        result.mdr_determination = mdr_result

        # Log MDR determination to audit
        event_id = audit_logger.log_event(
            MDRDeterminedEvent(
                event_id=generate_event_id(),
                resource_id=complaint.complaint_id,
                requires_mdr=mdr_result.requires_mdr,
                mdr_criteria_met=[c.value for c in mdr_result.mdr_criteria_met],
                confidence=mdr_result.ai_confidence,
                reasoning=mdr_result.ai_reasoning,
                review_priority=mdr_result.review_priority,
            )
        )
        result.audit_event_ids.append(event_id)
        logger.info(
            "MDR determination for %s: requires_mdr=%s, priority=%s",
            complaint.complaint_id,
            mdr_result.requires_mdr,
            mdr_result.review_priority,
        )

    except Exception as e:
        result.add_error(
            step="determine_mdr",
            message=f"MDR determination failed: {e}",
        )
        logger.error("MDR determination failed: %s", e)

    # Step 5: Save outputs
    if save_outputs and complaint:
        try:
            _save_complaint(complaint, complaints_dir)

            # Create and save coding decision
            if coding_result:
                decision = CodingDecision(
                    complaint_id=complaint.complaint_id,
                    suggested_codes=coding_result.suggestions,
                    suggestion_timestamp=coding_result.timestamp,
                )
                _save_decision(decision, decisions_dir)

        except OSError as e:
            result.add_error(
                step="save_outputs",
                message=f"Failed to save outputs: {e}",
            )
            logger.error("Failed to save outputs: %s", e)

    # Determine final status
    result.completed_at = datetime.now(UTC)

    if result.is_complete:
        result.status = ProcessingStatus.SUCCESS
    elif result.complaint is not None:
        result.status = ProcessingStatus.PARTIAL
    else:
        result.status = ProcessingStatus.FAILED

    logger.info(
        "Pipeline completed: processing_id=%s, status=%s, complaint_id=%s",
        processing_id,
        result.status.value,
        result.complaint_id,
    )

    return result


def process_form_file(
    file_path: Path | str,
    client: LLMClient | None = None,
    audit_logger: AuditLogger | None = None,
    complaints_dir: Path | None = None,
    decisions_dir: Path | None = None,
    save_outputs: bool = True,
) -> ProcessingResult:
    """Process a form submission from a JSON file.

    Convenience function that loads form data from a file and processes it.

    Args:
        file_path: Path to JSON file containing form submission.
        client: Optional LLM client.
        audit_logger: Optional audit logger.
        complaints_dir: Directory for complaint output files.
        decisions_dir: Directory for decision output files.
        save_outputs: Whether to save outputs to disk.

    Returns:
        ProcessingResult with all outputs and metadata.

    Raises:
        FileNotFoundError: If file does not exist.
        json.JSONDecodeError: If file is not valid JSON.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Form file not found: {file_path}")

    with open(file_path) as f:
        data = json.load(f)

    # Handle test case format vs direct form data
    raw_data = data.get("raw_input", data)

    return process_form(
        raw_data=raw_data,
        client=client,
        audit_logger=audit_logger,
        complaints_dir=complaints_dir,
        decisions_dir=decisions_dir,
        save_outputs=save_outputs,
    )
