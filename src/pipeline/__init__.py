"""End-to-end form processing pipeline.

This module provides the orchestration layer that combines form ingestion,
IMDRF code suggestion, MDR determination, and audit logging into a single
processing workflow.
"""

from src.pipeline.forms import process_form
from src.pipeline.models import PipelineError, ProcessingResult, ProcessingStatus

__all__ = [
    "process_form",
    "ProcessingResult",
    "ProcessingStatus",
    "PipelineError",
]
