"""Audit logging module for regulatory compliance.

This module provides append-only audit logging for all system actions,
required for FDA 21 CFR Part 11 compliance.

All events are stored with UTC timestamps and are immutable once created.
"""

from src.audit.logger import AuditLogger, get_audit_logger, log_event
from src.audit.models import (
    AuditAction,
    AuditEvent,
    CodingReviewedEvent,
    CodingSuggestedEvent,
    ComplaintCreatedEvent,
    MDRDeterminedEvent,
)

__all__ = [
    "AuditAction",
    "AuditEvent",
    "AuditLogger",
    "CodingReviewedEvent",
    "CodingSuggestedEvent",
    "ComplaintCreatedEvent",
    "MDRDeterminedEvent",
    "get_audit_logger",
    "log_event",
]
