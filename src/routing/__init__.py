"""Routing module for complaint handling.

This module provides services for routing complaints, including:
- MDR (Medical Device Report) determination
- Priority/urgency flagging
- Queue assignment
"""

from src.routing.mdr import determine_mdr, determine_mdr_rules_only

__all__ = ["determine_mdr", "determine_mdr_rules_only"]
