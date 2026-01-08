"""MDR (Medical Device Report) determination service.

This module implements the logic to determine whether a complaint requires
FDA Medical Device Reporting per 21 CFR Part 803.

Key principles:
- 100% sensitivity is required (no false negatives)
- Conservative defaults (flag uncertain cases for review)
- Human-in-the-loop (AI determination requires human confirmation)
"""

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from src.llm.client import LLMClient, LLMError, create_client
from src.llm.parsing import ParseError, parse_json_response
from src.llm.prompts import MDR_DETERMINATION_TEMPLATE, create_messages, render_prompt
from src.models.complaint import ComplaintRecord
from src.models.mdr import MDRCriteria, MDRDetermination

logger = logging.getLogger(__name__)


# Keywords that strongly suggest death
DEATH_KEYWORDS = [
    r"\bdeath\b",
    r"\bdied\b",
    r"\bdeceased\b",
    r"\bfatal\b",
    r"\bfatality\b",
    r"\bpassed\s+away\b",
    r"\bloss\s+of\s+life\b",
    r"\bexpired\b",  # Medical term for death
    r"\bDOA\b",  # Dead on arrival
]

# Keywords suggesting serious injury
SERIOUS_INJURY_KEYWORDS = [
    # Life-threatening
    r"\blife[- ]threatening\b",
    r"\bcritical\s+condition\b",
    r"\bICU\b",
    r"\bintensive\s+care\b",
    r"\bresuscitat",  # resuscitation, resuscitated
    r"\bcardiac\s+arrest\b",
    r"\brespiratory\s+failure\b",
    r"\bsepsis\b",
    r"\bseptic\s+shock\b",
    # Hospitalization
    r"\bhospitali[sz]",  # hospitalized, hospitalization
    r"\badmitted\s+to\s+(?:the\s+)?hospital\b",
    r"\bemergency\s+(?:room|department|surgery)\b",
    r"\bER\s+visit\b",
    # Surgical intervention
    r"\bsurgery\b",
    r"\bsurgical\s+intervention\b",
    r"\boperation\b",
    r"\bprocedure\b",
    r"\bexplant",  # explantation, explanted
    r"\brevision\s+surgery\b",
    # Permanent damage
    r"\bpermanent\b",
    r"\birreversible\b",
    r"\bdisability\b",
    r"\bdisabled\b",
    r"\bamputation\b",
    r"\bloss\s+of\s+(?:function|vision|hearing|limb)\b",
    r"\bparalysis\b",
    r"\bparalyzed\b",
    r"\bblind(?:ness)?\b",
    r"\bdeaf(?:ness)?\b",
    # Severe injuries
    r"\bserious\s+(?:injury|harm|damage)\b",
    r"\bsevere\s+(?:injury|harm|damage|bleeding|infection)\b",
    r"\bsignificant\s+(?:injury|harm)\b",
    r"\btrauma\b",
    r"\bfracture\b",
    r"\bhemorrhag",  # hemorrhage, hemorrhaging
    r"\binfection\b",
]

# Keywords suggesting device malfunction
MALFUNCTION_KEYWORDS = [
    r"\bmalfunction",
    r"\bfail(?:ed|ure|ing)?\b",
    r"\bdefect(?:ive)?\b",
    r"\bbroke(?:n)?\b",
    r"\bnot\s+work(?:ing)?\b",
    r"\bstopped\s+working\b",
    r"\bdisconnect",
    r"\bleak(?:ing|ed|age)?\b",
    r"\bcrac(?:k|ked)\b",
    r"\bfractur(?:e|ed)\b",
    r"\berror\b",
    r"\bshort[- ]circuit\b",
    r"\boverheat",
    r"\bbatter(?:y|ies)\s+(?:depleted|drained|dead|failed)\b",
    # Diagnostic device inaccuracies (critical for blood glucose meters, etc.)
    r"\bfalsely?\s+(?:low|high|positive|negative)\b",
    r"\bwrong\s+(?:reading|value|display|result)\b",
    r"\breading\s+(?:approximately|about)?\s*\d+\s*(?:mg|mmol)?/?(?:dl)?\s+(?:low|high|off)\b",
    r"\bwas\s+reading\s+(?:approximately|about)?\s*\d+",  # "meter was reading X mg/dL low"
    r"\b(?:displayed|showed|read)\s+(?:a\s+)?(?:reading|value)\s+of\s+\d+",  # "displayed a reading of 52"
    r"\bunexpected(?:ly)?\s+(?:shut|turn|power)\b",
    r"\balarm\s+(?:failure|malfunction|did\s+not)\b",
    r"\binaccurate\b",
    r"\bincorrect\s+(?:reading|display|value|dose)\b",
    r"\bfalse\s+(?:reading|positive|negative)\b",
    r"\bcalibration\b",
    r"\bdrift(?:ed)?\b",
    r"\bcontaminat",
    r"\bsterility\b",
]

# Keywords suggesting user error (may reduce MDR likelihood)
USER_ERROR_KEYWORDS = [
    r"\buser\s+error\b",
    r"\bpatient\s+error\b",
    r"\boperator\s+error\b",
    r"\bincorrect(?:ly)?\s+(?:used|programmed|configured|set)\b",
    r"\baccidental(?:ly)?\s+(?:changed|modified|set|programmed)\b",
    r"\bmisuse\b",
    r"\bfunctioned\s+(?:correctly|properly|as\s+(?:designed|intended|programmed))\b",
    r"\bworked\s+(?:correctly|properly|as\s+(?:designed|intended|programmed))\b",
    r"\bno\s+device\s+(?:malfunction|defect|problem)\b",
    r"\btraining\s+(?:issue|needed|provided)\b",
]


class RuleBasedMDRResult(BaseModel):
    """Result from rules-based MDR analysis."""

    death_detected: bool = Field(default=False)
    death_evidence: list[str] = Field(default_factory=list)
    serious_injury_detected: bool = Field(default=False)
    serious_injury_evidence: list[str] = Field(default_factory=list)
    malfunction_detected: bool = Field(default=False)
    malfunction_evidence: list[str] = Field(default_factory=list)
    user_error_detected: bool = Field(default=False)
    user_error_evidence: list[str] = Field(default_factory=list)


def _search_patterns(text: str, patterns: list[str]) -> list[str]:
    """Search for regex patterns in text and return matching snippets.

    Args:
        text: Text to search.
        patterns: List of regex patterns.

    Returns:
        List of matching text snippets with context.
    """
    evidence: list[str] = []
    text_lower = text.lower()

    for pattern in patterns:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            snippet = text[start:end].strip()
            if snippet not in evidence:
                evidence.append(f"...{snippet}...")

    return evidence


def _analyze_text_for_mdr(
    narrative: str,
    patient_outcome: str | None,
    device_outcome: str | None,
) -> RuleBasedMDRResult:
    """Analyze complaint text using rules-based pattern matching.

    Args:
        narrative: Event description narrative.
        patient_outcome: Patient outcome text.
        device_outcome: Device outcome text.

    Returns:
        RuleBasedMDRResult with detected criteria and evidence.
    """
    # Combine all text for analysis
    combined_text = " ".join(
        filter(None, [narrative, patient_outcome, device_outcome])
    )

    result = RuleBasedMDRResult()

    # Check for death
    result.death_evidence = _search_patterns(combined_text, DEATH_KEYWORDS)
    result.death_detected = len(result.death_evidence) > 0

    # Check for serious injury
    result.serious_injury_evidence = _search_patterns(
        combined_text, SERIOUS_INJURY_KEYWORDS
    )
    result.serious_injury_detected = len(result.serious_injury_evidence) > 0

    # Check for malfunction
    result.malfunction_evidence = _search_patterns(combined_text, MALFUNCTION_KEYWORDS)
    result.malfunction_detected = len(result.malfunction_evidence) > 0

    # Check for user error indicators
    result.user_error_evidence = _search_patterns(combined_text, USER_ERROR_KEYWORDS)
    result.user_error_detected = len(result.user_error_evidence) > 0

    return result


def _determine_priority(
    requires_mdr: bool,
    death_detected: bool,
    serious_injury_detected: bool,
    confidence: float,
) -> str:
    """Determine review priority based on findings.

    Args:
        requires_mdr: Whether MDR appears to be required.
        death_detected: Whether death was mentioned.
        serious_injury_detected: Whether serious injury was mentioned.
        confidence: AI confidence in determination.

    Returns:
        Priority level: "urgent", "high", "normal", or "low".
    """
    if death_detected:
        return "urgent"
    if requires_mdr and serious_injury_detected:
        return "high"
    if requires_mdr:
        return "high"
    if confidence < 0.7:
        return "normal"  # Low confidence needs review
    return "low"


def determine_mdr_rules_only(complaint: ComplaintRecord) -> MDRDetermination:
    """Determine MDR requirement using rules-based analysis only.

    This is a fallback when LLM is unavailable or for faster processing.
    Uses conservative defaults - will flag uncertain cases for review.

    Args:
        complaint: The complaint record to analyze.

    Returns:
        MDRDetermination with rules-based analysis results.
    """
    # Extract text from complaint
    narrative = complaint.narrative
    patient_outcome = complaint.event_info.patient_outcome
    device_outcome = complaint.event_info.device_outcome

    # Run rules-based analysis
    analysis = _analyze_text_for_mdr(narrative, patient_outcome, device_outcome)

    # Determine criteria met
    criteria_met: list[MDRCriteria] = []
    key_factors: list[str] = []

    if analysis.death_detected:
        criteria_met.append(MDRCriteria.DEATH)
        key_factors.append("Death mentioned in complaint")

    if analysis.serious_injury_detected:
        criteria_met.append(MDRCriteria.SERIOUS_INJURY)
        key_factors.append("Serious injury indicators found")

    # Malfunction analysis - need to consider if it could cause harm if it recurs
    if analysis.malfunction_detected:
        if not analysis.user_error_detected:
            # Malfunction without clear user error - conservative approach
            if analysis.serious_injury_detected or analysis.death_detected:
                criteria_met.append(MDRCriteria.MALFUNCTION_COULD_CAUSE_DEATH)
                key_factors.append(
                    "Device malfunction associated with death/serious injury"
                )
            else:
                # Malfunction could potentially cause harm if it recurs
                criteria_met.append(MDRCriteria.MALFUNCTION_COULD_CAUSE_SERIOUS_INJURY)
                key_factors.append("Device malfunction could cause harm if it recurs")
        else:
            key_factors.append(
                "User error detected - device may have functioned correctly"
            )

    # Determine if MDR is required
    # Conservative: require MDR if any criteria met, UNLESS clear user error
    requires_mdr = len(criteria_met) > 0

    # If user error is clearly indicated AND no death/serious injury, be less aggressive
    if (
        analysis.user_error_detected
        and not analysis.death_detected
        and not analysis.serious_injury_detected
    ):
        # Still flag for review but lower confidence
        confidence = 0.4
        key_factors.append("User error indicators present - needs careful review")
    elif requires_mdr:
        confidence = 0.75  # Moderate confidence from rules
    else:
        confidence = 0.6  # Lower confidence when not flagging

    # Build reasoning
    reasoning_parts = []
    if analysis.death_detected:
        reasoning_parts.append(
            f"Death mentioned: {'; '.join(analysis.death_evidence[:2])}"
        )
    if analysis.serious_injury_detected:
        reasoning_parts.append(
            f"Serious injury indicators: {'; '.join(analysis.serious_injury_evidence[:2])}"
        )
    if analysis.malfunction_detected:
        reasoning_parts.append(
            f"Device malfunction: {'; '.join(analysis.malfunction_evidence[:2])}"
        )
    if analysis.user_error_detected:
        reasoning_parts.append(
            f"User error indicators: {'; '.join(analysis.user_error_evidence[:2])}"
        )

    reasoning = (
        "Rules-based analysis: " + " | ".join(reasoning_parts)
        if reasoning_parts
        else "No clear MDR criteria detected by rules-based analysis."
    )

    # Determine priority
    priority = _determine_priority(
        requires_mdr,
        analysis.death_detected,
        analysis.serious_injury_detected,
        confidence,
    )

    return MDRDetermination(
        complaint_id=complaint.complaint_id,
        requires_mdr=requires_mdr,
        mdr_criteria_met=criteria_met,
        ai_confidence=confidence,
        ai_reasoning=reasoning,
        key_factors=key_factors,
        needs_human_review=True,  # Always require human review
        review_priority=priority,
    )


def _call_llm_for_mdr(
    complaint: ComplaintRecord,
    client: LLMClient,
) -> dict[str, Any]:
    """Call LLM for MDR determination assistance.

    Args:
        complaint: The complaint record to analyze.
        client: LLM client to use.

    Returns:
        Parsed JSON response from LLM.

    Raises:
        LLMError: If LLM call fails.
        ParseError: If response cannot be parsed.
    """
    # Prepare variables for prompt
    variables = {
        "device_name": complaint.device_info.device_name,
        "manufacturer": complaint.device_info.manufacturer,
        "event_description": complaint.narrative,
        "patient_outcome": complaint.event_info.patient_outcome or "Not specified",
        "device_status": complaint.event_info.device_outcome or "Unknown",
    }

    # Render prompt
    system_prompt, user_prompt = render_prompt(
        MDR_DETERMINATION_TEMPLATE,
        variables,
    )

    # Create messages
    messages = create_messages(system_prompt, user_prompt)

    # Call LLM
    response = client.complete(
        messages=messages,
        temperature=0.0,  # Deterministic for consistency
        response_format={"type": "json_object"},
    )

    # Parse response
    return parse_json_response(response.content)


def determine_mdr(
    complaint: ComplaintRecord,
    client: LLMClient | None = None,
    use_llm: bool = True,
) -> MDRDetermination:
    """Determine if a complaint requires MDR filing.

    Uses a combination of rules-based analysis and LLM-assisted evaluation
    for comprehensive MDR determination.

    The approach is conservative:
    - 100% sensitivity target (no false negatives)
    - Uncertain cases are flagged for review
    - Human review is always required for final determination

    Args:
        complaint: The complaint record to analyze.
        client: Optional LLM client. If not provided and use_llm is True,
            a default client will be created.
        use_llm: Whether to use LLM for enhanced analysis. If False,
            only rules-based analysis is used.

    Returns:
        MDRDetermination with comprehensive analysis results.
    """
    # First, run rules-based analysis as baseline
    rules_result = determine_mdr_rules_only(complaint)

    # If not using LLM, return rules-based result
    if not use_llm:
        return rules_result

    # Try LLM-assisted analysis
    try:
        if client is None:
            client = create_client()

        llm_response = _call_llm_for_mdr(complaint, client)

        # Extract LLM determination
        llm_requires_mdr = llm_response.get("requires_mdr", True)  # Conservative default
        llm_confidence = llm_response.get("confidence", 0.5)
        llm_criteria = llm_response.get("criteria_met", [])
        llm_reasoning = llm_response.get("reasoning", "")
        llm_evidence = llm_response.get("evidence", [])

        # Map LLM criteria to our enum
        criteria_met: list[MDRCriteria] = list(rules_result.mdr_criteria_met)
        for criterion in llm_criteria:
            criterion_lower = criterion.lower().replace(" ", "_")
            if "death" in criterion_lower:
                if MDRCriteria.DEATH not in criteria_met:
                    criteria_met.append(MDRCriteria.DEATH)
            elif "serious" in criterion_lower and "injury" in criterion_lower:
                if MDRCriteria.SERIOUS_INJURY not in criteria_met:
                    criteria_met.append(MDRCriteria.SERIOUS_INJURY)
            elif "malfunction" in criterion_lower:
                if (
                    "death" in criterion_lower
                    and MDRCriteria.MALFUNCTION_COULD_CAUSE_DEATH not in criteria_met
                ):
                    criteria_met.append(MDRCriteria.MALFUNCTION_COULD_CAUSE_DEATH)
                elif (
                    MDRCriteria.MALFUNCTION_COULD_CAUSE_SERIOUS_INJURY not in criteria_met
                ):
                    criteria_met.append(MDRCriteria.MALFUNCTION_COULD_CAUSE_SERIOUS_INJURY)

        # Combine evidence
        key_factors = list(rules_result.key_factors)
        for evidence in llm_evidence:
            if evidence not in key_factors:
                key_factors.append(evidence)

        # Conservative combination of results
        # If EITHER rules OR LLM thinks MDR is required, flag it
        requires_mdr = rules_result.requires_mdr or llm_requires_mdr

        # Use higher confidence if both agree, lower if they disagree
        if rules_result.requires_mdr == llm_requires_mdr:
            confidence = max(rules_result.ai_confidence, llm_confidence)
        else:
            # Disagreement - use average but require review
            confidence = (rules_result.ai_confidence + llm_confidence) / 2

        # Combined reasoning
        combined_reasoning = (
            f"Combined analysis - Rules: {rules_result.ai_reasoning} | "
            f"LLM: {llm_reasoning}"
        )

        # Determine priority
        priority = _determine_priority(
            requires_mdr,
            MDRCriteria.DEATH in criteria_met,
            MDRCriteria.SERIOUS_INJURY in criteria_met,
            confidence,
        )

        return MDRDetermination(
            complaint_id=complaint.complaint_id,
            requires_mdr=requires_mdr,
            mdr_criteria_met=criteria_met,
            ai_confidence=confidence,
            ai_reasoning=combined_reasoning,
            key_factors=key_factors,
            needs_human_review=True,
            review_priority=priority,
        )

    except (LLMError, ParseError, ValueError) as e:
        # LLM failed - fall back to rules-based result with note
        logger.warning("LLM-assisted MDR determination failed: %s", e)
        rules_result.ai_reasoning = (
            f"{rules_result.ai_reasoning} (LLM analysis unavailable: {e})"
        )
        return rules_result
