"""IMDRF code database and utility functions."""

import json
from functools import lru_cache
from pathlib import Path

from src.imdrf.models import IMDRFCode
from src.models.enums import IMDRFCodeType

# Data directory path
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "imdrf"


@lru_cache(maxsize=1)
def _load_all_codes() -> dict[str, IMDRFCode]:
    """Load all IMDRF codes from JSON files into memory.

    Returns a dictionary mapping code_id to IMDRFCode.
    """
    codes: dict[str, IMDRFCode] = {}

    # Load Annex A - Device Problems
    annex_a_path = DATA_DIR / "annex_a_device_problems.json"
    if annex_a_path.exists():
        with open(annex_a_path) as f:
            data = json.load(f)
            for code_data in data["codes"]:
                code = IMDRFCode(
                    code_id=code_data["code_id"],
                    name=code_data["name"],
                    code_type=IMDRFCodeType.DEVICE_PROBLEM,
                    parent_id=code_data.get("parent_id"),
                    description=code_data.get("description"),
                    examples=code_data.get("examples", []),
                    level=code_data.get("level", 1),
                )
                codes[code.code_id] = code

    # Load Annex C - Patient Problems
    annex_c_path = DATA_DIR / "annex_c_patient_problems.json"
    if annex_c_path.exists():
        with open(annex_c_path) as f:
            data = json.load(f)
            for code_data in data["codes"]:
                code = IMDRFCode(
                    code_id=code_data["code_id"],
                    name=code_data["name"],
                    code_type=IMDRFCodeType.PATIENT_PROBLEM,
                    parent_id=code_data.get("parent_id"),
                    description=code_data.get("description"),
                    examples=code_data.get("examples", []),
                    level=code_data.get("level", 1),
                )
                codes[code.code_id] = code

    return codes


def get_all_codes() -> list[IMDRFCode]:
    """Get all IMDRF codes.

    Returns:
        List of all IMDRFCode objects.
    """
    return list(_load_all_codes().values())


def get_code_by_id(code_id: str) -> IMDRFCode | None:
    """Get a specific IMDRF code by its ID.

    Args:
        code_id: The code identifier (e.g., 'A0601').

    Returns:
        IMDRFCode if found, None otherwise.
    """
    return _load_all_codes().get(code_id)


def validate_code(code_id: str) -> bool:
    """Check if a code ID is valid.

    Args:
        code_id: The code identifier to validate.

    Returns:
        True if the code exists, False otherwise.
    """
    return code_id in _load_all_codes()


def get_children(code_id: str) -> list[IMDRFCode]:
    """Get all direct children of a code.

    Args:
        code_id: The parent code identifier.

    Returns:
        List of IMDRFCode objects that have this code as their parent.
    """
    all_codes = _load_all_codes()
    return [code for code in all_codes.values() if code.parent_id == code_id]


def get_ancestors(code_id: str) -> list[IMDRFCode]:
    """Get all ancestors of a code (parent, grandparent, etc.).

    Args:
        code_id: The code identifier.

    Returns:
        List of IMDRFCode objects from immediate parent to root.
        Empty list if code is top-level or not found.
    """
    all_codes = _load_all_codes()
    ancestors: list[IMDRFCode] = []

    current = all_codes.get(code_id)
    if current is None:
        return ancestors

    while current.parent_id is not None:
        parent = all_codes.get(current.parent_id)
        if parent is None:
            break
        ancestors.append(parent)
        current = parent

    return ancestors


def get_full_path(code_id: str) -> str | None:
    """Get the full hierarchical path for a code.

    Args:
        code_id: The code identifier.

    Returns:
        Full path string (e.g., 'Material Problem > Material Integrity > Crack'),
        or None if code not found.
    """
    code = get_code_by_id(code_id)
    if code is None:
        return None

    ancestors = get_ancestors(code_id)
    path_parts = [a.name for a in reversed(ancestors)] + [code.name]
    return " > ".join(path_parts)


def search_codes(
    query: str,
    code_type: IMDRFCodeType | None = None,
    limit: int = 20,
) -> list[IMDRFCode]:
    """Search for codes matching a query string.

    Searches code names, descriptions, and examples.

    Args:
        query: Search query (case-insensitive).
        code_type: Optional filter by code type.
        limit: Maximum number of results to return.

    Returns:
        List of matching IMDRFCode objects, ordered by relevance.
    """
    query_lower = query.lower()
    all_codes = _load_all_codes()
    results: list[tuple[int, IMDRFCode]] = []

    for code in all_codes.values():
        # Apply type filter if specified
        if code_type is not None and code.code_type != code_type:
            continue

        score = 0

        # Exact match in code_id (highest priority)
        if query_lower == code.code_id.lower():
            score += 100

        # Match in name
        if query_lower in code.name.lower():
            score += 50
            # Bonus for match at start of name
            if code.name.lower().startswith(query_lower):
                score += 20

        # Match in description
        if code.description and query_lower in code.description.lower():
            score += 30

        # Match in examples
        for example in code.examples:
            if query_lower in example.lower():
                score += 10
                break  # Only count once

        if score > 0:
            results.append((score, code))

    # Sort by score descending
    results.sort(key=lambda x: x[0], reverse=True)

    return [code for _, code in results[:limit]]


def get_codes_by_type(code_type: IMDRFCodeType) -> list[IMDRFCode]:
    """Get all codes of a specific type.

    Args:
        code_type: The type of codes to retrieve.

    Returns:
        List of IMDRFCode objects of the specified type.
    """
    all_codes = _load_all_codes()
    return [code for code in all_codes.values() if code.code_type == code_type]


def get_top_level_codes(code_type: IMDRFCodeType | None = None) -> list[IMDRFCode]:
    """Get all top-level codes (codes without parents).

    Args:
        code_type: Optional filter by code type.

    Returns:
        List of top-level IMDRFCode objects.
    """
    all_codes = _load_all_codes()
    return [
        code
        for code in all_codes.values()
        if code.is_top_level and (code_type is None or code.code_type == code_type)
    ]


def clear_cache() -> None:
    """Clear the code cache (useful for testing)."""
    _load_all_codes.cache_clear()
