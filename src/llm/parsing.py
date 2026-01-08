"""Response parsing utilities for LLM outputs."""

import json
import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ParseError(Exception):
    """Error parsing LLM response."""

    def __init__(self, message: str, raw_response: str | None = None):
        super().__init__(message)
        self.raw_response = raw_response


def extract_json_from_response(response: str) -> str:
    """Extract JSON from an LLM response that may contain markdown or other text.

    Handles common patterns:
    - JSON wrapped in ```json ... ``` code blocks
    - JSON wrapped in ``` ... ``` code blocks
    - Raw JSON objects/arrays
    - JSON embedded in explanatory text

    Args:
        response: Raw LLM response text.

    Returns:
        Extracted JSON string.

    Raises:
        ParseError: If no valid JSON can be extracted.
    """
    if not response:
        raise ParseError("Empty response", raw_response=response)

    response = response.strip()

    # Try to extract from markdown code blocks first
    # Pattern: ```json ... ``` or ``` ... ```
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    code_matches = re.findall(code_block_pattern, response)

    for match in code_matches:
        match_str: str = str(match).strip()
        if match_str.startswith("{") or match_str.startswith("["):
            try:
                json.loads(match_str)
                return match_str
            except json.JSONDecodeError:
                continue

    # Try to find raw JSON object
    # Look for outermost { } or [ ]
    json_patterns = [
        r"(\{[\s\S]*\})",  # Object
        r"(\[[\s\S]*\])",  # Array
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            match_str = str(match)
            try:
                json.loads(match_str)
                return match_str
            except json.JSONDecodeError:
                continue

    # If response itself is valid JSON, return it
    try:
        json.loads(response)
        return response
    except json.JSONDecodeError:
        pass

    raise ParseError(
        "Could not extract valid JSON from response",
        raw_response=response,
    )


def parse_json_response(response: str) -> dict[str, Any] | list[Any]:
    """Parse LLM response as JSON.

    Args:
        response: Raw LLM response text.

    Returns:
        Parsed JSON as dict or list.

    Raises:
        ParseError: If response cannot be parsed as JSON.
    """
    try:
        json_str = extract_json_from_response(response)
        return json.loads(json_str)  # type: ignore[no-any-return]
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}", raw_response=response) from e


def validate_response_schema(
    response: str,
    model_class: type[T],
) -> T:
    """Parse and validate LLM response against a Pydantic model.

    Args:
        response: Raw LLM response text.
        model_class: Pydantic model class to validate against.

    Returns:
        Validated model instance.

    Raises:
        ParseError: If response cannot be parsed or doesn't match schema.
    """
    try:
        data = parse_json_response(response)
        if not isinstance(data, dict):
            raise ParseError(
                f"Expected JSON object, got {type(data).__name__}",
                raw_response=response,
            )
        return model_class.model_validate(data)
    except ValidationError as e:
        # Format validation errors nicely
        errors = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            msg = err["msg"]
            errors.append(f"  {loc}: {msg}")
        error_msg = "Response validation failed:\n" + "\n".join(errors)
        raise ParseError(error_msg, raw_response=response) from e


def safe_parse_json(
    response: str,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Safely parse JSON response, returning default on failure.

    Args:
        response: Raw LLM response text.
        default: Default value to return on parse failure.

    Returns:
        Parsed JSON dict or default value.
    """
    try:
        result = parse_json_response(response)
        if isinstance(result, dict):
            return result
        return default or {}
    except ParseError as e:
        logger.warning("Failed to parse JSON response: %s", e)
        return default or {}


def extract_field(
    data: dict[str, Any],
    field_path: str,
    default: Any = None,
) -> Any:
    """Extract a nested field from parsed JSON data.

    Args:
        data: Parsed JSON dictionary.
        field_path: Dot-separated path to the field (e.g., "suggestions.0.code_id").
        default: Default value if field not found.

    Returns:
        Field value or default.
    """
    parts = field_path.split(".")
    current: Any = data

    for part in parts:
        if current is None:
            return default

        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx] if 0 <= idx < len(current) else None
            except (ValueError, IndexError):
                return default
        else:
            return default

    return current if current is not None else default
