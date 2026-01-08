"""Prompt template management for LLM interactions."""

import re
from typing import Any

from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    """A reusable prompt template with variable substitution.

    Templates use {{variable_name}} syntax for placeholders.
    Variables can have default values: {{variable_name:default_value}}
    """

    name: str = Field(..., description="Template name for identification")
    system_prompt: str = Field(..., description="System message template")
    user_prompt: str = Field(..., description="User message template")
    description: str = Field(
        default="", description="Description of the template's purpose"
    )
    required_variables: list[str] = Field(
        default_factory=list, description="Variables that must be provided"
    )
    default_variables: dict[str, str] = Field(
        default_factory=dict, description="Default values for optional variables"
    )

    def model_post_init(self, __context: Any) -> None:
        """Extract required variables from templates after initialization."""
        # Find all variables in both templates
        all_vars = set()
        pattern = r"\{\{(\w+)(?::([^}]*))?\}\}"

        for template in [self.system_prompt, self.user_prompt]:
            for match in re.finditer(pattern, template):
                var_name = match.group(1)
                default_value = match.group(2)
                all_vars.add(var_name)
                if default_value is not None and var_name not in self.default_variables:
                    self.default_variables[var_name] = default_value

        # Update required variables (those without defaults)
        if not self.required_variables:
            self.required_variables = [
                v for v in all_vars if v not in self.default_variables
            ]

    def get_variables(self) -> set[str]:
        """Get all variable names used in the template.

        Returns:
            Set of variable names found in both system and user prompts.
        """
        pattern = r"\{\{(\w+)(?::[^}]*)?\}\}"
        variables = set()
        for template in [self.system_prompt, self.user_prompt]:
            variables.update(re.findall(pattern, template))
        return variables

    def validate_variables(self, variables: dict[str, Any]) -> list[str]:
        """Check if all required variables are provided.

        Args:
            variables: Dictionary of variable values.

        Returns:
            List of missing required variable names.
        """
        missing = []
        for var in self.required_variables:
            if var not in variables and var not in self.default_variables:
                missing.append(var)
        return missing


def _substitute_variables(template: str, variables: dict[str, Any]) -> str:
    """Substitute variables in a template string.

    Args:
        template: Template string with {{variable}} placeholders.
        variables: Dictionary of variable values.

    Returns:
        Template with variables substituted.
    """

    def replace_var(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default_value = match.group(2)

        if var_name in variables:
            value = variables[var_name]
            return str(value) if value is not None else ""

        if default_value is not None:
            return default_value

        # Return empty string for missing variables
        return ""

    pattern = r"\{\{(\w+)(?::([^}]*))?\}\}"
    return re.sub(pattern, replace_var, template)


def render_prompt(
    template: PromptTemplate,
    variables: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Render a prompt template with variable substitution.

    Args:
        template: The prompt template to render.
        variables: Dictionary of variable values to substitute.

    Returns:
        Tuple of (system_prompt, user_prompt) with variables substituted.

    Raises:
        ValueError: If required variables are missing.
    """
    variables = variables or {}

    # Merge with defaults
    merged_vars = {**template.default_variables, **variables}

    # Validate required variables
    missing = template.validate_variables(merged_vars)
    if missing:
        raise ValueError(f"Missing required variables: {', '.join(missing)}")

    # Render templates
    system = _substitute_variables(template.system_prompt, merged_vars)
    user = _substitute_variables(template.user_prompt, merged_vars)

    return system, user


def create_messages(
    system_prompt: str,
    user_prompt: str,
    examples: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Create a message list for chat completion.

    Args:
        system_prompt: The system message.
        user_prompt: The user message.
        examples: Optional list of (user, assistant) example pairs for few-shot.

    Returns:
        List of message dictionaries ready for the API.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Add few-shot examples if provided
    if examples:
        for user_example, assistant_example in examples:
            messages.append({"role": "user", "content": user_example})
            messages.append({"role": "assistant", "content": assistant_example})

    # Add the actual user prompt
    messages.append({"role": "user", "content": user_prompt})

    return messages


# Common prompt templates for the application


IMDRF_CODING_TEMPLATE = PromptTemplate(
    name="imdrf_coding",
    description="Suggest IMDRF codes for a medical device complaint",
    system_prompt="""You are an expert medical device regulatory specialist with deep knowledge of IMDRF (International Medical Device Regulators Forum) adverse event terminology.

Your task is to analyze medical device complaints and suggest appropriate IMDRF codes.

## IMDRF Code Categories
- **Annex A (Device Problem Codes)**: Problems with the device itself (e.g., material issues, software bugs, mechanical failures)
- **Annex C (Patient Problem Codes)**: Problems affecting the patient (e.g., injuries, clinical symptoms)

## Guidelines
1. Analyze the complaint narrative carefully
2. Identify both device problems AND patient problems
3. Provide confidence scores (0.0-1.0) for each suggestion
4. Cite the specific text that supports each code
5. Only suggest codes from the valid IMDRF reference
6. When uncertain, suggest the broader parent code

## Available Codes
{{available_codes}}

## Output Format
Respond with a JSON object containing your suggestions:
```json
{
  "suggestions": [
    {
      "code_id": "A0601",
      "code_name": "Material Problem",
      "confidence": 0.85,
      "source_text": "the relevant text from the complaint",
      "reasoning": "explanation of why this code applies"
    }
  ]
}
```""",
    user_prompt="""Analyze the following medical device complaint and suggest appropriate IMDRF codes.

## Device Information
- Device: {{device_name}}
- Manufacturer: {{manufacturer}}
- Type: {{device_type}}

## Complaint Narrative
{{narrative}}

## Patient Outcome
{{patient_outcome:Not specified}}

Please provide your IMDRF code suggestions in the specified JSON format.""",
)


MDR_DETERMINATION_TEMPLATE = PromptTemplate(
    name="mdr_determination",
    description="Determine if a complaint requires MDR (Medical Device Report) filing",
    system_prompt="""You are a medical device regulatory expert specializing in FDA MDR (Medical Device Reporting) requirements.

## MDR Reporting Criteria (21 CFR 803)
A manufacturer must report when they become aware that a device may have:
1. **Caused or contributed to a death**
2. **Caused or contributed to a serious injury**
3. **Malfunctioned in a way that would be likely to cause or contribute to death or serious injury if the malfunction were to recur**

## Serious Injury Definition
An injury or illness that:
- Is life-threatening
- Results in permanent impairment or damage to a body function/structure
- Necessitates medical or surgical intervention to prevent such impairment

## Key Considerations
- User error alone (without device malfunction) typically does NOT require MDR
- Near-misses where malfunction COULD have caused harm if it recurs DO require MDR
- When in doubt, err on the side of requiring review (false positives are acceptable)

## Output Format
```json
{
  "requires_mdr": true,
  "confidence": 0.9,
  "criteria_met": ["death", "serious_injury", "malfunction_could_cause_harm"],
  "reasoning": "explanation of the determination",
  "evidence": ["specific text supporting the determination"]
}
```""",
    user_prompt="""Analyze the following medical device complaint and determine if it requires MDR filing.

## Device Information
- Device: {{device_name}}
- Manufacturer: {{manufacturer}}

## Event Description
{{event_description}}

## Patient Outcome
{{patient_outcome:Not specified}}

## Device Status
{{device_status:Unknown}}

Please provide your MDR determination in the specified JSON format.""",
)
