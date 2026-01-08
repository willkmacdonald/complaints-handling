"""Tests for LLM client module."""

import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from src.llm import (
    LLMClient,
    LLMConfig,
    LLMResponse,
    ParseError,
    PromptTemplate,
    TokenUsage,
    extract_json_from_response,
    parse_json_response,
    render_prompt,
    validate_response_schema,
)
from src.llm.parsing import extract_field, safe_parse_json
from src.llm.prompts import create_messages


class TestTokenUsage:
    """Tests for TokenUsage model."""

    def test_create_token_usage(self) -> None:
        """Create TokenUsage with values."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_add_token_usage(self) -> None:
        """Add two TokenUsage instances."""
        usage1 = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        usage2 = TokenUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300)

        combined = usage1 + usage2

        assert combined.prompt_tokens == 300
        assert combined.completion_tokens == 150
        assert combined.total_tokens == 450

    def test_default_token_usage(self) -> None:
        """Default TokenUsage has zero values."""
        usage = TokenUsage()

        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0


class TestLLMResponse:
    """Tests for LLMResponse model."""

    def test_create_response(self) -> None:
        """Create LLMResponse with all fields."""
        response = LLMResponse(
            content="Test response",
            model="gpt-4",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
            latency_ms=150.5,
        )

        assert response.content == "Test response"
        assert response.model == "gpt-4"
        assert response.usage.total_tokens == 15
        assert response.finish_reason == "stop"
        assert response.latency_ms == 150.5


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_create_config(self) -> None:
        """Create config with required fields."""
        config = LLMConfig(
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="gpt-4o",
        )

        assert config.endpoint == "https://test.openai.azure.com/"
        assert config.api_key == "test-key"
        assert config.deployment_name == "gpt-4o"
        assert config.api_version == "2024-02-15-preview"  # default

    def test_config_from_env(self) -> None:
        """Load config from environment variables."""
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-4o",
                "AZURE_OPENAI_API_VERSION": "2024-01-01",
            },
        ):
            config = LLMConfig.from_env()

            assert config.endpoint == "https://test.openai.azure.com/"
            assert config.api_key == "test-key"
            assert config.deployment_name == "gpt-4o"
            assert config.api_version == "2024-01-01"

    def test_config_from_env_missing_required(self) -> None:
        """Missing required env vars raise ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear all Azure OpenAI env vars
            for key in [
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_DEPLOYMENT_NAME",
            ]:
                os.environ.pop(key, None)

            with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
                LLMConfig.from_env()


class TestLLMClient:
    """Tests for LLMClient."""

    def test_client_creation(self) -> None:
        """Create client with config."""
        config = LLMConfig(
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="gpt-4o",
        )

        client = LLMClient(config=config)

        assert client.config == config
        assert client._client is not None

    def test_client_tracks_usage(self) -> None:
        """Client tracks cumulative token usage."""
        config = LLMConfig(
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="gpt-4o",
        )
        client = LLMClient(config=config)

        # Initial usage is zero
        assert client.get_total_usage().total_tokens == 0

        # Reset works
        client.reset_usage()
        assert client.get_total_usage().total_tokens == 0

    def test_calculate_backoff(self) -> None:
        """Backoff calculation with exponential increase."""
        config = LLMConfig(
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="gpt-4o",
            base_delay=1.0,
            max_delay=60.0,
        )
        client = LLMClient(config=config)

        # First attempt: base_delay (plus jitter)
        delay0 = client._calculate_backoff(0)
        assert 1.0 <= delay0 <= 1.25

        # Second attempt: 2 * base_delay (plus jitter)
        delay1 = client._calculate_backoff(1)
        assert 2.0 <= delay1 <= 2.5

        # Respects max_delay
        delay_high = client._calculate_backoff(10)
        assert delay_high <= 60.0

        # Uses retry_after if provided
        delay_retry = client._calculate_backoff(0, retry_after=30.0)
        assert delay_retry == 30.0

    def test_complete_success(self) -> None:
        """Successful completion request with mocked client."""
        config = LLMConfig(
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="gpt-4o",
        )
        client = LLMClient(config=config)

        # Create mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        # Patch the client's internal _client
        client._client.chat.completions.create = MagicMock(return_value=mock_response)

        response = client.complete(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.0,
        )

        assert response.content == "Test response"
        assert response.model == "gpt-4o"
        assert response.usage.total_tokens == 15
        assert response.finish_reason == "stop"
        assert response.latency_ms > 0

        # Usage is tracked
        assert client.get_total_usage().total_tokens == 15


class TestPromptTemplate:
    """Tests for PromptTemplate."""

    def test_create_template(self) -> None:
        """Create a prompt template."""
        template = PromptTemplate(
            name="test",
            system_prompt="You are a {{role}}.",
            user_prompt="Help me with {{task}}.",
        )

        assert template.name == "test"
        assert "role" in template.get_variables()
        assert "task" in template.get_variables()

    def test_template_with_defaults(self) -> None:
        """Template with default values."""
        template = PromptTemplate(
            name="test",
            system_prompt="You are a {{role:assistant}}.",
            user_prompt="Help me with {{task}}.",
        )

        assert "assistant" in template.default_variables.get("role", "")
        assert "role" not in template.required_variables
        assert "task" in template.required_variables

    def test_validate_variables(self) -> None:
        """Validate required variables are present."""
        template = PromptTemplate(
            name="test",
            system_prompt="You are a {{role}}.",
            user_prompt="Help me with {{task}}.",
        )

        # Missing both
        missing = template.validate_variables({})
        assert "role" in missing
        assert "task" in missing

        # Partial
        missing = template.validate_variables({"role": "assistant"})
        assert "role" not in missing
        assert "task" in missing

        # All present
        missing = template.validate_variables({"role": "assistant", "task": "coding"})
        assert len(missing) == 0


class TestRenderPrompt:
    """Tests for render_prompt function."""

    def test_render_basic(self) -> None:
        """Render template with variables."""
        template = PromptTemplate(
            name="test",
            system_prompt="You are a {{role}}.",
            user_prompt="Help me with {{task}}.",
        )

        system, user = render_prompt(template, {"role": "assistant", "task": "coding"})

        assert system == "You are a assistant."
        assert user == "Help me with coding."

    def test_render_with_defaults(self) -> None:
        """Render uses defaults for missing variables."""
        template = PromptTemplate(
            name="test",
            system_prompt="You are a {{role:helpful assistant}}.",
            user_prompt="Help me with {{task}}.",
        )

        system, user = render_prompt(template, {"task": "coding"})

        assert system == "You are a helpful assistant."
        assert user == "Help me with coding."

    def test_render_missing_required_raises(self) -> None:
        """Rendering with missing required variables raises."""
        template = PromptTemplate(
            name="test",
            system_prompt="You are a {{role}}.",
            user_prompt="Help me with {{task}}.",
        )

        with pytest.raises(ValueError, match="Missing required variables"):
            render_prompt(template, {"role": "assistant"})


class TestCreateMessages:
    """Tests for create_messages function."""

    def test_basic_messages(self) -> None:
        """Create basic message list."""
        messages = create_messages(
            system_prompt="You are helpful.",
            user_prompt="Hello!",
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello!"

    def test_with_examples(self) -> None:
        """Create messages with few-shot examples."""
        messages = create_messages(
            system_prompt="You are helpful.",
            user_prompt="What is 3+3?",
            examples=[
                ("What is 1+1?", "2"),
                ("What is 2+2?", "4"),
            ],
        )

        assert len(messages) == 6
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is 1+1?"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "2"
        assert messages[5]["role"] == "user"
        assert messages[5]["content"] == "What is 3+3?"


class TestExtractJsonFromResponse:
    """Tests for extract_json_from_response."""

    def test_extract_raw_json(self) -> None:
        """Extract raw JSON object."""
        response = '{"key": "value"}'
        result = extract_json_from_response(response)
        assert result == '{"key": "value"}'

    def test_extract_from_code_block(self) -> None:
        """Extract JSON from markdown code block."""
        response = """Here's the result:
```json
{"key": "value"}
```"""
        result = extract_json_from_response(response)
        assert result == '{"key": "value"}'

    def test_extract_from_plain_code_block(self) -> None:
        """Extract JSON from plain code block."""
        response = """Result:
```
{"key": "value"}
```"""
        result = extract_json_from_response(response)
        assert result == '{"key": "value"}'

    def test_extract_embedded_json(self) -> None:
        """Extract JSON embedded in text."""
        response = 'The answer is {"key": "value"} as shown.'
        result = extract_json_from_response(response)
        assert result == '{"key": "value"}'

    def test_extract_array(self) -> None:
        """Extract JSON array."""
        response = "[1, 2, 3]"
        result = extract_json_from_response(response)
        assert result == "[1, 2, 3]"

    def test_extract_empty_raises(self) -> None:
        """Empty response raises ParseError."""
        with pytest.raises(ParseError, match="Empty response"):
            extract_json_from_response("")

    def test_extract_no_json_raises(self) -> None:
        """Response without JSON raises ParseError."""
        with pytest.raises(ParseError, match="Could not extract"):
            extract_json_from_response("No JSON here!")


class TestParseJsonResponse:
    """Tests for parse_json_response."""

    def test_parse_object(self) -> None:
        """Parse JSON object."""
        result = parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_array(self) -> None:
        """Parse JSON array."""
        result = parse_json_response("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_parse_from_code_block(self) -> None:
        """Parse JSON from code block."""
        result = parse_json_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_parse_invalid_raises(self) -> None:
        """Invalid JSON raises ParseError."""
        with pytest.raises(ParseError):
            parse_json_response("not json")


class TestValidateResponseSchema:
    """Tests for validate_response_schema."""

    def test_validate_valid_response(self) -> None:
        """Validate response matching schema."""

        class TestModel(BaseModel):
            name: str
            value: int

        result = validate_response_schema(
            '{"name": "test", "value": 42}',
            TestModel,
        )

        assert isinstance(result, TestModel)
        assert result.name == "test"
        assert result.value == 42

    def test_validate_from_code_block(self) -> None:
        """Validate JSON from code block."""

        class TestModel(BaseModel):
            name: str

        result = validate_response_schema(
            '```json\n{"name": "test"}\n```',
            TestModel,
        )

        assert result.name == "test"

    def test_validate_missing_field_raises(self) -> None:
        """Missing required field raises ParseError."""

        class TestModel(BaseModel):
            name: str
            value: int

        with pytest.raises(ParseError, match="validation failed"):
            validate_response_schema('{"name": "test"}', TestModel)

    def test_validate_wrong_type_raises(self) -> None:
        """Wrong field type raises ParseError."""

        class TestModel(BaseModel):
            value: int

        with pytest.raises(ParseError, match="validation failed"):
            validate_response_schema('{"value": "not an int"}', TestModel)


class TestSafeParseJson:
    """Tests for safe_parse_json."""

    def test_parse_valid(self) -> None:
        """Parse valid JSON."""
        result = safe_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_invalid_returns_default(self) -> None:
        """Invalid JSON returns default."""
        result = safe_parse_json("not json", default={"default": True})
        assert result == {"default": True}

    def test_parse_array_returns_default(self) -> None:
        """Array returns default (expects dict)."""
        result = safe_parse_json("[1, 2, 3]", default={})
        assert result == {}


class TestExtractField:
    """Tests for extract_field."""

    def test_extract_simple(self) -> None:
        """Extract simple field."""
        data = {"name": "test", "value": 42}
        assert extract_field(data, "name") == "test"
        assert extract_field(data, "value") == 42

    def test_extract_nested(self) -> None:
        """Extract nested field."""
        data = {"outer": {"inner": "value"}}
        assert extract_field(data, "outer.inner") == "value"

    def test_extract_array_index(self) -> None:
        """Extract array element by index."""
        data = {"items": ["a", "b", "c"]}
        assert extract_field(data, "items.0") == "a"
        assert extract_field(data, "items.2") == "c"

    def test_extract_missing_returns_default(self) -> None:
        """Missing field returns default."""
        data = {"name": "test"}
        assert extract_field(data, "missing") is None
        assert extract_field(data, "missing", default="default") == "default"

    def test_extract_deep_nested(self) -> None:
        """Extract deeply nested field."""
        data = {
            "suggestions": [
                {"code_id": "A01", "confidence": 0.9},
                {"code_id": "A02", "confidence": 0.7},
            ]
        }
        assert extract_field(data, "suggestions.0.code_id") == "A01"
        assert extract_field(data, "suggestions.1.confidence") == 0.7


class TestIMDRFCodingTemplate:
    """Tests for the IMDRF coding prompt template."""

    def test_imdrf_template_exists(self) -> None:
        """IMDRF coding template is importable."""
        from src.llm.prompts import IMDRF_CODING_TEMPLATE

        assert IMDRF_CODING_TEMPLATE.name == "imdrf_coding"

    def test_imdrf_template_variables(self) -> None:
        """IMDRF template has expected variables."""
        from src.llm.prompts import IMDRF_CODING_TEMPLATE

        variables = IMDRF_CODING_TEMPLATE.get_variables()
        assert "device_name" in variables
        assert "manufacturer" in variables
        assert "narrative" in variables
        assert "available_codes" in variables


class TestMDRDeterminationTemplate:
    """Tests for the MDR determination prompt template."""

    def test_mdr_template_exists(self) -> None:
        """MDR determination template is importable."""
        from src.llm.prompts import MDR_DETERMINATION_TEMPLATE

        assert MDR_DETERMINATION_TEMPLATE.name == "mdr_determination"

    def test_mdr_template_variables(self) -> None:
        """MDR template has expected variables."""
        from src.llm.prompts import MDR_DETERMINATION_TEMPLATE

        variables = MDR_DETERMINATION_TEMPLATE.get_variables()
        assert "device_name" in variables
        assert "event_description" in variables
