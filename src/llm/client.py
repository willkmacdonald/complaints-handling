"""Azure OpenAI client wrapper with retry logic and token tracking."""

import contextlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0
DEFAULT_TIMEOUT = 60.0


class TokenUsage(BaseModel):
    """Token usage statistics for an LLM call."""

    prompt_tokens: int = Field(default=0, description="Tokens in the prompt")
    completion_tokens: int = Field(default=0, description="Tokens in the completion")
    total_tokens: int = Field(default=0, description="Total tokens used")

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage instances together."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class LLMResponse(BaseModel):
    """Response from an LLM call."""

    content: str = Field(..., description="The response content")
    model: str = Field(..., description="Model that generated the response")
    usage: TokenUsage = Field(
        default_factory=TokenUsage, description="Token usage statistics"
    )
    finish_reason: str | None = Field(
        default=None, description="Reason the generation stopped"
    )
    latency_ms: float = Field(
        default=0.0, description="Request latency in milliseconds"
    )


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""

    endpoint: str
    api_key: str
    deployment_name: str
    api_version: str = "2024-02-15-preview"
    max_retries: int = DEFAULT_MAX_RETRIES
    base_delay: float = DEFAULT_BASE_DELAY
    max_delay: float = DEFAULT_MAX_DELAY
    timeout: float = DEFAULT_TIMEOUT

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create config from environment variables.

        Required environment variables:
            AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL
            AZURE_OPENAI_API_KEY: API key
            AZURE_OPENAI_DEPLOYMENT_NAME: Deployment name (e.g., gpt-4o)

        Optional environment variables:
            AZURE_OPENAI_API_VERSION: API version (default: 2024-02-15-preview)
            LLM_MAX_RETRIES: Maximum retry attempts (default: 3)
            LLM_TIMEOUT: Request timeout in seconds (default: 60)

        Raises:
            ValueError: If required environment variables are missing.
        """
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")
        if not deployment_name:
            raise ValueError(
                "AZURE_OPENAI_DEPLOYMENT_NAME environment variable is required"
            )

        return cls(
            endpoint=endpoint,
            api_key=api_key,
            deployment_name=deployment_name,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))),
            timeout=float(os.getenv("LLM_TIMEOUT", str(DEFAULT_TIMEOUT))),
        )


class LLMError(Exception):
    """Base exception for LLM errors."""

    pass


class LLMConnectionError(LLMError):
    """Error connecting to the LLM service."""

    pass


class LLMRateLimitError(LLMError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMResponseError(LLMError):
    """Error in LLM response."""

    pass


@dataclass
class LLMClient:
    """Client for Azure OpenAI with retry logic and token tracking.

    This client wraps the Azure OpenAI API with:
    - Automatic retry with exponential backoff
    - Token usage tracking
    - Request latency measurement
    - Error handling and logging
    """

    config: LLMConfig
    _client: Any = field(default=None, repr=False)
    _total_usage: TokenUsage = field(default_factory=TokenUsage, repr=False)

    def __post_init__(self) -> None:
        """Initialize the Azure OpenAI client."""
        self._init_client()

    def _init_client(self) -> None:
        """Initialize or reinitialize the underlying client."""
        try:
            from openai import AzureOpenAI

            self._client = AzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                api_version=self.config.api_version,
                timeout=self.config.timeout,
            )
        except ImportError:
            logger.warning(
                "openai package not installed. Install with: pip install openai"
            )
            self._client = None

    def _calculate_backoff(
        self, attempt: int, retry_after: float | None = None
    ) -> float:
        """Calculate backoff delay with exponential increase and jitter.

        Args:
            attempt: Current attempt number (0-indexed).
            retry_after: Optional retry-after value from rate limit response.

        Returns:
            Delay in seconds before next retry.
        """
        if retry_after is not None:
            return float(min(retry_after, self.config.max_delay))

        # Exponential backoff: base_delay * 2^attempt
        delay = self.config.base_delay * (2**attempt)
        # Add jitter (up to 25% of delay)
        import random

        jitter = delay * 0.25 * random.random()
        return float(min(delay + jitter, self.config.max_delay))

    def _is_retryable_error(self, error: Exception) -> tuple[bool, float | None]:
        """Check if an error is retryable and extract retry-after if available.

        Args:
            error: The exception that occurred.

        Returns:
            Tuple of (is_retryable, retry_after_seconds).
        """
        # Import here to avoid issues if openai not installed
        try:
            from openai import APIConnectionError, APITimeoutError, RateLimitError
        except ImportError:
            return False, None

        if isinstance(error, RateLimitError):
            # Try to extract retry-after from headers
            retry_after = None
            if hasattr(error, "response") and error.response is not None:
                retry_after_str = error.response.headers.get("retry-after")
                if retry_after_str:
                    with contextlib.suppress(ValueError):
                        retry_after = float(retry_after_str)
            return True, retry_after

        if isinstance(error, (APIConnectionError, APITimeoutError)):
            return True, None

        # Check for specific HTTP status codes that are retryable
        if hasattr(error, "status_code") and error.status_code in (
            429,
            500,
            502,
            503,
            504,
        ):
            return True, None

        return False, None

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request with automatic retry.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0 for deterministic).
            max_tokens: Maximum tokens in response (None for model default).
            response_format: Optional response format (e.g., {"type": "json_object"}).

        Returns:
            LLMResponse with content, usage, and metadata.

        Raises:
            LLMConnectionError: If unable to connect after retries.
            LLMRateLimitError: If rate limited and retries exhausted.
            LLMResponseError: If response is invalid.
            LLMError: For other errors.
        """
        if self._client is None:
            raise LLMConnectionError(
                "OpenAI client not initialized. Is the openai package installed?"
            )

        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.perf_counter()

                # Build request kwargs
                kwargs: dict[str, Any] = {
                    "model": self.config.deployment_name,
                    "messages": messages,
                    "temperature": temperature,
                }
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                if response_format is not None:
                    kwargs["response_format"] = response_format

                # Make the request
                response = self._client.chat.completions.create(**kwargs)

                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000

                # Extract response data
                choice = response.choices[0]
                content = choice.message.content or ""

                # Build usage
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=(
                        response.usage.completion_tokens if response.usage else 0
                    ),
                    total_tokens=response.usage.total_tokens if response.usage else 0,
                )

                # Track cumulative usage
                self._total_usage = self._total_usage + usage

                logger.debug(
                    "LLM request completed: %d tokens, %.0fms",
                    usage.total_tokens,
                    latency_ms,
                )

                return LLMResponse(
                    content=content,
                    model=response.model,
                    usage=usage,
                    finish_reason=choice.finish_reason,
                    latency_ms=latency_ms,
                )

            except Exception as e:
                last_error = e
                is_retryable, retry_after = self._is_retryable_error(e)

                if not is_retryable or attempt >= self.config.max_retries:
                    # No more retries
                    break

                # Calculate backoff and wait
                delay = self._calculate_backoff(attempt, retry_after)
                logger.warning(
                    "LLM request failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self.config.max_retries + 1,
                    delay,
                    str(e),
                )
                time.sleep(delay)

        # All retries exhausted
        if last_error is not None:
            error_msg = f"LLM request failed after {self.config.max_retries + 1} attempts: {last_error}"
            logger.error(error_msg)

            # Classify the error
            try:
                from openai import RateLimitError

                if isinstance(last_error, RateLimitError):
                    raise LLMRateLimitError(error_msg) from last_error
            except ImportError:
                pass

            raise LLMError(error_msg) from last_error

        raise LLMError("Unknown error in LLM request")

    def get_total_usage(self) -> TokenUsage:
        """Get cumulative token usage across all requests.

        Returns:
            TokenUsage with total tokens used by this client instance.
        """
        return self._total_usage

    def reset_usage(self) -> None:
        """Reset the cumulative token usage counter."""
        self._total_usage = TokenUsage()


# Module-level default client (lazy initialization)
_default_client: LLMClient | None = None


def create_client(config: LLMConfig | None = None) -> LLMClient:
    """Create a new LLM client.

    Args:
        config: Optional configuration. If not provided, loads from environment.

    Returns:
        Configured LLMClient instance.
    """
    if config is None:
        config = LLMConfig.from_env()
    return LLMClient(config=config)


def get_default_client() -> LLMClient:
    """Get or create the default LLM client.

    The default client is created once and reused. Configuration is loaded
    from environment variables.

    Returns:
        The default LLMClient instance.

    Raises:
        ValueError: If required environment variables are missing.
    """
    global _default_client
    if _default_client is None:
        _default_client = create_client()
    return _default_client
