"""
Shared Retry Configuration

Provides reusable tenacity-based retry decorators for all API calls.
Uses exponential backoff with jitter to gracefully handle rate limits
from free-tier providers (HF, OpenRouter, Groq).
"""

import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# Exceptions that should trigger a retry
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
)

# Try to import provider-specific retryable errors
try:
    from openai import RateLimitError, APIConnectionError, APITimeoutError
    RETRYABLE_EXCEPTIONS = (
        ConnectionError,
        TimeoutError,
        RateLimitError,
        APIConnectionError,
        APITimeoutError,
    )
except ImportError:
    pass


def llm_retry(max_attempts: int = 3, initial_wait: float = 1.0, max_wait: float = 30.0):
    """
    Create a tenacity retry decorator for LLM API calls.

    Uses exponential backoff with jitter:
        Attempt 1: immediate
        Attempt 2: ~1-2s wait
        Attempt 3: ~4-8s wait

    Only retries on rate limits, connection errors, and timeouts.
    Does NOT retry on auth errors, model-not-found, or validation errors.

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        initial_wait: Initial wait time in seconds (default: 1.0)
        max_wait: Maximum wait time in seconds (default: 30.0)

    Returns:
        Configured tenacity retry decorator
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=initial_wait, max=max_wait, jitter=2),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
