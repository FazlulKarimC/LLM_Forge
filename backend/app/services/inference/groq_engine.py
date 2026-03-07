"""
Groq Inference Engine

Thin adapter over OpenAIEngine for the Groq API.
Groq provides free, fast inference via an OpenAI-compatible endpoint.

Free models (as of 2025-01):
- llama-3.1-8b-instant
- gemma2-9b-it
- mixtral-8x7b-32768

Rate limits (free tier):
- 30 requests/minute
- 14,400 requests/day
- 6,000 tokens/minute
"""

import logging
from typing import Optional

from app.services.inference.openai_engine import OpenAIEngine
from app.core.config import settings

logger = logging.getLogger(__name__)

# Base URL for Groq's OpenAI-compatible API
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Free models available on Groq
GROQ_FREE_MODELS = [
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]


class GroqEngine(OpenAIEngine):
    """
    Groq inference engine.

    Inherits all generation logic from OpenAIEngine —
    only overrides initialization to set the correct base_url and API key.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "llama-3.1-8b-instant",
    ):
        """
        Initialize Groq engine.

        Args:
            api_key: Groq API key (from settings/.env if None)
            model_name: Model identifier on Groq
        """
        resolved_key = api_key or settings.GROQ_API_KEY
        if not resolved_key:
            raise ValueError(
                "Groq API key is required. "
                "Set GROQ_API_KEY in your .env file. "
                "Get a free key at https://console.groq.com/keys"
            )

        super().__init__(
            base_url=GROQ_BASE_URL,
            api_key=resolved_key,
            model_name=model_name,
        )
        logger.info(
            "GroqEngine initialized (model=%s, base_url=%s)",
            model_name, GROQ_BASE_URL,
        )
