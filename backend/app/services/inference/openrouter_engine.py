"""
OpenRouter Inference Engine

Thin adapter over OpenAIEngine for the OpenRouter API.
OpenRouter provides free models via an OpenAI-compatible endpoint.

Free models (as of 2025-01):
- meta-llama/llama-3.1-8b-instruct:free
- qwen/qwen3-8b:free
- google/gemma-3-1b-it:free
"""

import logging
from typing import Optional

from app.services.inference.openai_engine import OpenAIEngine
from app.core.config import settings

logger = logging.getLogger(__name__)

# Base URL for OpenRouter's OpenAI-compatible API
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Free models available on OpenRouter
OPENROUTER_FREE_MODELS = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "qwen/qwen3-8b:free",
    "google/gemma-3-1b-it:free",
]


class OpenRouterEngine(OpenAIEngine):
    """
    OpenRouter inference engine.

    Inherits all generation logic from OpenAIEngine —
    only overrides initialization to set the correct base_url and API key.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "meta-llama/llama-3.1-8b-instruct:free",
    ):
        """
        Initialize OpenRouter engine.

        Args:
            api_key: OpenRouter API key (from settings/.env if None)
            model_name: Model identifier on OpenRouter
        """
        resolved_key = api_key or settings.OPENROUTER_API_KEY
        if not resolved_key:
            raise ValueError(
                "OpenRouter API key is required. "
                "Set OPENROUTER_API_KEY in your .env file. "
                "Get a free key at https://openrouter.ai/keys"
            )

        super().__init__(
            base_url=OPENROUTER_BASE_URL,
            api_key=resolved_key,
            model_name=model_name,
        )
        logger.info(
            "OpenRouterEngine initialized (model=%s, base_url=%s)",
            model_name, OPENROUTER_BASE_URL,
        )
