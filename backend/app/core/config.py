"""
Application Configuration

Centralized configuration using Pydantic Settings.
All configuration is loaded from environment variables.

Environment Variables:
    DATABASE_URL: PostgreSQL connection string (NeonDB)
    QDRANT_URL: Vector database URL
    QDRANT_API_KEY: Vector database API key
    HF_TOKEN: Hugging Face API token for model downloads
    ENVIRONMENT: development | staging | production
    CORS_ORIGINS: Comma-separated list of allowed CORS origins (e.g., "http://localhost:3000,https://example.com")

TODO (Iteration 1): Add validation for required secrets
TODO (Iteration 2): Add config for model loading preferences
TODO (Iteration 3): Add feature flags system
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the backend root (backend/) once at import time.
# In Docker (WORKDIR /app) this is /app; locally it's the backend/ folder.
_BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Allow extra env vars (e.g., UPSTASH_REDIS_REST_TOKEN)
    )
    
    # ----- Application -----
    PROJECT_NAME: str = "LLM Research Platform"
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False # Default to False for production readiness
    
    # ----- CORS -----
    # Comma-separated string of allowed origins, parsed into a list.
    # Trailing slashes are stripped automatically (browsers send Origin without them).
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000,https://llmforge.vercel.app"

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS_ORIGINS as a Python list (split on commas, trailing slashes stripped)."""
        return [o.strip().rstrip("/") for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def data_dir(self) -> Path:
        """Absolute path to the data/ directory (datasets, knowledge_base, cache)."""
        return _BACKEND_ROOT / "data"

    @property
    def configs_dir(self) -> Path:
        """Absolute path to the configs/ directory (CoT examples, sample questions)."""
        return _BACKEND_ROOT / "configs"

    # ----- Database (NeonDB) -----
    DATABASE_URL: str = ""  # Required: postgresql://...
    
    # TODO (Iteration 1): Add connection pool settings
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    
    # ----- Vector Database (Qdrant) -----
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "documents"
    
    # ----- Hugging Face -----
    HF_TOKEN: str = ""
    HF_CACHE_DIR: str = "/tmp/hf_cache" # Use a more standard temporary directory
    
    # ----- Model Defaults -----
    DEFAULT_MODEL: str = "meta-llama/Llama-3.2-1B-Instruct"
    DEFAULT_MAX_TOKENS: int = 256
    DEFAULT_TEMPERATURE: float = 0.7
    
    # ----- Inference -----
    INFERENCE_ENGINE: str = "mock"  # mock or hf_api
    INFERENCE_BATCH_SIZE: int = 1
    INFERENCE_TIMEOUT_SECONDS: int = 60
    
    # ----- Redis (Upstash) -----
    # Optional: Only required when using RQ background tasks
    # Validated at runtime in redis.py when connection is requested
    REDIS_URL: str = ""
    
    # ----- Logging -----
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json | text


# Global settings instance
settings = Settings()
