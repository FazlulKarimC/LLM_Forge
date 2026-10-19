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

"""

from pathlib import Path

from pydantic import Field
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
    PROJECT_NAME: str = "LlmForge"
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"
    LEGACY_DATASET_WRITES: bool = False
    DEBUG: bool = False # Default to False for production readiness
    # Non-secret release metadata only. Normally embedded by the build workflow.
    BUILD_COMMIT: str = ""
    BUILD_ID: str = ""
    
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
    
    DB_POOL_SIZE: int = 2
    DB_MAX_OVERFLOW: int = 1
    DB_POOL_TIMEOUT: float = 10

    # Clerk session verification. No development authentication bypass.
    CLERK_ISSUER_URL: str = ""
    CLERK_AUTHORIZED_PARTIES: str = "http://localhost:3000"
    CLERK_AUDIENCE: str = ""
    CLERK_JWT_PUBLIC_KEY: str = ""  # Optional PEM for networkless verification
    
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

    # Small hosted benchmark profile. These are application budgets, not quotas.
    BENCHMARK_MAX_SAMPLES: int = Field(20, ge=1, le=500)
    BENCHMARK_MAX_CALLS: int = Field(20, ge=1, le=1000)
    BENCHMARK_MAX_SECONDS: int = Field(300, ge=1, le=3600)
    BENCHMARK_CALL_TIMEOUT_SECONDS: int = Field(30, ge=1, le=120)
    BENCHMARK_REACT_MAX_SAMPLES: int = Field(5, ge=1, le=20)
    BENCHMARK_REACT_MAX_ITERATIONS: int = Field(3, ge=1, le=20)
    
    # ----- Multi-Provider (Phase 6) -----
    OPENROUTER_API_KEY: str = ""   # Get from https://openrouter.ai/keys
    GROQ_API_KEY: str = ""         # Get from https://console.groq.com/keys
    
    # ----- Redis (Upstash) -----
    # Optional: Only required when using RQ background tasks
    # Validated at runtime in redis.py when connection is requested
    REDIS_URL: str = ""

    # ----- Task Dispatch / Queue -----
    # auto: use RQ if Upstash healthy + worker alive, else inline
    # upstash: always try RQ (fail hard if unavailable)
    # inline: always run inline via BackgroundTasks
    QUEUE_BACKEND_MODE: str = "inline"
    UPSTASH_CIRCUIT_OPEN_MINUTES: int = 30
    UPSTASH_HEALTHCHECK_TIMEOUT_MS: int = 1500
    UPSTASH_HEALTHCHECK_CACHE_SECONDS: int = 60
    RQ_WORKER_HEARTBEAT_INTERVAL_SECONDS: int = 30
    RQ_WORKER_HEARTBEAT_TTL_SECONDS: int = 90
    
    # ----- Logging -----
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json | text

    # ----- Observability (Sentry) -----
    SENTRY_DSN: str = ""  # Optional: Sentry DSN for error reporting


# Global settings instance
settings = Settings()
