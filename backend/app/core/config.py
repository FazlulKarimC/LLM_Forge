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

TODO (Iteration 1): Add validation for required secrets
TODO (Iteration 2): Add config for model loading preferences
TODO (Iteration 3): Add feature flags system
"""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )
    
    # ----- Application -----
    PROJECT_NAME: str = "LLM Research Platform"
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # ----- CORS -----
    # TODO: Restrict in production
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
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
    HF_CACHE_DIR: str = "./models"
    
    # ----- Model Defaults -----
    DEFAULT_MODEL: str = "microsoft/phi-2"
    DEFAULT_MAX_TOKENS: int = 256
    DEFAULT_TEMPERATURE: float = 0.7
    
    # ----- Inference -----
    # TODO (Iteration 2): Add batching configuration
    INFERENCE_BATCH_SIZE: int = 1
    INFERENCE_TIMEOUT_SECONDS: int = 60
    
    # ----- Logging -----
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json | text


# Global settings instance
settings = Settings()
