"""
FastAPI Application Entry Point

This module initializes the FastAPI application with:
- CORS middleware for frontend communication
- API routers for experiments, results, and metrics
- Lifespan events for startup/shutdown
- Health check endpoint
"""

import logging
import logging.config
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api import experiments, results, health
from app.core.middleware import RequestContextMiddleware
from app.core.custom_exceptions import AppException
from fastapi.exceptions import RequestValidationError
from app.core.exception_handlers import (
    app_exception_handler,
    validation_exception_handler,
    global_exception_handler,
)

# ── Logging setup ──────────────────────────────────────────────────────────
# Configure logging early so all modules (including uvicorn workers in
# HF Spaces / Docker) emit structured output captured by the log viewer.
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Startup:
        - Initialize database connection pool
        - Load ML models into memory (if configured)
        - Connect to vector database
    
    Shutdown:
        - Close database connections
        - Cleanup GPU memory
        - Flush pending logs
    
    TODO (Iteration 1): Implement database initialization
    TODO (Iteration 2): Add model preloading with GPU memory management
    TODO (Iteration 3): Add graceful shutdown with request draining
    """
    # Startup
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} (env={settings.ENVIRONMENT})")

    # Warn if HF_TOKEN is missing when HF API inference is expected
    if settings.INFERENCE_ENGINE == "hf_api" and not settings.HF_TOKEN:
        logger.warning(
            "HF_TOKEN is not set but INFERENCE_ENGINE=hf_api. "
            "All inference calls will fail. Set HF_TOKEN in your environment."
        )

    logger.info(f"CORS allowed origins: {settings.cors_origins_list}")

    yield

    # Shutdown
    logger.info("Shutting down...")


def create_application() -> FastAPI:
    """
    Application factory pattern.
    
    Creates and configures the FastAPI application instance.
    This pattern allows for easier testing and multiple app instances.
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="LLM Research Engineering Platform",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        redirect_slashes=False,
        lifespan=lifespan,
    )
    
    # Configure CORS — origins are read from the CORS_ORIGINS env var
    # so additional origins (e.g. Vercel URL) can be added without code changes.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add our custom Request ID middleware
    app.add_middleware(RequestContextMiddleware)
    
    # Register global exception handlers
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)
    
    # Register routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(
        experiments.router,
        prefix=f"{settings.API_V1_PREFIX}/experiments",
        tags=["Experiments"],
    )
    app.include_router(
        results.router,
        prefix=f"{settings.API_V1_PREFIX}/results",
        tags=["Results"],
    )
    
    return app


app = create_application()
