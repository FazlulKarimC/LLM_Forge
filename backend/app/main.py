"""
FastAPI Application Entry Point

This module configures CORS, exception handling, project-scoped API routers,
and application startup/shutdown logging.
"""

import logging
import logging.config
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api import experiments, results, health
from app.api import prompts as prompts_api
from app.api import workspaces
from app.api import prompt_library, project_keys, sdk_prompts
from app.api import datasets, evaluations
from app.api import sdk_evaluations
from app.api import demo
from app.core.tenancy import get_project_context
from app.core.middleware import RequestContextMiddleware
from app.core.custom_exceptions import AppException
from fastapi.exceptions import RequestValidationError
from app.core.exception_handlers import (
    app_exception_handler,
    validation_exception_handler,
    global_exception_handler,
    http_exception_handler,
)
from starlette.exceptions import HTTPException as StarletteHTTPException

# ── Logging setup ──────────────────────────────────────────────────────────
# Configure logging early so all modules (including uvicorn workers in
# HF Spaces / Docker) emit structured output captured by the log viewer.
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# ── Sentry error monitoring (optional) ─────────────────────────────────
# Enabled only when SENTRY_DSN is set. Captures unhandled exceptions.
# No performance tracing, no PII collection.
if settings.SENTRY_DSN:
    import sentry_sdk
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        traces_sample_rate=0.0,
        enable_tracing=False,
        send_default_pii=False,
        environment=settings.ENVIRONMENT,
        release=settings.VERSION,
    )

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Log configuration and preflight warnings. Database sessions are opened on
    demand; no provider or vector connection is established here.
    """
    # Startup
    logger.info("Starting %s v%s (env=%s)", settings.PROJECT_NAME, settings.VERSION, settings.ENVIRONMENT)
    logger.info("Inference engine: %s", settings.INFERENCE_ENGINE)
    logger.info("Queue backend mode: %s", settings.QUEUE_BACKEND_MODE)
    logger.info("Data directory: %s", settings.data_dir)

    # Preflight checks
    if settings.ENVIRONMENT != "development":
        if not settings.DATABASE_URL:
            logger.error("DATABASE_URL is not set — database operations will fail")
        if not settings.REDIS_URL:
            logger.warning("REDIS_URL is not set — experiments will use inline execution")

    # Warn if HF_TOKEN is missing when HF API inference is expected
    if settings.INFERENCE_ENGINE == "hf_api" and not settings.HF_TOKEN:
        logger.warning(
            "HF_TOKEN is not set but INFERENCE_ENGINE=hf_api. "
            "All inference calls will fail. Set HF_TOKEN in your environment."
        )

    logger.info("CORS allowed origins: %s", settings.cors_origins_list)

    # Startup must not mutate work owned by another API/worker instance.
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
        expose_headers=["X-Request-ID", "Retry-After"],
    )
    
    # Add our custom Request ID middleware
    app.add_middleware(RequestContextMiddleware)
    
    # Register global exception handlers
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)
    
    # Register routers
    app.include_router(sdk_evaluations.router, prefix=f"{settings.API_V1_PREFIX}/sdk/evaluations")
    app.include_router(datasets.router, prefix=f"{settings.API_V1_PREFIX}/datasets", dependencies=[Depends(get_project_context)])
    app.include_router(evaluations.router, prefix=f"{settings.API_V1_PREFIX}/evaluations", dependencies=[Depends(get_project_context)])
    app.include_router(prompt_library.router, prefix=f"{settings.API_V1_PREFIX}/prompt-library", dependencies=[Depends(get_project_context)])
    app.include_router(project_keys.router, prefix=f"{settings.API_V1_PREFIX}/project-keys", dependencies=[Depends(get_project_context)])
    app.include_router(sdk_prompts.router, prefix=f"{settings.API_V1_PREFIX}/sdk/prompts")
    app.include_router(health.router, tags=["Health"])
    app.include_router(
        experiments.router,
        prefix=f"{settings.API_V1_PREFIX}/experiments",
        tags=["Experiments"],
        dependencies=[Depends(get_project_context)],
    )
    app.include_router(
        results.router,
        prefix=f"{settings.API_V1_PREFIX}/results",
        tags=["Results"],
        dependencies=[Depends(get_project_context)],
    )
    app.include_router(
        prompts_api.router,
        prefix=f"{settings.API_V1_PREFIX}/prompts",
        tags=["Prompts"],
        dependencies=[Depends(get_project_context)],
    )
    
    app.include_router(workspaces.router, prefix=f"{settings.API_V1_PREFIX}/workspaces")
    app.include_router(demo.router, prefix=f"{settings.API_V1_PREFIX}/demo", dependencies=[Depends(get_project_context)])
    return app


app = create_application()
