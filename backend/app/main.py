"""
FastAPI Application Entry Point

This module initializes the FastAPI application with:
- CORS middleware for frontend communication
- API routers for experiments, results, and metrics
- Lifespan events for startup/shutdown
- Health check endpoint

TODO (Iteration 1): Add database connection on startup
TODO (Iteration 2): Add model preloading for faster inference
TODO (Iteration 3): Add request tracing and monitoring
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api import experiments, results, health


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
    print(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    # TODO: Initialize services here
    
    yield
    
    # Shutdown
    print("Shutting down...")
    # TODO: Cleanup resources here


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
        lifespan=lifespan,
    )
    
    # Configure CORS
    # TODO (Iteration 1): Restrict origins in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
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
