"""Aggregate results API router.

The concrete route groups live in sibling modules so each file has one job:
comparison, read/export, background jobs, regression, and routing telemetry.
This module preserves the existing ``app.api.results.router`` import contract.
"""

from fastapi import APIRouter

from app.api import (
    results_compare,
    results_jobs,
    results_read,
    results_regression,
    results_routing,
)
from app.api.results_common import result_to_metrics_response as _result_to_metrics_response

router = APIRouter()

# Static routes are included before parameterized ``/{experiment_id}`` routes.
router.include_router(results_compare.router)
router.include_router(results_jobs.router)
router.include_router(results_read.router)
router.include_router(results_regression.router)
router.include_router(results_routing.router)
