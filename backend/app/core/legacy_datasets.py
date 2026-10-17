"""Opt-in compatibility for retired hosted dataset authoring."""
from fastapi import HTTPException, Request
from app.core.config import settings

async def legacy_dataset_write(request: Request):
    if request.method not in ("GET", "HEAD") and not settings.LEGACY_DATASET_WRITES:
        raise HTTPException(410, "Hosted dataset management is retired. Evaluate local cases with the Python SDK.")
