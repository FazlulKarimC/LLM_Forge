from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.custom_exceptions import AppException

from app.core.diagnostics import logger


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """
    Handles our custom application exceptions.
    Returns structured JSON corresponding to the custom exception details.
    """
    request_id = getattr(request.state, "request_id", None)

    logger.warning(
        "application_request_failed",
        extra={
            "request_id": request_id,
            "status_code": exc.status_code,
            "exception_type": type(exc).__name__,
        },
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.message,
            "status_code": exc.status_code,
            "request_id": request_id,
        },
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Overrides the default FastAPI 422 Unprocessable Entity handler.
    Formats Pydantic validation errors into cleaner, user-friendly messages.
    """
    request_id = getattr(request.state, "request_id", None)

    # Extract details from Pydantic errors to create a human-readable message
    errors = exc.errors()
    clean_errors = []

    for err in errors:
        loc = " -> ".join(str(part) for part in err.get("loc", []))
        msg = err.get("msg", "")
        clean_errors.append(f"{loc}: {msg}")

    combined_message = "Validation Error: " + "; ".join(clean_errors)

    logger.warning(
        "request_validation_failed",
        extra={"request_id": request_id, "status_code": 422},
    )

    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "message": combined_message,
            "status_code": 422,
            "request_id": request_id,
            "details": [
                {
                    "field": ".".join(str(part) for part in err.get("loc", [])),
                    "issue": err.get("msg", ""),
                }
                for err in errors
            ],
        },
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all for any unhandled Python exceptions (500 Internal Server Error).
    Emits identifiers and exception type without arbitrary exception context.
    """
    request_id = getattr(request.state, "request_id", None)

    logger.error(
        "unhandled_request_failed",
        extra={
            "request_id": request_id,
            "status_code": 500,
            "exception_type": type(exc).__name__,
        },
    )

    # Return a generic, safe response to the client
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "An unexpected internal server error occurred.",
            "status_code": 500,
            "request_id": request_id,
        },
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """
    Handles FastAPI/Starlette HTTPExceptions (e.g., 404, 405, 422).
    Ensures they return our standardized JSON error format.
    """
    request_id = getattr(request.state, "request_id", None)

    return JSONResponse(
        status_code=exc.status_code,
        headers=exc.headers,
        content={
            "error": True,
            "message": exc.detail or f"HTTP {exc.status_code}",
            "status_code": exc.status_code,
            "request_id": request_id,
        },
    )
