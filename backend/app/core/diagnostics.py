"""Allow-listed platform diagnostics, distinct from application telemetry."""

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone

context = ContextVar("platform_diagnostic_context", default={})
FIELDS = {
    "request_id",
    "project_id",
    "run_id",
    "attempt",
    "stage",
    "duration_ms",
    "outcome",
    "status_code",
    "exception_type",
}


class PlatformFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.msg,
            **{key: value for key, value in context.get().items() if key in FIELDS},
        }
        payload.update(
            {
                key: str(value) if key.endswith("_id") else value
                for key, value in record.__dict__.items()
                if key in FIELDS
            }
        )
        return json.dumps(payload, default=str)


logger = logging.getLogger("app.platform")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(PlatformFormatter())
    logger.addHandler(handler)
    logger.propagate = False
logger.setLevel(logging.INFO)


def sanitize_sentry_event(event, hint):
    """Keep operational types/identifiers; discard bodies, values and breadcrumbs."""
    values = event.get("exception", {}).get("values", [])
    safe = {
        key: event[key]
        for key in (
            "event_id",
            "timestamp",
            "level",
            "platform",
            "environment",
            "release",
        )
        if key in event
    }
    if values:
        safe["exception"] = {
            "values": [{"type": item.get("type", "Exception")} for item in values]
        }
    safe["tags"] = {
        key: value for key, value in event.get("tags", {}).items() if key in FIELDS
    }
    return safe
