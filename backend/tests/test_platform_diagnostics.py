import json
import logging

from app.core.diagnostics import PlatformFormatter, context, sanitize_sentry_event


def test_platform_logs_discard_secret_context_and_exception_values():
    token = context.set(
        {"run_id": "run", "api_key": "secret", "prompt": "private input"}
    )
    try:
        record = logging.LogRecord(
            "app.platform", logging.ERROR, __file__, 1, "stage_failed", (), None
        )
        record.output = "private output"
        record.exception_type = "RuntimeError"
        payload = json.loads(PlatformFormatter().format(record))
        assert payload["run_id"] == "run"
        assert payload["exception_type"] == "RuntimeError"
        assert "secret" not in json.dumps(payload)
        assert "private" not in json.dumps(payload)
    finally:
        context.reset(token)
    result = sanitize_sentry_event(
        {
            "request": {"data": "secret"},
            "breadcrumbs": ["private"],
            "exception": {
                "values": [
                    {
                        "type": "RuntimeError",
                        "value": "secret",
                        "stacktrace": {"frames": [{"vars": {"api_key": "secret"}}]},
                    }
                ]
            },
        },
        {},
    )
    assert result == {"exception": {"values": [{"type": "RuntimeError"}]}, "tags": {}}
