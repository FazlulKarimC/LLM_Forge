"""
Tests for the task dispatch abstraction and Upstash circuit breaker.

Covers:
- Auto dispatch uses RQ when Upstash and worker are healthy
- Missing REDIS_URL falls back to inline
- Missing worker heartbeat falls back to inline
- Connection/timeout/auth errors open the circuit for 30 minutes
- Archived-instance-style errors open the circuit
- Open circuit skips Upstash probe
- Half-open success closes circuit
- Half-open failure reopens circuit
"""

import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from uuid import uuid4

import pytest

from app.core.upstash_circuit import (
    _circuit,
    classify_upstash_failure,
    get_circuit_snapshot,
    is_open,
    allow_probe,
    record_failure,
    record_success,
    reset_circuit,
)
from app.core.task_dispatch import (
    AutoDispatchBackend,
    InlineDispatchBackend,
    UpstashRQDispatchBackend,
    dispatch_experiment,
    DispatchResult,
)


# ── Helpers ─────────────────────────────────────────────────────────────


def _run(coro):
    """Run an async coroutine in a fresh event loop for sync tests."""
    return asyncio.run(coro)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_circuit():
    """Reset the circuit breaker before every test."""
    reset_circuit()
    yield
    reset_circuit()


@pytest.fixture
def mock_background_tasks():
    return MagicMock()


# ── Circuit Breaker Unit Tests ──────────────────────────────────────────


class TestCircuitBreaker:
    """Tests for upstash_circuit.py"""

    def test_starts_closed(self):
        assert _circuit.state == "closed"
        assert not is_open()

    def test_record_success_keeps_closed(self):
        record_success()
        assert _circuit.state == "closed"
        assert not is_open()

    def test_connection_error_opens_circuit(self):
        from redis.exceptions import ConnectionError as RedisConnectionError

        exc = RedisConnectionError("Connection refused")
        record_failure(exc)
        assert _circuit.state == "open"
        assert is_open()
        assert "ConnectionError" in _circuit.last_failure_reason

    def test_timeout_error_opens_circuit(self):
        from redis.exceptions import TimeoutError as RedisTimeoutError

        exc = RedisTimeoutError("Read timed out")
        record_failure(exc)
        assert _circuit.state == "open"
        assert is_open()

    def test_auth_error_opens_circuit(self):
        from redis.exceptions import AuthenticationError as RedisAuthError

        exc = RedisAuthError("WRONGPASS invalid password")
        record_failure(exc)
        assert _circuit.state == "open"

    def test_archived_instance_error_opens_circuit(self):
        from redis.exceptions import ResponseError as RedisResponseError

        exc = RedisResponseError("NOPERM this instance has been archived")
        record_failure(exc)
        assert _circuit.state == "open"
        assert "archived" in _circuit.last_failure_reason.lower()

    def test_not_found_error_opens_circuit(self):
        from redis.exceptions import ResponseError as RedisResponseError

        exc = RedisResponseError("ERR instance not found")
        record_failure(exc)
        assert _circuit.state == "open"

    def test_gone_error_opens_circuit(self):
        from redis.exceptions import ResponseError as RedisResponseError

        exc = RedisResponseError("ERR this instance is gone")
        record_failure(exc)
        assert _circuit.state == "open"

    def test_noauth_error_opens_circuit(self):
        from redis.exceptions import ResponseError as RedisResponseError

        exc = RedisResponseError("NOAUTH Authentication required")
        record_failure(exc)
        assert _circuit.state == "open"

    def test_non_availability_response_error_does_not_trip_circuit(self):
        """Application bugs (e.g., bad EVALSHA) should NOT open the circuit."""
        from redis.exceptions import ResponseError as RedisResponseError

        exc = RedisResponseError("NOSCRIPT No matching script found")
        record_failure(exc)
        assert _circuit.state == "closed"

    def test_application_exception_does_not_trip_circuit(self):
        exc = ValueError("bad argument format")
        record_failure(exc)
        assert _circuit.state == "closed"

    def test_open_circuit_blocks_for_30_minutes(self):
        from redis.exceptions import ConnectionError as RedisConnectionError

        record_failure(RedisConnectionError("boom"))
        assert is_open()

        # Simulate time passage just short of 30 minutes
        _circuit.reopen_at = time.monotonic() + 10  # still in the future
        assert is_open()

    def test_open_circuit_transitions_to_half_open_after_timeout(self):
        from redis.exceptions import ConnectionError as RedisConnectionError

        record_failure(RedisConnectionError("boom"))
        # Simulate timeout expired
        _circuit.reopen_at = time.monotonic() - 1
        assert not is_open()  # should transition to half_open
        assert _circuit.state == "half_open"

    def test_half_open_success_closes_circuit(self):
        _circuit.state = "half_open"
        record_success()
        assert _circuit.state == "closed"

    def test_half_open_failure_reopens_circuit(self):
        from redis.exceptions import TimeoutError as RedisTimeoutError

        _circuit.state = "half_open"
        record_failure(RedisTimeoutError("timeout again"))
        assert _circuit.state == "open"

    def test_open_circuit_skips_probe(self):
        from redis.exceptions import ConnectionError as RedisConnectionError

        record_failure(RedisConnectionError("boom"))
        assert not allow_probe()

    def test_half_open_allows_probe(self):
        _circuit.state = "half_open"
        assert allow_probe()

    def test_get_circuit_snapshot(self):
        snapshot = get_circuit_snapshot()
        assert snapshot["state"] == "closed"
        assert snapshot["last_failure_reason"] is None

    def test_classify_dns_failure(self):
        exc = OSError("Name or service not known")
        reason = classify_upstash_failure(exc)
        assert reason is not None
        assert "OSError" in reason


# ── Failure Classification ──────────────────────────────────────────────


class TestFailureClassification:
    """Focused tests for classify_upstash_failure()"""

    def test_connection_error(self):
        from redis.exceptions import ConnectionError as RedisConnectionError
        assert classify_upstash_failure(RedisConnectionError("x")) is not None

    def test_timeout_error(self):
        from redis.exceptions import TimeoutError as RedisTimeoutError
        assert classify_upstash_failure(RedisTimeoutError("x")) is not None

    def test_auth_error(self):
        from redis.exceptions import AuthenticationError as RedisAuthError
        assert classify_upstash_failure(RedisAuthError("x")) is not None

    def test_response_error_archived(self):
        from redis.exceptions import ResponseError
        assert classify_upstash_failure(ResponseError("archived")) is not None

    def test_response_error_wrongpass(self):
        from redis.exceptions import ResponseError
        assert classify_upstash_failure(ResponseError("WRONGPASS")) is not None

    def test_response_error_app_bug_returns_none(self):
        from redis.exceptions import ResponseError
        assert classify_upstash_failure(ResponseError("NOSCRIPT foo")) is None

    def test_value_error_returns_none(self):
        assert classify_upstash_failure(ValueError("bad arg")) is None

    def test_type_error_returns_none(self):
        assert classify_upstash_failure(TypeError("wrong type")) is None

    def test_os_error(self):
        assert classify_upstash_failure(OSError("DNS failed")) is not None


# ── Dispatch Integration Tests ──────────────────────────────────────────


class TestAutoDispatch:
    """Tests for AutoDispatchBackend decision tree."""

    def _make_backend(self):
        return AutoDispatchBackend()

    @patch("app.core.task_dispatch.settings")
    def test_missing_redis_url_falls_back_inline(self, mock_settings, mock_background_tasks):
        mock_settings.REDIS_URL = ""
        mock_settings.QUEUE_BACKEND_MODE = "auto"

        backend = self._make_backend()
        result = _run(backend.dispatch(mock_background_tasks, uuid4()))

        assert result.backend_used == "inline"
        assert "not configured" in result.fallback_reason.lower()

    @patch("app.core.task_dispatch._check_worker_heartbeat_async", new_callable=AsyncMock, return_value=False)
    @patch("app.core.redis.probe_redis")
    @patch("app.core.task_dispatch.settings")
    def test_missing_worker_heartbeat_falls_back_inline(
        self, mock_settings, mock_probe, mock_heartbeat, mock_background_tasks
    ):
        from app.core.redis import RedisProbeResult

        mock_settings.REDIS_URL = "rediss://test"
        mock_settings.QUEUE_BACKEND_MODE = "auto"
        mock_settings.UPSTASH_HEALTHCHECK_CACHE_SECONDS = 60
        mock_settings.UPSTASH_CIRCUIT_OPEN_MINUTES = 30
        mock_probe.return_value = RedisProbeResult(healthy=True, latency_ms=5.0)

        backend = self._make_backend()
        result = _run(backend.dispatch(mock_background_tasks, uuid4()))

        assert result.backend_used == "inline"
        assert result.worker_available is False
        assert "heartbeat" in result.fallback_reason.lower()

    @patch("app.core.task_dispatch._check_worker_heartbeat_async", new_callable=AsyncMock, return_value=True)
    @patch("app.core.redis.probe_redis")
    @patch("app.core.task_dispatch.settings")
    def test_auto_dispatch_uses_rq_when_healthy(
        self, mock_settings, mock_probe, mock_heartbeat, mock_background_tasks
    ):
        from app.core.redis import RedisProbeResult

        mock_settings.REDIS_URL = "rediss://test"
        mock_settings.QUEUE_BACKEND_MODE = "auto"
        mock_settings.UPSTASH_HEALTHCHECK_CACHE_SECONDS = 60
        mock_settings.UPSTASH_CIRCUIT_OPEN_MINUTES = 30
        mock_probe.return_value = RedisProbeResult(healthy=True, latency_ms=5.0)

        # Mock the actual RQ enqueue
        with patch("app.core.task_dispatch.UpstashRQDispatchBackend.dispatch", new_callable=AsyncMock) as mock_rq:
            mock_rq.return_value = DispatchResult(backend_used="rq")
            backend = self._make_backend()
            result = _run(backend.dispatch(mock_background_tasks, uuid4()))

        assert result.backend_used == "rq"
        assert result.worker_available is True

    @patch("app.core.task_dispatch.settings")
    def test_open_circuit_skips_upstash_probe(self, mock_settings, mock_background_tasks):
        from redis.exceptions import ConnectionError as RedisConnectionError

        mock_settings.REDIS_URL = "rediss://test"
        mock_settings.QUEUE_BACKEND_MODE = "auto"
        mock_settings.UPSTASH_CIRCUIT_OPEN_MINUTES = 30
        mock_settings.UPSTASH_HEALTHCHECK_CACHE_SECONDS = 60

        # Open the circuit
        record_failure(RedisConnectionError("dead"))
        assert is_open()

        backend = self._make_backend()
        result = _run(backend.dispatch(mock_background_tasks, uuid4()))

        assert result.backend_used == "inline"
        assert result.circuit_state == "open"

    @patch("app.core.redis.probe_redis")
    @patch("app.core.task_dispatch.settings")
    def test_probe_failure_opens_circuit_and_goes_inline(
        self, mock_settings, mock_probe, mock_background_tasks
    ):
        from app.core.redis import RedisProbeResult

        mock_settings.REDIS_URL = "rediss://test"
        mock_settings.QUEUE_BACKEND_MODE = "auto"
        mock_settings.UPSTASH_CIRCUIT_OPEN_MINUTES = 30
        mock_settings.UPSTASH_HEALTHCHECK_CACHE_SECONDS = 60

        mock_probe.return_value = RedisProbeResult(
            healthy=False, error="Connection refused"
        )

        backend = self._make_backend()
        result = _run(backend.dispatch(mock_background_tasks, uuid4()))

        assert result.backend_used == "inline"
        assert result.circuit_state == "open"
        assert "probe failed" in result.fallback_reason.lower()


# ── Inline Backend Tests ────────────────────────────────────────────────


class TestInlineDispatch:
    def test_inline_dispatches_to_background_tasks(self, mock_background_tasks):
        backend = InlineDispatchBackend()
        result = _run(backend.dispatch(mock_background_tasks, uuid4()))
        assert result.backend_used == "inline"
        mock_background_tasks.add_task.assert_called_once()


# ── Top-Level dispatch_experiment Tests ─────────────────────────────────


class TestDispatchExperiment:

    @patch("app.core.task_dispatch.settings")
    def test_inline_mode_always_inline(self, mock_settings, mock_background_tasks):
        mock_settings.QUEUE_BACKEND_MODE = "inline"

        result = _run(dispatch_experiment(mock_background_tasks, uuid4()))
        assert result.backend_used == "inline"

    @patch("app.core.task_dispatch.settings")
    def test_auto_mode_no_redis(self, mock_settings, mock_background_tasks):
        mock_settings.QUEUE_BACKEND_MODE = "auto"
        mock_settings.REDIS_URL = ""

        result = _run(dispatch_experiment(mock_background_tasks, uuid4()))
        assert result.backend_used == "inline"
