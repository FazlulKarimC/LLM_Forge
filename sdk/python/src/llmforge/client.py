import math
import os
import time
from collections import OrderedDict
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from .models import Evaluation, Prompt


class LLMForgeError(Exception):
    """Transport, authentication, or server error; never contains request bodies."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class EvaluationTimeout(LLMForgeError):
    def __init__(self, run_id: str):
        super().__init__(f"Evaluation {run_id} exceeded the wait timeout")
        self.run_id = run_id


class LLMForge:
    """Synchronous SDK. Use a context manager to close HTTP resources."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        timeout: float = 10,
        transport: httpx.BaseTransport | None = None,
    ):
        self._key = api_key or os.getenv("LLMFORGE_API_KEY", "")
        if not self._key.strip():
            raise ValueError("Set LLMFORGE_API_KEY or pass api_key")
        url = (
            base_url or os.getenv("LLMFORGE_URL", "http://localhost:8000/api/v1")
        ).rstrip("/")
        parsed = urlparse(url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "base_url must be an HTTP(S) API root without credentials, query or fragment"
            )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be positive")
        self._http = httpx.Client(
            base_url=url + "/",
            headers={"Authorization": "Bearer " + self._key},
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
        )
        self._cache: OrderedDict[str, tuple[str, Prompt]] = OrderedDict()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        self._cache.clear()
        self._http.close()

    def _request(
        self, method, path, *, params=None, body=None, headers=None, allow_304=False
    ):
        try:
            response = self._http.request(
                method, path, params=params, json=body, headers=headers
            )
        except httpx.HTTPError:
            raise LLMForgeError(
                "Could not reach LLMForge. Check its URL and availability; no request was retried."
            ) from None
        if allow_304 and response.status_code == 304:
            return response
        if not 200 <= response.status_code < 300:
            try:
                message = response.json().get("message") or response.json().get(
                    "detail"
                )
            except (ValueError, AttributeError):
                message = None
            message = (
                message if isinstance(message, str) else "LLMForge rejected the request"
            )
            secrets = [self._key]
            if isinstance(body, dict):
                judge = body.get("judge")
                secrets += [
                    body.get("api_key", ""),
                    judge.get("api_key", "") if isinstance(judge, dict) else "",
                ]
            for secret in secrets:
                if isinstance(secret, str) and secret:
                    message = message.replace(secret, "[redacted]")
            raise LLMForgeError(
                f"HTTP {response.status_code}: {message[:1000]}", response.status_code
            )
        return response

    @staticmethod
    def _json(response):
        try:
            return response.json()
        except ValueError:
            raise LLMForgeError("LLMForge returned invalid JSON") from None

    def get_prompt(
        self, name: str, *, label: str | None = None, version: int | None = None
    ) -> Prompt:
        if not name or "/" in name:
            raise ValueError("Prompt name cannot be blank or contain /")
        if label is not None and version is not None:
            raise ValueError("Choose a label or a version, not both")
        if label not in (None, "staging", "production") or (
            version is not None and (type(version) is not int or version < 1)
        ):
            raise ValueError("Use staging/production or a positive version number")
        params = (
            {"version": version}
            if version is not None
            else {"label": label or "production"}
        )
        cache_key = f"{name}|{params}"
        cached = self._cache.get(cache_key)
        try:
            response = self._request(
                "GET",
                "sdk/prompts/" + quote(name, safe=""),
                params=params,
                headers={"If-None-Match": cached[0]} if cached else None,
                allow_304=True,
            )
        except LLMForgeError:
            self._cache.pop(cache_key, None)
            raise
        if response.status_code == 304:
            if cached is None:
                raise LLMForgeError("Unexpected cache response")
            self._cache.move_to_end(cache_key)
            return cached[1]
        data = self._json(response)
        try:
            prompt = Prompt(
                **{
                    key: data[key]
                    for key in (
                        "id",
                        "prompt_id",
                        "name",
                        "version",
                        "template_text",
                        "template_format",
                    )
                },
                variables=tuple(data["variables"]),
            )
        except (KeyError, TypeError):
            raise LLMForgeError(
                "LLMForge returned an invalid prompt snapshot"
            ) from None
        etag = response.headers.get("etag")
        self._cache.pop(cache_key, None)
        if etag:
            self._cache[cache_key] = (etag, prompt)
            self._cache.move_to_end(cache_key)
            if len(self._cache) > 128:
                self._cache.popitem(last=False)
        return prompt

    def get_dataset(self, name: str, *, version: int | None = None) -> dict[str, Any]:
        if (
            not name
            or "/" in name
            or (version is not None and (type(version) is not int or version < 1))
        ):
            raise ValueError("Use a dataset name and optional positive revision number")
        return self._json(
            self._request(
                "GET",
                "sdk/evaluations/datasets/" + quote(name, safe=""),
                params={"version": version} if version is not None else None,
            )
        )

    def start_evaluation(
        self,
        prompt_version_id: str,
        dataset_revision_id: str,
        *,
        assertions: list[dict] | None = None,
        provider: str = "mock",
        model: str = "demo-model",
        api_key: str | None = None,
        temperature: float = 0,
        max_tokens: int = 256,
        judge: dict | None = None,
    ) -> str:
        body = {
            "prompt_version_id": prompt_version_id,
            "dataset_revision_id": dataset_revision_id,
            "assertions": assertions
            if assertions is not None
            else [{"kind": "exact_match"}],
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if api_key:
            body["api_key"] = api_key
        if judge:
            body["judge"] = judge
        return self._json(self._request("POST", "sdk/evaluations/runs", body=body))[
            "id"
        ]

    def get_evaluation(self, run_id: str) -> Evaluation:
        data = self._json(
            self._request("GET", "sdk/evaluations/runs/" + quote(run_id, safe=""))
        )
        try:
            result = Evaluation(run=data["run"], results=data["results"])
            if result.status not in (
                "queued",
                "running",
                "completed",
                "failed",
                "cancelled",
            ) or not isinstance(result.results, list):
                raise ValueError("Invalid evaluation shape")
            return result
        except (KeyError, TypeError, ValueError):
            raise LLMForgeError(
                "LLMForge returned an invalid evaluation response"
            ) from None

    def cancel_evaluation(self, run_id: str) -> dict:
        return self._json(
            self._request(
                "POST", "sdk/evaluations/runs/" + quote(run_id, safe="") + "/cancel"
            )
        )

    def wait_for_evaluation(
        self,
        run_id: str,
        *,
        timeout: float = 180,
        poll_interval: float = 2,
        cancel_on_timeout: bool = False,
    ) -> Evaluation:
        if (
            not math.isfinite(timeout)
            or not math.isfinite(poll_interval)
            or timeout <= 0
            or poll_interval <= 0
        ):
            raise ValueError("Wait timeout and poll interval must be positive")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.get_evaluation(run_id)
            if result.status not in ("queued", "running"):
                return result
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(poll_interval, remaining))
        if cancel_on_timeout:
            try:
                self.cancel_evaluation(run_id)
            except LLMForgeError:
                pass
        raise EvaluationTimeout(run_id)

    def submit_evaluation(
        self,
        prompt_version_id: str,
        dataset_revision_id: str,
        *,
        model: str,
        results: list[dict],
        metrics: dict[str, float] | None = None,
    ) -> str:
        """Persist application-computed outputs/checks; never calls a provider."""
        body = {
            "prompt_version_id": prompt_version_id,
            "dataset_revision_id": dataset_revision_id,
            "model": model,
            "results": results,
            "metrics": metrics or {},
        }
        return self._json(
            self._request("POST", "sdk/evaluations/submissions", body=body)
        )["id"]
