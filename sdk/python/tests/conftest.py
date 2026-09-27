"""Fresh SDK response payloads and a credential-isolated mock client factory."""

import httpx
import pytest
from llmforge import LLMForge


@pytest.fixture
def sdk_client():
    return lambda handler: LLMForge(
        api_key="lf_live_SECRET",
        base_url="https://forge.example/api/v1",
        transport=httpx.MockTransport(handler),
    )


@pytest.fixture
def snapshot():
    return {
        "id": "version-id",
        "prompt_id": "prompt-id",
        "name": "Echo demo",
        "version": 1,
        "template_text": "{{query}}",
        "template_format": "mustache",
        "variables": ["query"],
    }


@pytest.fixture
def completed_run():
    return {
        "id": "run-id",
        "status": "completed",
        "total": 2,
        "completed": 2,
        "passed": 2,
        "errors": 0,
    }
