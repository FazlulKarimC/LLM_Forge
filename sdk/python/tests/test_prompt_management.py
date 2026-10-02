import json
import httpx
import pytest


def test_chat_snapshot_compile_and_metadata(snapshot, sdk_client):
    data = {**snapshot, "name": "support/chat", "template_text": "", "prompt_type": "chat", "config": {"temperature": 0}, "tags": ["support"], "labels": ["latest"], "messages": [{"role": "system", "content": "Use {{language}}"}, {"role": "user", "content": "{{query}}"}]}
    with sdk_client(lambda request: httpx.Response(200, json=data)) as forge:
        prompt = forge.get_prompt("support/chat", label="latest")
        assert prompt.config == {"temperature": 0} and prompt.tags == ("support",)
        assert prompt.labels == ("latest",)
        assert prompt.compile(language="English", query="{{literal}}") == [{"role": "system", "content": "Use English"}, {"role": "user", "content": "{{literal}}"}]
        with pytest.raises(ValueError, match="Missing variables"):
            prompt.compile(query="hi")


def test_create_and_label_requests_do_not_retry(snapshot, sdk_client):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(201, json=snapshot)
    with sdk_client(handler) as forge:
        forge.create_prompt("support/answer", "{{query}}", labels=["experiment-a"], config={"temperature": 0}, tags=["support"], base_version=1)
        payload = json.loads(requests[0].content)
        assert requests[0].method == "POST" and payload["base_version"] == 1
        assert payload["labels"] == ["experiment-a"] and payload["prompt_type"] == "text"
        assert payload["config"] == {"temperature": 0}
        forge.set_prompt_label("support/answer", 1, "production")
        assert requests[1].method == "PUT" and json.loads(requests[1].content) == {"version": 1}
        assert requests[1].url.path.endswith("support/answer/labels/production")
        with pytest.raises(ValueError):
            forge.set_prompt_label("support/answer", 1, "latest")
        assert len(requests) == 2


def test_custom_selector_and_list_filters(snapshot, sdk_client):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=snapshot if len(requests) == 1 else {"items": [], "total": 0})
    with sdk_client(handler) as forge:
        forge.get_prompt("support/answer", label="experiment-a")
        assert requests[0].url.params["label"] == "experiment-a"
        assert forge.list_prompts(tag="support", label="production", limit=10) == {"items": [], "total": 0}
        assert requests[1].url.params["tag"] == "support" and requests[1].url.params["limit"] == "10"
