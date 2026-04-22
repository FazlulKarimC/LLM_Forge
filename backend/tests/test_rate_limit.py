from types import SimpleNamespace

from app.core.rate_limit import _get_client_ip


def _make_request(client_host: str, forwarded_for: str | None = None):
    headers = {}
    if forwarded_for is not None:
        headers["X-Forwarded-For"] = forwarded_for
    return SimpleNamespace(
        headers=headers,
        client=SimpleNamespace(host=client_host),
    )


def test_get_client_ip_ignores_forwarded_header_for_direct_clients():
    request = _make_request("8.8.8.8", forwarded_for="1.1.1.1")

    assert _get_client_ip(request) == "8.8.8.8"


def test_get_client_ip_prefers_rightmost_global_forwarded_ip_from_proxy():
    request = _make_request("10.0.0.5", forwarded_for="1.1.1.1, 8.8.8.8")

    assert _get_client_ip(request) == "8.8.8.8"


def test_get_client_ip_falls_back_to_last_forwarded_ip_when_chain_is_private():
    request = _make_request("127.0.0.1", forwarded_for="10.1.1.1, 10.2.2.2")

    assert _get_client_ip(request) == "10.2.2.2"
