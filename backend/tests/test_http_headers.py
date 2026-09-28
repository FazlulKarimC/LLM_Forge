"""The shared error envelope must retain operational HTTP headers."""

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import create_application


def test_auth_and_rate_limit_headers_survive_error_handling():
    app = create_application()

    @app.get("/rate-limit-probe")
    def rate_limit_probe():
        raise HTTPException(429, "Try later", headers={"Retry-After": "30"})

    with TestClient(app) as client:
        auth = client.get("/api/v1/workspaces")
        assert auth.status_code == 401
        assert auth.headers["WWW-Authenticate"] == "Bearer"

        limited = client.get(
            "/rate-limit-probe", headers={"Origin": "http://localhost:3000"}
        )
        assert limited.status_code == 429
        assert limited.headers["Retry-After"] == "30"
        exposed = limited.headers["Access-Control-Expose-Headers"].lower()
        assert "retry-after" in exposed
        assert "x-request-id" in exposed
