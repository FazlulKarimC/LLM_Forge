"""Shared workspace setup helpers; no tests or fixtures are imported here."""

from app.core.auth import Identity, get_current_user


def login(app, subject):
    app.dependency_overrides[get_current_user] = lambda: Identity(
        subject, subject.removeprefix("user_")
    )


async def bootstrap(app, client, subject):
    login(app, subject)
    response = await client.get("/api/v1/workspaces")
    assert response.status_code == 200, response.text
    org = response.json()["organizations"][0]
    return org, {"X-Project-ID": org["projects"][0]["id"]}
