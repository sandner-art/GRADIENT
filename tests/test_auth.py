# tests/test_auth.py
"""Optional API token: needed from other computers when set, never from this one."""

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from gradient.api import auth
from gradient.api.main import app

TOKEN = "s3cret token;with=odd chars"


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setattr(auth, "api_token", lambda: TOKEN)
    return TOKEN


def remote(**kw):
    return TestClient(app, client=("192.168.1.20", 50000), **kw)


def local(**kw):
    return TestClient(app, client=("127.0.0.1", 50000), **kw)


def test_no_token_allows_everyone():
    with remote() as c:
        assert c.get("/api/v1/status").status_code == 200


def test_remote_needs_the_token(token):
    with remote() as c:
        r = c.get("/api/v1/status")
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
        assert "Authorization: Bearer" in r.json()["detail"]
        assert c.get("/api/v1/status", headers={"Authorization": "Bearer nope"}).status_code == 401
        ok = c.get("/api/v1/status", headers={"Authorization": f"Bearer {token}"})
        assert ok.status_code == 200 and ok.json()["status"]
        page = c.get("/some/page")
        assert page.status_code == 401 and "?token=" in page.text


def test_this_computer_never_needs_it(token):
    with local() as c:
        assert c.get("/api/v1/status").status_code == 200
        # through a proxy on this computer it counts as remote
        r = c.get("/api/v1/status", headers={"X-Forwarded-For": "10.0.0.5"})
        assert r.status_code == 401


def test_browser_opens_with_token_once(token):
    with remote(follow_redirects=False) as c:
        r = c.get("/gallery", params={"token": token, "tab": "runs"})
        assert r.status_code == 303 and r.headers["location"] == "/gallery?tab=runs"
        cookie = r.headers["set-cookie"]
        assert "HttpOnly" in cookie and "SameSite=Strict" in cookie and token not in cookie
        assert c.cookies.get(auth.COOKIE) == auth.cookie_value(token)
        assert c.get("/api/v1/status").status_code == 200  # the cookie is sent from now on
        c.cookies.clear()
        assert c.get("/api/v1/status").status_code == 401
        wrong = c.get("/gallery", params={"token": "guess"})
        assert wrong.status_code == 401 and "set-cookie" not in wrong.headers


def test_websocket_needs_the_token(token):
    with remote() as c:
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("/ws") as ws:
                ws.receive_text()
    with local() as c:
        with c.websocket_connect("/ws"):
            pass


@pytest.mark.parametrize(
    "host, loopback",
    [("127.0.0.1", True), ("::1", True), ("localhost", True), ("0.0.0.0", False), ("", False)],
)
def test_is_loopback(host, loopback):
    assert auth.is_loopback(host) is loopback


def test_setting_from_environment(monkeypatch):
    from gradient.core import config

    monkeypatch.setenv("GRADIENT_API_TOKEN", "from-env")
    config.load_settings()
    try:
        assert auth.api_token() == "from-env"
    finally:
        monkeypatch.delenv("GRADIENT_API_TOKEN")
        config.load_settings()
    assert auth.api_token() == ""
