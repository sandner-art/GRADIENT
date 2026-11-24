# src/gradient/api/auth.py
"""Optional API token for clients on other machines.

With `app.api_token` (or GRADIENT_API_TOKEN) set, requests from other machines need it:
`Authorization: Bearer <token>` for API clients, or, for the web UI in a browser, the
cookie that opening any page with `?token=<token>` sets once. Requests from this computer
never need it, so the local web UI and `gradient stop` keep working. A request that came
through a proxy (X-Forwarded-For, Forwarded or X-Real-IP header) counts as remote, so a
reverse proxy on this computer does not open GRADIENT to everyone.

Without a token every request is allowed, as before (GRADIENT listens on 127.0.0.1 by
default; the console warns when it listens on other addresses without a token).
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
from http.cookies import SimpleCookie
from urllib.parse import parse_qsl, urlencode

from gradient.core.config import get_setting

COOKIE = "gradient_token"
COOKIE_MAX_AGE = 30 * 24 * 3600
PROXY_HEADERS = (b"x-forwarded-for", b"forwarded", b"x-real-ip")


def api_token() -> str:
    return str(get_setting("app.api_token", "") or "").strip()


def is_loopback(host: str | None) -> bool:
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def _headers(scope) -> dict[bytes, bytes]:
    return {k.lower(): v for k, v in scope.get("headers") or []}


def _is_local(scope, headers: dict[bytes, bytes]) -> bool:
    client = scope.get("client")
    if not client or not is_loopback(client[0]):
        return False
    return not any(h in headers for h in PROXY_HEADERS)


def cookie_value(token: str) -> str:
    """The cookie holds a value derived from the token, not the token itself."""
    return hmac.new(token.encode(), b"gradient-web-ui", hashlib.sha256).hexdigest()


def _same(a: str, b: str) -> bool:
    return bool(a) and hmac.compare_digest(a.encode(), b.encode())


def _authorized(headers: dict[bytes, bytes], token: str) -> bool:
    auth = headers.get(b"authorization", b"").decode("latin-1")
    if auth[:7].lower() == "bearer " and _same(auth[7:].strip(), token):
        return True
    cookie = SimpleCookie()
    try:
        cookie.load(headers.get(b"cookie", b"").decode("latin-1"))
    except Exception:
        return False
    return COOKIE in cookie and _same(cookie[COOKIE].value, cookie_value(token))


class TokenMiddleware:
    """ASGI middleware (HTTP and WebSocket) that enforces `app.api_token` for remote clients."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        token = api_token()
        headers = _headers(scope)
        if not token or _is_local(scope, headers) or _authorized(headers, token):
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return None
        query = parse_qsl(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
        given = next((v for k, v in query if k == "token"), "")
        if scope.get("method") == "GET" and _same(given, token):
            # a browser opened GRADIENT with ?token=...: keep it in a cookie, drop it from the URL
            rest = urlencode([(k, v) for k, v in query if k != "token"])
            path = scope.get("raw_path") or scope["path"].encode("utf-8")
            location = path + (f"?{rest}".encode("latin-1") if rest else b"")
            secure = "; Secure" if scope.get("scheme") == "https" else ""
            cookie = (
                f"{COOKIE}={cookie_value(token)}; Path=/; Max-Age={COOKIE_MAX_AGE}; HttpOnly; "
                "SameSite=Strict" + secure
            )
            await send(
                {
                    "type": "http.response.start",
                    "status": 303,
                    "headers": [
                        (b"location", location),
                        (b"set-cookie", cookie.encode("latin-1")),
                        (b"cache-control", b"no-store"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return None
        if scope["path"].startswith("/api/"):
            body = json.dumps(
                {
                    "detail": "This GRADIENT needs an API token from other computers: send "
                    "'Authorization: Bearer <token>'."
                }
            ).encode()
            content_type = b"application/json"
        else:
            body = (
                b"This GRADIENT needs an API token from other computers. "
                b"Open it once as http://<address>/?token=<token>."
            )
            content_type = b"text/plain; charset=utf-8"
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", content_type),
                    (b"www-authenticate", b"Bearer"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
        return None
