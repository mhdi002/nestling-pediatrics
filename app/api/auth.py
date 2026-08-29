"""
API auth: optional shared API key (programmatic) and optional interactive
username/password login issuing a signed session token (browser).

Both are opt-in. With neither configured the API stays open, which is the
historical behaviour and is fine on a trusted private network. `deploy.sh`
configures login credentials by default so a public deployment is not open.

Deliberately stdlib-only (hashlib/hmac/secrets): password hashing and token
signing here need no third-party dependency, and adding one to the runtime
image for this would be a poor trade.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time

from fastapi import Header, HTTPException, Request

from assistant.settings import get_settings

_PBKDF2_ITERATIONS = 240_000
_SCHEME = "pbkdf2_sha256"

# Fallback signing key when none is configured: sessions then die on restart,
# which is safe (fails closed) rather than using a guessable constant.
_EPHEMERAL_SECRET = secrets.token_hex(32)


def hash_password(password: str, *, salt: bytes | None = None, iterations: int = _PBKDF2_ITERATIONS) -> str:
    """
    Return an encoded PBKDF2 hash: pbkdf2_sha256:iterations:salt:hash.

    Note the ':' separator rather than the conventional '$'. This value lives
    in .env, and Docker Compose performs variable interpolation on those, so a
    '$' would silently truncate the hash to everything before the first '$'.
    """
    if salt is None:
        salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{_SCHEME}:{iterations}:{salt.hex()}:{digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time check of a password against an encoded hash."""
    try:
        # Accept the legacy '$' form too, so an existing .env keeps working.
        parts = encoded.split(":", 3) if ":" in encoded else encoded.split("$", 3)
        scheme, iterations_s, salt_hex, hash_hex = parts
        if scheme != _SCHEME:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations_s)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), hash_hex)


def _signing_secret() -> str:
    return get_settings().nestling_auth_secret or _EPHEMERAL_SECRET


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def make_token(username: str, ttl_hours: int | None = None) -> str:
    """Issue a signed, expiring session token. Stateless: no server-side store."""
    settings = get_settings()
    ttl = ttl_hours if ttl_hours is not None else settings.nestling_session_ttl_hours
    payload = {"sub": username, "exp": int(time.time()) + int(ttl) * 3600}
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(_signing_secret().encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64e(sig)}"


def verify_token(token: str) -> str | None:
    """Return the username for a valid, unexpired token, else None."""
    try:
        body, sig = token.split(".", 1)
        expected = hmac.new(
            _signing_secret().encode("utf-8"), body.encode("ascii"), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(_b64d(sig), expected):
            return None
        payload = json.loads(_b64d(body))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    if int(payload.get("exp", 0)) < time.time():
        return None
    sub = payload.get("sub")
    return sub if isinstance(sub, str) else None


def login_enabled() -> bool:
    """
    Login is active when an env-configured admin exists, or when any account
    has been registered. Checked lazily so a fresh database still allows the
    first sign-up.
    """
    s = get_settings()
    if s.nestling_auth_username and s.nestling_auth_password_hash:
        return True
    try:
        from app.services import get_services

        return get_services().db.count_users() > 0
    except Exception:
        return False


def authenticate(username: str, password: str) -> str | None:
    """
    Return a user identifier on success, else None.

    Registered database accounts take precedence; the env-configured admin is
    kept as a break-glass account that works even with an empty user table.
    """
    try:
        from app.services import get_services

        row = get_services().db.get_user(username or "")
        if row and verify_password(password or "", row.get("password_hash") or ""):
            return str(row.get("user_id"))
    except Exception:
        pass

    s = get_settings()
    if s.nestling_auth_username and s.nestling_auth_password_hash:
        # Constant-time on both fields so neither can be probed by timing.
        user_ok = hmac.compare_digest(username or "", s.nestling_auth_username)
        pass_ok = verify_password(password or "", s.nestling_auth_password_hash)
        if user_ok and pass_ok:
            return f"admin:{s.nestling_auth_username}"
    return None


def _is_open_path(path: str) -> bool:
    """Endpoints reachable without credentials (liveness + sign-in/sign-up)."""
    tail = path.rstrip("/")
    return (
        tail.endswith("/health")
        or tail.endswith("/ready")
        or tail.endswith("/auth/login")
        or tail.endswith("/auth/register")
        or tail.endswith("/auth/config")
    )


def current_user(request: Request) -> str | None:
    """
    User id for the caller, or None for API-key/unauthenticated access.
    Used to scope data so one account cannot read another's children.
    """
    auth_header = request.headers.get("authorization") or ""
    if auth_header.lower().startswith("bearer "):
        return verify_token(auth_header[7:].strip())
    return None


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """
    Gate every /api/* route. Accepts either:
      * X-API-Key / Authorization: Bearer <api key>  -- programmatic clients
      * Authorization: Bearer <session token>        -- browser after login
    When neither an API key nor login credentials are configured, the API is
    open (unchanged historical behaviour).
    """
    settings = get_settings()
    expected_key = settings.nestling_api_key
    if not expected_key and not login_enabled():
        return
    if _is_open_path(request.url.path):
        return

    bearer = None
    auth_header = request.headers.get("authorization") or ""
    if auth_header.lower().startswith("bearer "):
        bearer = auth_header[7:].strip()

    # Constant-time compare so a wrong key cannot be recovered by timing.
    if expected_key:
        for candidate in (x_api_key, bearer):
            if candidate and hmac.compare_digest(candidate, expected_key):
                return

    if bearer and login_enabled() and verify_token(bearer):
        return

    raise HTTPException(
        status_code=401,
        detail={"error": "unauthorized", "detail": "Sign in or supply a valid API key"},
    )
