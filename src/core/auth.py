"""Authentication middleware for FinAI FastAPI service.

LibreChat authenticates users with JWT tokens (HS256, ``JWT_SECRET``).
When proxying requests to FinAI, LibreChat forwards a short-lived
verification token in the ``X-LibreChat-Token`` header. This middleware
validates that token and extracts the authenticated ``user_id`` from
the JWT payload (``{ "id": "<mongo_objectid>" }``).

Two modes:

* **Enforced** (``FINAI_AUTH_ENABLED=true``) — every request must carry
  a valid token; unauthenticated requests get ``401``.
* **Passthrough** (default, for local dev / demo) — missing or invalid
  tokens are logged but the request proceeds with ``user_id`` set to
  whatever the client sent (or ``"anonymous"``). This preserves the
  current behaviour for curl / OpenAI playground / direct API tests.

The ``JWT_SECRET`` **must** match LibreChat's ``JWT_SECRET`` env var
for token validation to work. In Docker, both containers share the same
``.env`` file so this is automatic.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import jwt

log = logging.getLogger("finai.auth")

_JWT_ALGORITHM = "HS256"
_HEADER_NAME = "x-librechat-token"
_BEARER_PREFIX = "Bearer "


@dataclass
class AuthResult:
    user_id: str
    authenticated: bool
    error: Optional[str] = None


def get_jwt_secret() -> Optional[str]:
    """Return the JWT secret from the environment, or ``None`` if unset."""
    return os.environ.get("JWT_SECRET") or os.environ.get("LIBRECHAT_JWT_SECRET")


def is_auth_enabled() -> bool:
    """Check if auth enforcement is turned on.

    Controlled by ``FINAI_AUTH_ENABLED`` env var (``true``/``1``/``yes``).
    Default is **off** (passthrough mode) to preserve the current demo
    behaviour.
    """
    val = os.environ.get("FINAI_AUTH_ENABLED", "").lower()
    return val in ("true", "1", "yes")


def verify_token(token: str, secret: str) -> AuthResult:
    """Verify a JWT token and return the authenticated user ID.

    Returns :class:`AuthResult` with ``authenticated=True`` on success,
    or ``authenticated=False`` with an error message on failure.
    """
    try:
        payload = jwt.decode(token, secret, algorithms=[_JWT_ALGORITHM])
        user_id = payload.get("id")
        if not user_id or not isinstance(user_id, str):
            return AuthResult(
                user_id="anonymous",
                authenticated=False,
                error="JWT payload missing 'id' field",
            )
        return AuthResult(user_id=user_id, authenticated=True)
    except jwt.ExpiredSignatureError:
        return AuthResult(
            user_id="anonymous",
            authenticated=False,
            error="Token expired",
        )
    except jwt.InvalidTokenError as exc:
        return AuthResult(
            user_id="anonymous",
            authenticated=False,
            error=f"Invalid token: {exc}",
        )


def authenticate_request(
    headers: dict[str, str],
    fallback_user: Optional[str] = None,
) -> AuthResult:
    """Authenticate an incoming request using the LibreChat token header.

    Parameters
    ----------
    headers:
        The incoming request headers (case-insensitive lookup expected).
    fallback_user:
        The ``user`` field from the OpenAI request body, used as a
        fallback identity when auth is in passthrough mode.

    Returns
    -------
    :class:`AuthResult` — always populated; ``authenticated`` flag
    indicates whether the user was verified via JWT.
    """
    secret = get_jwt_secret()
    enabled = is_auth_enabled()

    if not secret:
        if enabled:
            log.error("FINAI_AUTH_ENABLED=true but JWT_SECRET is not set")
            return AuthResult(
                user_id=fallback_user or "anonymous",
                authenticated=False,
                error="JWT_SECRET not configured",
            )
        log.debug("No JWT_SECRET configured; running in passthrough mode")
        return AuthResult(
            user_id=fallback_user or "anonymous",
            authenticated=False,
        )

    token = _extract_token(headers)
    if token is None:
        if enabled:
            return AuthResult(
                user_id="anonymous",
                authenticated=False,
                error="Missing X-LibreChat-Token header",
            )
        log.debug("No auth token; passthrough with fallback user")
        return AuthResult(
            user_id=fallback_user or "anonymous",
            authenticated=False,
        )

    result = verify_token(token, secret)

    if not result.authenticated:
        if enabled:
            log.warning("Auth failed (enforced): %s", result.error)
            return result
        log.debug("Auth failed (passthrough): %s — using fallback user", result.error)
        return AuthResult(
            user_id=fallback_user or "anonymous",
            authenticated=False,
            error=result.error,
        )

    log.info("Authenticated user: %s", result.user_id)
    return result


def _extract_token(headers: dict[str, str]) -> Optional[str]:
    """Extract the JWT token from request headers.

    Checks in order:
    1. ``X-LibreChat-Token`` header (direct token)
    2. ``Authorization: Bearer <token>`` header (standard)
    """
    for key, value in headers.items():
        if key.lower() == _HEADER_NAME:
            return value.strip()

    for key, value in headers.items():
        if key.lower() == "authorization":
            if value.startswith(_BEARER_PREFIX):
                return value[len(_BEARER_PREFIX):].strip()

    return None


__all__ = [
    "AuthResult",
    "authenticate_request",
    "get_jwt_secret",
    "is_auth_enabled",
    "verify_token",
]
