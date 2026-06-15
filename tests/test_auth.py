"""Tests for the authentication middleware (:mod:`src.core.auth`).

Covers:
- JWT token verification (valid, expired, malformed, missing ``id``)
- Passthrough mode (default, ``FINAI_AUTH_ENABLED`` unset)
- Enforced mode (``FINAI_AUTH_ENABLED=true``)
- Header extraction (``X-LibreChat-Token``, ``Authorization: Bearer``)
- Fallback user resolution
"""

from __future__ import annotations

import os
import time
from unittest.mock import patch

import jwt as pyjwt
import pytest

from src.core.auth import (
    AuthResult,
    authenticate_request,
    get_jwt_secret,
    is_auth_enabled,
    verify_token,
)

_SECRET = "test-jwt-secret-for-auth-tests"
_ALG = "HS256"


def _make_token(payload: dict, secret: str = _SECRET) -> str:
    return pyjwt.encode(payload, secret, algorithm=_ALG)


def _valid_token(user_id: str = "507f1f77bcf86cd799439011") -> str:
    return _make_token({"id": user_id})


def _expired_token(user_id: str = "507f1f77bcf86cd799439011") -> str:
    return _make_token({"id": user_id, "exp": int(time.time()) - 3600})


# ---- verify_token ----


class TestVerifyToken:
    def test_valid_token(self):
        result = verify_token(_valid_token(), _SECRET)
        assert result.authenticated is True
        assert result.user_id == "507f1f77bcf86cd799439011"
        assert result.error is None

    def test_valid_token_different_user(self):
        result = verify_token(_valid_token("abc123"), _SECRET)
        assert result.authenticated is True
        assert result.user_id == "abc123"

    def test_expired_token(self):
        result = verify_token(_expired_token(), _SECRET)
        assert result.authenticated is False
        assert result.error == "Token expired"

    def test_malformed_token(self):
        result = verify_token("not.a.valid.token", _SECRET)
        assert result.authenticated is False
        assert "Invalid token" in result.error

    def test_wrong_secret(self):
        token = _make_token({"id": "user1"}, "wrong-secret")
        result = verify_token(token, _SECRET)
        assert result.authenticated is False
        assert "Invalid token" in result.error

    def test_missing_id_field(self):
        token = _make_token({"sub": "user1"})
        result = verify_token(token, _SECRET)
        assert result.authenticated is False
        assert "missing 'id' field" in result.error

    def test_id_field_not_string(self):
        token = _make_token({"id": 12345})
        result = verify_token(token, _SECRET)
        assert result.authenticated is False
        assert "missing 'id' field" in result.error

    def test_empty_id_field(self):
        token = _make_token({"id": ""})
        result = verify_token(token, _SECRET)
        assert result.authenticated is False
        assert "missing 'id' field" in result.error


# ---- authenticate_request ----


class TestAuthenticateRequestPassthrough:
    """Default mode: FINAI_AUTH_ENABLED not set, no enforcement."""

    @patch.dict(os.environ, {}, clear=True)
    def test_no_secret_no_token(self):
        os.environ.pop("JWT_SECRET", None)
        os.environ.pop("LIBRECHAT_JWT_SECRET", None)
        os.environ.pop("FINAI_AUTH_ENABLED", None)
        result = authenticate_request({}, fallback_user="curl_user")
        assert result.authenticated is False
        assert result.user_id == "curl_user"

    @patch.dict(os.environ, {}, clear=True)
    def test_no_secret_no_fallback(self):
        os.environ.pop("JWT_SECRET", None)
        os.environ.pop("LIBRECHAT_JWT_SECRET", None)
        os.environ.pop("FINAI_AUTH_ENABLED", None)
        result = authenticate_request({})
        assert result.user_id == "anonymous"

    @patch.dict(os.environ, {"JWT_SECRET": _SECRET}, clear=True)
    def test_valid_token_passthrough(self):
        os.environ.pop("FINAI_AUTH_ENABLED", None)
        result = authenticate_request(
            {"x-librechat-token": _valid_token()},
            fallback_user="fallback",
        )
        assert result.authenticated is True
        assert result.user_id == "507f1f77bcf86cd799439011"

    @patch.dict(os.environ, {"JWT_SECRET": _SECRET}, clear=True)
    def test_invalid_token_uses_fallback(self):
        os.environ.pop("FINAI_AUTH_ENABLED", None)
        result = authenticate_request(
            {"x-librechat-token": "bad-token"},
            fallback_user="fallback_user",
        )
        assert result.authenticated is False
        assert result.user_id == "fallback_user"


class TestAuthenticateRequestEnforced:
    """FINAI_AUTH_ENABLED=true: unauthenticated requests get rejected."""

    @patch.dict(
        os.environ,
        {"JWT_SECRET": _SECRET, "FINAI_AUTH_ENABLED": "true"},
        clear=True,
    )
    def test_valid_token_passes(self):
        result = authenticate_request(
            {"x-librechat-token": _valid_token("user_abc")},
        )
        assert result.authenticated is True
        assert result.user_id == "user_abc"

    @patch.dict(
        os.environ,
        {"JWT_SECRET": _SECRET, "FINAI_AUTH_ENABLED": "true"},
        clear=True,
    )
    def test_missing_token_rejected(self):
        result = authenticate_request({})
        assert result.authenticated is False
        assert "Missing" in result.error
        assert result.user_id == "anonymous"

    @patch.dict(
        os.environ,
        {"JWT_SECRET": _SECRET, "FINAI_AUTH_ENABLED": "true"},
        clear=True,
    )
    def test_expired_token_rejected(self):
        result = authenticate_request({"x-librechat-token": _expired_token()})
        assert result.authenticated is False
        assert result.error == "Token expired"

    @patch.dict(
        os.environ,
        {"JWT_SECRET": _SECRET, "FINAI_AUTH_ENABLED": "1"},
        clear=True,
    )
    def test_auth_enabled_value_1(self):
        result = authenticate_request({})
        assert result.authenticated is False

    @patch.dict(
        os.environ,
        {"JWT_SECRET": _SECRET, "FINAI_AUTH_ENABLED": "yes"},
        clear=True,
    )
    def test_auth_enabled_value_yes(self):
        result = authenticate_request({})
        assert result.authenticated is False

    @patch.dict(
        os.environ,
        {"FINAI_AUTH_ENABLED": "true"},
        clear=True,
    )
    def test_enabled_but_no_secret(self):
        os.environ.pop("JWT_SECRET", None)
        result = authenticate_request({}, fallback_user="dev_user")
        assert result.authenticated is False
        assert "JWT_SECRET not configured" in result.error
        assert result.user_id == "dev_user"


# ---- Header extraction ----


class TestHeaderExtraction:
    @patch.dict(os.environ, {"JWT_SECRET": _SECRET}, clear=True)
    def test_x_librechat_token_header(self):
        result = authenticate_request({"x-librechat-token": _valid_token("h1")})
        assert result.authenticated is True
        assert result.user_id == "h1"

    @patch.dict(os.environ, {"JWT_SECRET": _SECRET}, clear=True)
    def test_x_librechat_token_case_insensitive(self):
        result = authenticate_request({"X-LibreChat-Token": _valid_token("h2")})
        assert result.authenticated is True
        assert result.user_id == "h2"

    @patch.dict(os.environ, {"JWT_SECRET": _SECRET}, clear=True)
    def test_authorization_bearer_header(self):
        token = _valid_token("bearer_user")
        result = authenticate_request({"authorization": f"Bearer {token}"})
        assert result.authenticated is True
        assert result.user_id == "bearer_user"

    @patch.dict(os.environ, {"JWT_SECRET": _SECRET}, clear=True)
    def test_x_librechat_token_takes_precedence(self):
        result = authenticate_request(
            {
                "x-librechat-token": _valid_token("direct"),
                "authorization": f"Bearer {_valid_token('bearer')}",
            }
        )
        assert result.authenticated is True
        assert result.user_id == "direct"

    @patch.dict(os.environ, {"JWT_SECRET": _SECRET}, clear=True)
    def test_authorization_without_bearer_prefix(self):
        result = authenticate_request({"authorization": _valid_token("raw")})
        assert result.authenticated is False


# ---- is_auth_enabled / get_jwt_secret ----


class TestConfigHelpers:
    @patch.dict(os.environ, {"FINAI_AUTH_ENABLED": "true"}, clear=True)
    def test_is_auth_enabled_true(self):
        assert is_auth_enabled() is True

    @patch.dict(os.environ, {"FINAI_AUTH_ENABLED": "false"}, clear=True)
    def test_is_auth_enabled_false(self):
        assert is_auth_enabled() is False

    @patch.dict(os.environ, {}, clear=True)
    def test_is_auth_enabled_default(self):
        os.environ.pop("FINAI_AUTH_ENABLED", None)
        assert is_auth_enabled() is False

    @patch.dict(os.environ, {"JWT_SECRET": "abc"}, clear=True)
    def test_get_jwt_secret_from_jwt_secret(self):
        assert get_jwt_secret() == "abc"

    @patch.dict(os.environ, {"LIBRECHAT_JWT_SECRET": "xyz"}, clear=True)
    def test_get_jwt_secret_from_librechat_jwt_secret(self):
        os.environ.pop("JWT_SECRET", None)
        assert get_jwt_secret() == "xyz"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_jwt_secret_none(self):
        os.environ.pop("JWT_SECRET", None)
        os.environ.pop("LIBRECHAT_JWT_SECRET", None)
        assert get_jwt_secret() is None

    @patch.dict(
        os.environ,
        {"JWT_SECRET": "primary", "LIBRECHAT_JWT_SECRET": "secondary"},
        clear=True,
    )
    def test_jwt_secret_takes_precedence(self):
        assert get_jwt_secret() == "primary"
