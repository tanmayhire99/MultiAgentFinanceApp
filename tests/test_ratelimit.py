"""Tests for the token-bucket rate limiter (:mod:`src.core.ratelimit`)."""

from __future__ import annotations

import os
import time
from unittest.mock import patch

from src.core.ratelimit import RateLimiter, create_limiter_from_env


class TestRateLimiter:
    def test_allows_within_capacity(self):
        rl = RateLimiter(max_tokens=3, refill_rate=1.0)
        for _ in range(3):
            result = rl.check("user1")
            assert result.allowed is True
        result = rl.check("user1")
        assert result.allowed is False
        assert result.retry_after_seconds is not None

    def test_independent_buckets_per_user(self):
        rl = RateLimiter(max_tokens=1, refill_rate=0.0)
        assert rl.check("alice").allowed is True
        assert rl.check("alice").allowed is False
        assert rl.check("bob").allowed is True

    def test_refill_over_time(self):
        rl = RateLimiter(max_tokens=1, refill_rate=100.0)
        rl.check("user1")
        assert rl.check("user1").allowed is False
        time.sleep(0.02)
        result = rl.check("user1")
        assert result.allowed is True

    def test_remaining_decrements(self):
        rl = RateLimiter(max_tokens=5, refill_rate=0.0)
        result = rl.check("user1")
        assert result.allowed is True
        assert result.remaining == 4

    def test_remaining_zero_when_empty(self):
        rl = RateLimiter(max_tokens=1, refill_rate=0.0)
        rl.check("user1")
        result = rl.check("user1")
        assert result.remaining == 0

    def test_reset_clears_bucket(self):
        rl = RateLimiter(max_tokens=1, refill_rate=0.0)
        rl.check("user1")
        assert rl.check("user1").allowed is False
        rl.reset("user1")
        assert rl.check("user1").allowed is True

    def test_active_users_count(self):
        rl = RateLimiter(max_tokens=5, refill_rate=0.0)
        rl.check("a")
        rl.check("b")
        assert rl.active_users == 2

    def test_retry_after_positive(self):
        rl = RateLimiter(max_tokens=1, refill_rate=10.0)
        rl.check("user1")
        result = rl.check("user1")
        assert result.retry_after_seconds > 0

    def test_bucket_caps_at_max(self):
        rl = RateLimiter(max_tokens=3, refill_rate=1000.0)
        time.sleep(0.01)
        result = rl.check("user1")
        assert result.remaining <= 3

    def test_reset_nonexistent_user(self):
        rl = RateLimiter(max_tokens=5, refill_rate=1.0)
        rl.reset("ghost")
        assert rl.active_users == 0


class TestCreateLimiterFromEnv:
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("FINAI_RATE_LIMIT_MAX", None)
            os.environ.pop("FINAI_RATE_LIMIT_REFILL", None)
            rl = create_limiter_from_env()
            assert rl.max_tokens == 20
            assert rl.refill_rate == 1.0

    def test_custom_values(self):
        with patch.dict(
            os.environ,
            {"FINAI_RATE_LIMIT_MAX": "5", "FINAI_RATE_LIMIT_REFILL": "2.0"},
            clear=True,
        ):
            rl = create_limiter_from_env()
            assert rl.max_tokens == 5
            assert rl.refill_rate == 2.0
