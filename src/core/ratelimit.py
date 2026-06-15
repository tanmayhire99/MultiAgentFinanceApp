"""Token-bucket rate limiter for per-user request throttling.

No external dependencies — uses only stdlib. Each user gets an
independent bucket. The bucket refills at ``refill_rate`` tokens per
second and caps at ``max_tokens``. A request costs 1 token; if the
bucket is empty, the request is rejected with a ``429``-style result.

This is intentionally lightweight: no Redis, no persistence across
restarts, no distributed coordination. For a single-container demo
this is sufficient; for multi-replica production you'd swap in a
Redis-backed limiter.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

log = logging.getLogger("finai.ratelimit")

_DEFAULT_MAX_TOKENS = 20
_DEFAULT_REFILL_RATE = 1.0


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    retry_after_seconds: Optional[float] = None


class RateLimiter:
    """Per-user token-bucket rate limiter.

    Parameters
    ----------
    max_tokens:
        Maximum burst capacity per user.
    refill_rate:
        Tokens added per second.
    """

    def __init__(
        self,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        refill_rate: float = _DEFAULT_REFILL_RATE,
    ) -> None:
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self._buckets: Dict[str, _Bucket] = {}

    def check(self, user_id: str) -> RateLimitResult:
        """Check whether *user_id* is allowed to make a request.

        Consumes 1 token on success; returns ``allowed=False`` when
        the bucket is empty.
        """
        now = time.monotonic()
        bucket = self._buckets.get(user_id)
        if bucket is None:
            bucket = _Bucket(tokens=float(self.max_tokens), last_refill=now)
            self._buckets[user_id] = bucket

        elapsed = now - bucket.last_refill
        bucket.tokens = min(
            self.max_tokens,
            bucket.tokens + elapsed * self.refill_rate,
        )
        bucket.last_refill = now

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return RateLimitResult(
                allowed=True,
                remaining=int(bucket.tokens),
            )

        retry_after = (1.0 - bucket.tokens) / self.refill_rate if self.refill_rate > 0 else None
        return RateLimitResult(
            allowed=False,
            remaining=0,
            retry_after_seconds=round(retry_after, 2) if retry_after is not None else None,
        )

    def reset(self, user_id: str) -> None:
        self._buckets.pop(user_id, None)

    @property
    def active_users(self) -> int:
        return len(self._buckets)


def create_limiter_from_env() -> RateLimiter:
    """Build a :class:`RateLimiter` from env vars.

    Reads ``FINAI_RATE_LIMIT_MAX`` (default 20) and
    ``FINAI_RATE_LIMIT_REFILL`` (default 1.0).
    """
    max_tokens = int(os.environ.get("FINAI_RATE_LIMIT_MAX", str(_DEFAULT_MAX_TOKENS)))
    refill_rate = float(os.environ.get("FINAI_RATE_LIMIT_REFILL", str(_DEFAULT_REFILL_RATE)))
    return RateLimiter(max_tokens=max_tokens, refill_rate=refill_rate)


__all__ = ["RateLimiter", "RateLimitResult", "create_limiter_from_env"]
