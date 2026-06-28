"""Unit tests for src.core.cache — get/put roundtrip + size-bounded eviction."""
from __future__ import annotations

import tempfile
import unittest

from src.core.cache import ResponseCache


def _put(cache: ResponseCache, *, agent: str, content: str = "hello world") -> None:
    cache.put(
        user_id="u",
        query="q",
        flow="f",
        agent=agent,
        agent_title=agent.title(),
        content=content,
    )


class CacheRoundtripTests(unittest.TestCase):
    def test_put_then_get_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            cache = ResponseCache(root=d, max_entries=0)
            _put(cache, agent="buffett", content="value pick")
            got = cache.get(user_id="u", query="q", flow="f", agent="buffett")
            self.assertIsNotNone(got)
            self.assertEqual(got.content, "value pick")

    def test_empty_content_is_not_cached(self):
        with tempfile.TemporaryDirectory() as d:
            cache = ResponseCache(root=d, max_entries=0)
            _put(cache, agent="buffett", content="   ")
            self.assertEqual(cache.stats()["entries"], 0)


class CacheEvictionTests(unittest.TestCase):
    def test_eviction_caps_entry_count(self):
        with tempfile.TemporaryDirectory() as d:
            cache = ResponseCache(root=d, max_entries=3)
            for i in range(10):
                _put(cache, agent=f"persona_{i}")
            self.assertLessEqual(cache.stats()["entries"], 3)

    def test_zero_max_entries_disables_eviction(self):
        with tempfile.TemporaryDirectory() as d:
            cache = ResponseCache(root=d, max_entries=0)
            for i in range(20):
                _put(cache, agent=f"persona_{i}")
            self.assertEqual(cache.stats()["entries"], 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
