"""Tests for src.mcp._retrieval — semantic re-ranking, dedup, freshness."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

# The retrieval re-ranker depends on sentence-transformers (which pulls torch),
# an optional/heavy dep. Skip this whole module when it isn't installed so the
# fast CI lane stays green; the full/nightly env installs it and runs these.
pytest.importorskip(
    "sentence_transformers",
    reason="optional retrieval re-ranker dep (sentence-transformers/torch)",
)

from src.mcp._retrieval import (
    _extract_year,
    deduplicate,
    filter_by_date_window,
    filter_by_freshness,
    is_reranking_available,
    process_items,
    semantic_rerank,
    RetrievalStats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
SAMPLE_ITEMS = [
    {
        "title": "Nvidia earnings beat expectations Q1 2025",
        "snippet": "Nvidia reported Q1 FY2025 revenue of $26B beating analyst estimates.",
        "url": "https://example.com/nvda-earnings",
        "published": "2025-05-22",
        "score": 0.9,
    },
    {
        "title": "Nvidia earnings beat expectations Q1 2025",
        "snippet": "Nvidia reported Q1 FY2025 revenue of $26B beating analyst estimates.",
        "url": "https://example.com/nvda-earnings-dup",
        "published": "2025-05-22",
        "score": 0.85,
    },
    {
        "title": "Old article about GPUs from 2018",
        "snippet": "GPU market overview from 2018 showing modest growth in data center segment.",
        "url": "https://example.com/old-gpu",
        "published": "2018-03-15",
        "score": 0.5,
    },
    {
        "title": "China AI chip restrictions analysis",
        "snippet": "New US export controls on AI chips to China could reshape supply chains.",
        "url": "https://example.com/china-chips",
        "published": None,
        "score": None,
    },
]


# ---------------------------------------------------------------------------
# _extract_year
# ---------------------------------------------------------------------------
class TestExtractYear:
    def test_from_published_iso(self):
        assert _extract_year({"published": "2024-06-15"}) == 2024

    def test_from_published_year_only(self):
        assert _extract_year({"published": "2023"}) == 2023

    def test_from_snippet_text(self):
        item = {"published": None, "snippet": "Report dated January 15, 2024 shows growth"}
        assert _extract_year(item) == 2024

    def test_from_title_text(self):
        item = {"published": None, "snippet": "", "title": "NVDA Q3 2025 earnings review"}
        assert _extract_year(item) == 2025

    def test_no_date_anywhere(self):
        item = {"published": None, "snippet": "Generic text without dates", "title": "No dates here"}
        assert _extract_year(item) is None

    def test_abbreviated_month(self):
        item = {"published": None, "snippet": "Updated Sep 2023 with new data"}
        assert _extract_year(item) == 2023

    def test_out_of_range_year_ignored(self):
        item = {"published": None, "snippet": "In the year 1999 and also 2024"}
        assert _extract_year(item) == 2024


# ---------------------------------------------------------------------------
# filter_by_freshness
# ---------------------------------------------------------------------------
class TestFilterByFreshness:
    def test_removes_outdated(self):
        result = filter_by_freshness(SAMPLE_ITEMS, min_year=2020)
        years = [it.get("_extracted_year") for it in result if "_extracted_year" in it]
        assert all(y >= 2020 for y in years)
        assert len(result) == 3  # 2 fresh + 1 no-date kept

    def test_keeps_no_date(self):
        result = filter_by_freshness(SAMPLE_ITEMS, min_year=2025)
        no_date_items = [it for it in result if it.get("url") == "https://example.com/china-chips"]
        assert len(no_date_items) == 1

    def test_empty_input(self):
        assert filter_by_freshness([], min_year=2020) == []


# ---------------------------------------------------------------------------
# filter_by_date_window
# ---------------------------------------------------------------------------
class TestFilterByDateWindow:
    def test_keeps_items_in_window(self):
        result = filter_by_date_window(SAMPLE_ITEMS, "2025-01-01", "2025-12-31")
        published = [it.get("published") for it in result if it.get("published")]
        for p in published:
            assert p.startswith("2025")

    def test_keeps_no_date_items(self):
        result = filter_by_date_window(SAMPLE_ITEMS, "2025-01-01", "2025-12-31")
        no_date = [it for it in result if it.get("url") == "https://example.com/china-chips"]
        assert len(no_date) == 1

    def test_invalid_window_returns_all(self):
        result = filter_by_date_window(SAMPLE_ITEMS, "not-a-date", "also-bad")
        assert len(result) == len(SAMPLE_ITEMS)

    def test_empty_input(self):
        assert filter_by_date_window([], "2024-01-01", "2024-12-31") == []


# ---------------------------------------------------------------------------
# semantic_rerank / deduplicate with model mocked
# ---------------------------------------------------------------------------
class TestSemanticRerankMocked:
    @patch("src.mcp._retrieval._get_model")
    def test_rerank_returns_all_items(self, mock_model):
        m = MagicMock()
        import numpy as np
        fake_sims = MagicMock()
        fake_sims.__len__ = lambda s: 2
        m.encode.return_value = MagicMock()
        mock_model.return_value = m

        with patch("src.mcp._retrieval.torch") as mock_torch:
            mock_torch.nn.functional.cosine_similarity.return_value = MagicMock()
            mock_torch.argsort.return_value.cpu.return_value.numpy.return_value = np.array([0, 1])
            mock_torch.nn.functional.cosine_similarity.return_value = MagicMock()
            items = [{"title": "A", "snippet": "alpha"}, {"title": "B", "snippet": "beta"}]
            result = semantic_rerank(items, "test query")
            assert len(result) == 2

    @patch("src.mcp._retrieval._get_model")
    def test_rerank_no_model_returns_unchanged(self, mock_model):
        mock_model.return_value = None
        items = [{"title": "A", "snippet": "alpha"}]
        result = semantic_rerank(items, "query")
        assert result == items

    @patch("src.mcp._retrieval._get_model")
    def test_dedup_no_model_returns_unchanged(self, mock_model):
        mock_model.return_value = None
        items = [{"title": "A", "snippet": "alpha"}, {"title": "B", "snippet": "beta"}]
        result = deduplicate(items)
        assert result == items

    @patch("src.mcp._retrieval._get_model")
    def test_dedup_single_item(self, mock_model):
        mock_model.return_value = None
        items = [{"title": "A", "snippet": "alpha"}]
        result = deduplicate(items)
        assert result == items


# ---------------------------------------------------------------------------
# process_items
# ---------------------------------------------------------------------------
class TestProcessItems:
    @patch("src.mcp._retrieval._get_model")
    def test_full_pipeline_no_model(self, mock_model):
        mock_model.return_value = None
        items, stats = process_items(
            SAMPLE_ITEMS,
            "nvidia earnings",
            rerank=True,
            dedup=True,
            freshness=True,
        )
        assert stats.initial == 4
        assert isinstance(stats, RetrievalStats)

    def test_freshness_only_no_model(self):
        with patch("src.mcp._retrieval._get_model", return_value=None):
            items, stats = process_items(
                SAMPLE_ITEMS,
                "nvidia earnings",
                rerank=False,
                dedup=False,
                freshness=True,
                min_year=2020,
            )
            assert len(items) == 3  # 2 fresh + 1 no-date
            assert stats.after_freshness == 3

    def test_date_window_filtering_no_model(self):
        with patch("src.mcp._retrieval._get_model", return_value=None):
            items, stats = process_items(
                SAMPLE_ITEMS,
                "nvidia earnings",
                rerank=False,
                dedup=False,
                date_window=("2025-01-01", "2025-12-31"),
            )
            assert len(items) == 3  # 2 in-window + 1 no-date

    def test_empty_items(self):
        items, stats = process_items([], "query")
        assert items == []
        assert stats.initial == 0


# ---------------------------------------------------------------------------
# is_reranking_available
# ---------------------------------------------------------------------------
class TestIsRerankingAvailable:
    def test_returns_bool(self):
        assert isinstance(is_reranking_available(), bool)


# ---------------------------------------------------------------------------
# RetrievalStats
# ---------------------------------------------------------------------------
class TestRetrievalStats:
    def test_to_dict(self):
        s = RetrievalStats(10)
        s.after_rerank = 8
        s.after_dedup = 7
        s.after_freshness = 6
        d = s.to_dict()
        assert d["initial"] == 10
        assert d["duplicates_removed"] == 1
        assert d["outdated_removed"] == 1
