"""Retrieval post-processing: semantic re-ranking, dedup, freshness filtering.

Adapted from Finance-LLMs/deep-research-python's retrieval_processor.py but
rewritten to fit our architecture:

* Works on our normalised item dicts (``title``, ``snippet``, ``url``,
  ``published``, ``score``) instead of Firecrawl's ``markdown``/``content``.
* Lazy-initialises the sentence-transformer model on first use (avoids
  pulling ~90 MB of weights when the container starts if no search ever
  happens).
* All three stages are optional — each public search function decides which
  stages to run based on context (e.g. freshness filtering only for
  historical queries).
* Graceful fallback: if ``sentence-transformers`` is not installed, every
  stage is a no-op and the original items are returned unchanged.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import torch  # type: ignore
except ImportError:  # pragma: no cover — optional dep
    SentenceTransformer = None  # type: ignore
    torch = None  # type: ignore

log = logging.getLogger("finai.retrieval")

# ---------------------------------------------------------------------------
# Model singleton (lazy-loaded on first call)
# ---------------------------------------------------------------------------
_MODEL: Optional[Any] = None
_MODEL_INIT_ATTEMPTED: bool = False
_MODEL_NAME: str = "all-MiniLM-L6-v2"


def _get_model() -> Optional[Any]:
    global _MODEL, _MODEL_INIT_ATTEMPTED
    if _MODEL_INIT_ATTEMPTED:
        return _MODEL
    _MODEL_INIT_ATTEMPTED = True
    if SentenceTransformer is None:
        log.info("sentence-transformers not installed — retrieval post-processing disabled")
        return None
    try:
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        _MODEL = SentenceTransformer(_MODEL_NAME, device=device)
        log.info("Retrieval model initialised (%s on %s)", _MODEL_NAME, device)
    except Exception as e:  # pragma: no cover — defensive
        log.warning("Failed to load retrieval model: %s", e)
        _MODEL = None
    return _MODEL


def is_reranking_available() -> bool:
    return _get_model() is not None


# ---------------------------------------------------------------------------
# Stage 1: Semantic re-ranking
# ---------------------------------------------------------------------------
def semantic_rerank(
    items: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    """Re-rank items by cosine similarity of snippet to query.

    Falls back to returning items unchanged if the model is unavailable.
    """
    model = _get_model()
    if not model or not items:
        return items

    texts = [((it.get("snippet") or "") + " " + (it.get("title") or ""))[:1000] for it in items]
    try:
        q_emb = model.encode(query, convert_to_tensor=True)
        d_emb = model.encode(texts, convert_to_tensor=True)
        sims = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), d_emb)
        order = torch.argsort(sims, descending=True).cpu().numpy()
    except Exception as e:  # pragma: no cover
        log.warning("Semantic re-ranking failed: %s", e)
        return items

    ranked = []
    for idx in order:
        item = items[int(idx)].copy()
        item["_similarity"] = round(float(sims[idx]), 4)
        ranked.append(item)
    log.info(
        "semantic_rerank: %d items, top similarity=%.3f",
        len(ranked),
        ranked[0].get("_similarity", 0) if ranked else 0,
    )
    return ranked


# ---------------------------------------------------------------------------
# Stage 2: Deduplication
# ---------------------------------------------------------------------------
def deduplicate(
    items: List[Dict[str, Any]],
    threshold: float = 0.90,
) -> List[Dict[str, Any]]:
    """Remove near-duplicate items based on snippet similarity.

    O(n²) pairwise comparison — acceptable because we typically have ≤10 items
    per search call. Falls back to returning items unchanged if the model is
    unavailable.
    """
    model = _get_model()
    if not model or len(items) <= 1:
        return items

    texts = [((it.get("snippet") or "") + " " + (it.get("title") or "")) for it in items]
    try:
        embeddings = model.encode(texts, convert_to_tensor=True)
        sims = torch.nn.functional.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2,
        )
    except Exception as e:  # pragma: no cover
        log.warning("Dedup embedding failed: %s", e)
        return items

    keep: List[int] = []
    removed: set = set()
    for i in range(len(items)):
        if i in removed:
            continue
        keep.append(i)
        for j in range(i + 1, len(items)):
            if j not in removed and float(sims[i][j]) > threshold:
                removed.add(j)

    result = [items[i] for i in keep]
    n_dupes = len(items) - len(result)
    if n_dupes:
        log.info("deduplicate: removed %d near-duplicates (threshold=%.2f)", n_dupes, threshold)
    return result


# ---------------------------------------------------------------------------
# Stage 3: Freshness filtering
# ---------------------------------------------------------------------------
_DATE_PATTERNS = [
    re.compile(r"(20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])"),
    re.compile(
        r"(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s+(20\d{2})",
    ),
    re.compile(
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[a-z]*\s+(20\d{2})",
    ),
    re.compile(r"\b(20\d{2})\b"),
]

_THIS_YEAR = datetime.now().year


def _extract_year(item: Dict[str, Any]) -> Optional[int]:
    """Best-effort year extraction from published date or snippet text."""
    pub = item.get("published")
    if pub:
        s = str(pub)[:10]
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y", "%Y"):
            try:
                return datetime.strptime(s, fmt).year
            except ValueError:
                continue
    text = (item.get("snippet") or "")[:2000] + " " + (item.get("title") or "")[:500]
    years: List[int] = []
    for pat in _DATE_PATTERNS:
        for m in pat.findall(text):
            try:
                y = int(re.search(r"20\d{2}", m if isinstance(m, str) else m[0]).group())
                if 2000 <= y <= _THIS_YEAR:
                    years.append(y)
            except (AttributeError, ValueError):
                continue
    return max(years) if years else None


def filter_by_freshness(
    items: List[Dict[str, Any]],
    min_year: int = 2020,
) -> List[Dict[str, Any]]:
    """Remove items published before ``min_year``.

    Items with no detectable date are kept (benefit of the doubt).
    """
    if not items:
        return items

    fresh: List[Dict[str, Any]] = []
    n_outdated = 0
    n_nodate = 0
    for item in items:
        year = _extract_year(item)
        if year is None:
            n_nodate += 1
            fresh.append(item)
        elif year >= min_year:
            item["_extracted_year"] = year
            fresh.append(item)
        else:
            n_outdated += 1

    if n_outdated or n_nodate:
        log.info(
            "filter_by_freshness: kept %d, removed %d outdated, %d no-date",
            len(fresh), n_outdated, n_nodate,
        )
    return fresh


def filter_by_date_window(
    items: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
) -> List[Dict[str, Any]]:
    """Keep only items whose published/extracted date falls within [start_date, end_date].

    Items with no detectable date are kept (benefit of the doubt).
    This closes the DDG ``enforced=False`` gap for historical queries.
    """
    if not items:
        return items

    try:
        s = datetime.strptime(start_date, "%Y-%m-%d")
        e = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        log.warning("Invalid date window %s / %s — skipping filter", start_date, end_date)
        return items

    kept: List[Dict[str, Any]] = []
    n_removed = 0
    n_nodate = 0
    for item in items:
        year = _extract_year(item)
        if year is None:
            n_nodate += 1
            kept.append(item)
            continue
        # If we only have a year, check that it overlaps the window.
        try:
            pub_str = str(item.get("published") or "")[:10]
            if pub_str and len(pub_str) >= 8:
                pub_dt = datetime.strptime(pub_str[:10], "%Y-%m-%d")
                in_window = s <= pub_dt <= e
            else:
                in_window = s.year <= year <= e.year
        except (ValueError, TypeError):
            in_window = s.year <= year <= e.year

        if in_window:
            item["_extracted_year"] = year
            kept.append(item)
        else:
            n_removed += 1

    if n_removed:
        log.info(
            "filter_by_date_window [%s..%s]: kept %d, removed %d outside window",
            start_date, end_date, len(kept), n_removed,
        )
    return kept


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------
class RetrievalStats:
    __slots__ = ("initial", "after_rerank", "after_dedup", "after_freshness")

    def __init__(self, initial: int):
        self.initial = initial
        self.after_rerank = initial
        self.after_dedup = initial
        self.after_freshness = initial

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial": self.initial,
            "after_rerank": self.after_rerank,
            "after_dedup": self.after_dedup,
            "after_freshness": self.after_freshness,
            "duplicates_removed": self.after_rerank - self.after_dedup,
            "outdated_removed": self.after_dedup - self.after_freshness,
        }


def process_items(
    items: List[Dict[str, Any]],
    query: str,
    *,
    rerank: bool = True,
    dedup: bool = True,
    freshness: bool = False,
    date_window: Optional[tuple] = None,
    min_year: int = 2020,
    dedup_threshold: float = 0.90,
) -> tuple:
    """Run the full retrieval post-processing pipeline.

    Returns ``(processed_items, RetrievalStats)``.
    """
    stats = RetrievalStats(len(items))
    if not items:
        return items, stats

    if rerank:
        items = semantic_rerank(items, query)
        stats.after_rerank = len(items)

    if dedup:
        items = deduplicate(items, threshold=dedup_threshold)
        stats.after_dedup = len(items)

    if freshness:
        items = filter_by_freshness(items, min_year=min_year)
        stats.after_freshness = len(items)

    if date_window:
        items = filter_by_date_window(items, date_window[0], date_window[1])
        stats.after_freshness = len(items)

    log.info(
        "retrieval pipeline: %d → %d items (rerank=%s dedup=%s fresh=%s window=%s)",
        stats.initial,
        len(items),
        rerank,
        dedup,
        freshness,
        bool(date_window),
    )
    return items, stats
