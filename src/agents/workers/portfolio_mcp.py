"""Portfolio (Upstox-style) MCP worker server.

Represents the user's brokerage account. For the demo this reads a
deterministic fixture (``data/fixtures/portfolio.json``) rather than calling
the real Upstox API - swapping in the real client is a drop-in change to
the ``_load_user_portfolio`` function.

Exposes both raw holdings + a set of **deterministic** analytics tools so
the persona agents have hard numbers to cite instead of having to reinvent
concentration arithmetic inside an LLM:

* ``get_holdings``                -> list of holdings with weights
* ``get_portfolio_summary``       -> total value, top holdings, geo split
* ``get_sector_allocation``       -> raw + grouped sector weights
* ``get_concentration_risks``     -> flagged risks (sector, single-stock, top-N)
* ``get_diversification_score``   -> HHI-based score 0-100 with reasons

Run as a standalone MCP server over stdio::

    python -m src.agents.workers.portfolio_mcp
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from fastmcp import FastMCP

from ._fixtures import load_fixture


# ---------------------------------------------------------------------------
# Fixture loading + enrichment
# ---------------------------------------------------------------------------
_RAW: Dict[str, Any] = load_fixture("portfolio")


def _grouped_sector(raw_sector: str) -> str:
    """Collapse the raw fixture sectors into broader analytics buckets.

    Keeps the demo readable ("Technology broad" is more useful than four
    overlapping tech subsectors when reporting concentration risks).
    """
    s = raw_sector.lower()
    if "information technology" in s or "it services" in s:
        return "Technology (IT Services)"
    if "technology" in s or "semiconductor" in s or "software" in s:
        return "Technology (Products/Cloud)"
    if "communication" in s:
        return "Communication Services"
    if "financial" in s or "bank" in s:
        return "Financial Services"
    if "energy" in s or "conglomerate" in s:
        return "Energy & Conglomerate"
    if "consumer staples" in s or "fmcg" in s:
        return "Consumer Staples"
    if "consumer cyclical" in s or "auto" in s:
        return "Consumer Cyclical"
    return raw_sector


def _load_user_portfolio(user_id: str) -> Dict[str, Any]:
    """Look up ``user_id`` in the fixture, compute derived fields once."""
    if user_id not in _RAW:
        available = sorted(k for k in _RAW.keys() if not k.startswith("_"))
        raise KeyError(f"User '{user_id}' not found. Available: {available}")
    profile = dict(_RAW[user_id])  # shallow copy
    holdings = [dict(h) for h in profile["holdings"]]
    total_value = sum(h["current_value_usd"] for h in holdings)
    for h in holdings:
        h["weight"] = round(h["current_value_usd"] / total_value, 4)
        h["grouped_sector"] = _grouped_sector(h["sector"])
    profile["holdings"] = holdings
    profile["total_value_usd"] = total_value
    return profile


# ---------------------------------------------------------------------------
# Deterministic analytics helpers
# ---------------------------------------------------------------------------
def _aggregate_weights(holdings: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    buckets: Dict[str, float] = defaultdict(float)
    for h in holdings:
        buckets[h[key]] += h["weight"]
    return {k: round(v, 4) for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])}


def _herfindahl(holdings: List[Dict[str, Any]]) -> float:
    return sum(h["weight"] ** 2 for h in holdings)


def _score_from_hhi(hhi: float) -> float:
    """Map Herfindahl-Hirschman index to a 0-100 diversification score.

    Reference ranges (standard in portfolio analysis):
      HHI < 0.10        -> highly diversified
      HHI 0.10 - 0.18   -> moderately concentrated
      HHI 0.18 - 0.25   -> highly concentrated
      HHI > 0.25        -> dangerously concentrated
    """
    if hhi < 0.10:
        return 90.0 - (hhi / 0.10) * 5.0  # 85-90
    if hhi < 0.18:
        return 85.0 - ((hhi - 0.10) / 0.08) * 30.0  # 55-85
    if hhi < 0.25:
        return 55.0 - ((hhi - 0.18) / 0.07) * 25.0  # 30-55
    return max(0.0, 30.0 - min((hhi - 0.25) / 0.15, 1.0) * 30.0)  # 0-30


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="portfolio",
    instructions=(
        "Represents the user's brokerage (Upstox) account. Exposes raw "
        "holdings plus deterministic analytics (sector allocation, "
        "concentration risks, diversification score) computed in Python "
        "so persona agents never need to estimate these numbers themselves."
    ),
)


@mcp.tool
def list_supported_users() -> Dict[str, Any]:
    """Return the user IDs this fixture portfolio server can serve."""
    return {"users": sorted(k for k in _RAW.keys() if not k.startswith("_"))}


@mcp.tool
def get_holdings(user_id: str = "demo") -> Dict[str, Any]:
    """Return the user's raw holdings with weights, P&L, and sector tags.

    Args:
        user_id: brokerage account identifier (default ``"demo"``).
    """
    p = _load_user_portfolio(user_id)
    return {
        "user_id": user_id,
        "portfolio_name": p["portfolio_name"],
        "base_currency": p["base_currency"],
        "total_value_usd": p["total_value_usd"],
        "holding_count": len(p["holdings"]),
        "holdings": p["holdings"],
    }


@mcp.tool
def get_portfolio_summary(user_id: str = "demo") -> Dict[str, Any]:
    """Return a compact, human-readable summary of the user's portfolio.

    Includes total value, top 5 holdings by weight, aggregate P&L, and
    geographic split (India vs US, by market value).
    """
    p = _load_user_portfolio(user_id)
    holdings = p["holdings"]
    total = p["total_value_usd"]

    gain = sum(h.get("absolute_gain_usd", 0) for h in holdings)
    cost = total - gain

    top5 = [
        {
            "ticker": h["ticker"],
            "name": h["name"],
            "sector": h["sector"],
            "weight_pct": round(h["weight"] * 100, 2),
            "value_usd": h["current_value_usd"],
        }
        for h in sorted(holdings, key=lambda x: -x["weight"])[:5]
    ]

    geo = _aggregate_weights(holdings, "country")

    return {
        "user_id": user_id,
        "portfolio_name": p["portfolio_name"],
        "total_value_usd": total,
        "total_cost_usd": cost,
        "absolute_gain_usd": gain,
        "absolute_gain_pct": round((gain / cost) * 100, 2) if cost else 0,
        "holding_count": len(holdings),
        "top_5_holdings": top5,
        "geographic_split": {k: round(v * 100, 2) for k, v in geo.items()},
    }


@mcp.tool
def get_sector_allocation(user_id: str = "demo") -> Dict[str, Any]:
    """Return sector weights in both raw and grouped form."""
    p = _load_user_portfolio(user_id)
    holdings = p["holdings"]
    raw = _aggregate_weights(holdings, "sector")
    grouped = _aggregate_weights(holdings, "grouped_sector")
    return {
        "user_id": user_id,
        "raw_sectors_pct": {k: round(v * 100, 2) for k, v in raw.items()},
        "grouped_sectors_pct": {k: round(v * 100, 2) for k, v in grouped.items()},
        "largest_raw_sector": max(raw.items(), key=lambda kv: kv[1])[0],
        "largest_grouped_sector": max(grouped.items(), key=lambda kv: kv[1])[0],
    }


@mcp.tool
def get_concentration_risks(user_id: str = "demo") -> Dict[str, Any]:
    """Return a list of flagged concentration risks using standard thresholds."""
    p = _load_user_portfolio(user_id)
    holdings = p["holdings"]

    risks: List[Dict[str, Any]] = []

    # Single-stock concentration (> 15%)
    for h in holdings:
        if h["weight"] > 0.15:
            risks.append({
                "type": "single_stock",
                "severity": "medium" if h["weight"] < 0.20 else "high",
                "detail": f"{h['ticker']} is {round(h['weight']*100, 1)}% of the portfolio "
                          f"(threshold 15%).",
            })

    # Grouped sector concentration (> 30%)
    grouped = _aggregate_weights(holdings, "grouped_sector")
    for sector, w in grouped.items():
        if w > 0.30:
            risks.append({
                "type": "sector",
                "severity": "high" if w > 0.50 else "medium",
                "detail": f"{sector} is {round(w*100, 1)}% of the portfolio "
                          f"(threshold 30%).",
            })

    # Top-5 concentration (> 60%)
    top5_weight = sum(h["weight"] for h in sorted(holdings, key=lambda x: -x["weight"])[:5])
    if top5_weight > 0.60:
        risks.append({
            "type": "top_n",
            "severity": "medium",
            "detail": f"Top 5 holdings are {round(top5_weight*100, 1)}% of the portfolio "
                      f"(threshold 60%).",
        })

    # Geographic concentration (single country > 80%)
    geo = _aggregate_weights(holdings, "country")
    max_geo = max(geo.items(), key=lambda kv: kv[1]) if geo else ("?", 0)
    if max_geo[1] > 0.80:
        risks.append({
            "type": "geographic",
            "severity": "low",
            "detail": f"{round(max_geo[1]*100, 1)}% of the portfolio is in a single country "
                      f"({max_geo[0]}). Consider international exposure.",
        })

    return {
        "user_id": user_id,
        "risk_count": len(risks),
        "risks": risks,
    }


@mcp.tool
def get_diversification_score(user_id: str = "demo") -> Dict[str, Any]:
    """Return a diversification score (0-100) based on Herfindahl-Hirschman + heuristics."""
    p = _load_user_portfolio(user_id)
    holdings = p["holdings"]
    hhi = _herfindahl(holdings)
    base_score = _score_from_hhi(hhi)

    # Adjustments based on broader diversification signals
    adjustments: List[Dict[str, Any]] = []
    score = base_score

    grouped = _aggregate_weights(holdings, "grouped_sector")
    max_sector = max(grouped.values()) if grouped else 0
    if max_sector > 0.50:
        score -= 20
        adjustments.append({
            "reason": f"Largest sector > 50% (is {round(max_sector*100, 1)}%)",
            "delta": -20,
        })
    elif max_sector > 0.30:
        score -= 10
        adjustments.append({
            "reason": f"Largest sector > 30% (is {round(max_sector*100, 1)}%)",
            "delta": -10,
        })

    countries = set(h["country"] for h in holdings)
    if len(countries) >= 2:
        score += 3
        adjustments.append({"reason": "Multi-country exposure", "delta": 3})

    score = max(0, min(100, round(score)))

    # Interpretation band
    if score >= 80:
        band = "well diversified"
    elif score >= 60:
        band = "moderately diversified"
    elif score >= 40:
        band = "concentrated"
    else:
        band = "highly concentrated"

    return {
        "user_id": user_id,
        "score": score,
        "band": band,
        "herfindahl_index": round(hhi, 4),
        "effective_holdings": round(1 / hhi, 2) if hhi > 0 else None,
        "base_score_from_hhi": round(base_score, 1),
        "adjustments": adjustments,
    }


if __name__ == "__main__":  # pragma: no cover - entrypoint
    mcp.run()
