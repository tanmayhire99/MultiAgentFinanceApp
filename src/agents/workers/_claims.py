"""LLM-based **claim extraction** and **claim-vs-reality comparison**.

This module is the "special sauce" of the Deep Stock Research flow. It
turns long, unstructured corporate text (earnings transcripts, 10-K
excerpts, press releases, news archives) into two structured pipelines:

1. :func:`extract_forward_claims` - pulls every testable forward-looking
   commitment from a document as a JSON list of ``{claim_text, metric,
   target_value, target_date, confidence, source}`` objects.
2. :func:`compare_claim_to_reality` - given one such claim plus a
   bundle of recent evidence (news, filings, metrics), produces a
   verdict: ``{verdict: met|missed|partial|pending|unknowable,
   variance_pct, variance_time, explanation, confidence,
   evidence_snippets}``.

Both use the same NVIDIA NIM chat model the rest of the system uses,
with ``response_format={"type": "json_object"}`` and a tight system
prompt so the output is parseable without tool-call scaffolding.

Design notes
------------
* **Strict inclusion/exclusion rules** in the extraction prompt are
  essential. Without them the LLM regresses to "everything is a claim"
  and floods the scratchpad with hedges ("we aim to", "potentially",
  "subject to market conditions").
* Extraction is **idempotent and stateless** - no dependence on
  previously-extracted claims. Each document is scored fresh. This
  keeps the pipeline easy to cache.
* The verdict prompt asks for ``evidence_snippets`` so the final
  rendered report can quote the actuals directly, not just summarise
  them. Crucial for an audience that's sceptical of LLM hand-waving.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.personas.base import build_chat_model


log = logging.getLogger("finai.claims")


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
_EXTRACT_SYSTEM = """You are a financial analyst extracting FORWARD-LOOKING CLAIMS from a corporate document.

A forward-looking claim is a SPECIFIC, TESTABLE commitment that can later be verified against reality. Each claim MUST have:
- A METRIC (revenue, unit count, margin, product milestone, autonomy level, launch date, market share, capacity, etc.)
- A TARGET VALUE (numeric like "$8-9B", "20M units", OR a named state like "L3 autonomy", "general availability")
- A TARGET DATE (absolute like "2025-12-31" or relative like "by EOY 2024", "Q2 2025", "FY26")

STRICT INCLUSION:
- Explicit guidance: "We expect FY26 revenue between $8-9B"
- Product milestones: "FSD v12 will reach L3 autonomy by EOY 2024"
- Launch plans: "Vision Pro launches in Q1 2026"
- Capacity / unit targets: "We will produce 20M vehicles by 2030"
- Margin targets: "Gross margin of 45% by FY25"
- Deal or contract milestones: "$2B multi-year deal signed, closing Q2 2026"

STRICT EXCLUSION:
- Hedging language: "We aim to", "We hope", "Potentially", "Subject to market conditions"
- Retrospective statements: "We achieved X", "In Q3 we shipped Y" (these are actuals, not claims)
- Industry / macro commentary: "The AI market will grow" (not a company claim)
- Vague aspiration: "We'll focus on innovation", "We'll continue investing"
- Risk-factor boilerplate from 10-K risk sections

OUTPUT FORMAT:
Return a single JSON object with a "claims" field (array). Each array element must match this schema:

{
  "claim_text": "<near-exact quote, max 220 characters>",
  "metric": "<short label, e.g. 'FY26 revenue', 'FSD autonomy level'>",
  "target_value": "<e.g. '$8-9B', 'L3', '100M units', 'Q2 2026 launch'>",
  "target_date": "<YYYY-MM-DD, or EOY YYYY, or Q[1-4] YYYY, or FY[YY]>",
  "confidence": "<low|medium|high>"
}

Confidence meaning:
- high = unambiguous, numeric, explicit forward-looking guidance
- medium = specific but slightly qualified ("assuming no further Covid disruption...")
- low = borderline - claim is specific but heavily hedged

If NO forward-looking claims exist, return {"claims": []}.
"""


_COMPARE_SYSTEM = """You are a financial analyst producing a VERDICT on whether a company's past claim was met.

INPUT:
- An ORIGINAL CLAIM (metric, target_value, target_date)
- RECENT EVIDENCE (news snippets, filings, current metrics) that may or may not relate to the claim

OUTPUT: A single JSON object:

{
  "verdict": "<met|missed|partial|pending|unknowable>",
  "variance_pct": <number or null>,
  "variance_time": "<string or null>",
  "explanation": "<1-2 sentences citing the specific evidence, <=320 chars>",
  "confidence": "<low|medium|high>",
  "evidence_snippets": ["<direct quote 1>", "<direct quote 2>"]
}

VERDICT RULES:
- met: clear evidence the target was achieved (numeric tolerance 5%, or the named state is explicitly reached)
- missed: clear evidence the target was NOT achieved (below target, delayed past target_date, product pivoted away)
- partial: evidence of partial delivery (e.g. "achieved L2 but not the promised L3", "$7.5B vs promised $8-9B")
- pending: target date hasn't passed yet, or the actuals don't address the claim yet
- unknowable: evidence doesn't directly address this claim (say so, don't speculate)

RULES:
- variance_pct: ONLY set when both target and actual are numeric; otherwise null.
- variance_time: e.g. "-18 months" if a milestone slipped 18 months past the target date; null otherwise.
- evidence_snippets: 1-3 direct quotes from the evidence text that back your verdict. Never invent quotes.
- confidence: low if the evidence is thin / circumstantial; high if there's unambiguous reporting.
- If verdict is "unknowable" or "pending", evidence_snippets may be an empty array.

DO NOT recommend buy/sell. DO NOT speculate beyond the evidence shown.
"""


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------
async def extract_forward_claims(
    text: str,
    *,
    source_label: str = "",
    max_claims: int = 15,
    input_char_budget: int = 8000,
) -> List[Dict[str, Any]]:
    """Extract forward-looking claims from ``text`` using an LLM JSON call.

    Args:
        text: source document (transcript, filing section, press release).
            For long documents (>10k words), the caller should chunk
            into 8000-char slices and call this once per slice.
        source_label: free-text tag added to every returned claim,
            e.g. ``"NVDA Q1 FY25 earnings call, 2024-05-22"``. Used
            later when rendering the verdict report so the audience
            can see where each claim came from.
        max_claims: hard cap - the LLM is instructed to stop at this
            many regardless of how many could be extracted.
        input_char_budget: hard truncation before sending to the LLM,
            so oversized inputs don't blow the context window.

    Returns:
        List of claim dicts. ``[]`` on LLM failure or parse error.
    """
    if not text or not text.strip():
        return []

    llm = build_chat_model(
        temperature=0.1,
        max_tokens=1800,
        streaming=False,
        response_format={"type": "json_object"},
    )

    user_msg = (
        f"Source: {source_label or '(unlabeled)'}\n\n"
        f"Document text (may be truncated to {input_char_budget} chars):\n\n"
        f"{text[:input_char_budget]}\n\n"
        f"Extract forward-looking claims per the system instructions. "
        f"Return at most {max_claims} claims, ordered by specificity. "
        f'Output the JSON object {{"claims": [...]}} and nothing else.'
    )
    messages = [
        SystemMessage(content=_EXTRACT_SYSTEM),
        HumanMessage(content=user_msg),
    ]

    try:
        resp = await llm.ainvoke(messages)
    except Exception as e:
        log.warning("Claim extraction LLM failed for %r: %s", source_label, e)
        return []

    content = getattr(resp, "content", "") or ""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        log.warning(
            "Claim extraction returned non-JSON for %r: %s | content=%r",
            source_label,
            e,
            content[:240],
        )
        return []

    # Accept either a raw list or an object with a "claims" key.
    if isinstance(data, list):
        claims = data
    elif isinstance(data, dict):
        claims = data.get("claims") or []
    else:
        return []

    if not isinstance(claims, list):
        return []

    # Tag each claim with its source so downstream rendering can attribute.
    out: List[Dict[str, Any]] = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        c = dict(c)  # shallow copy so we don't mutate the LLM's output
        c.setdefault("source", source_label or "unknown")
        out.append(c)
        if len(out) >= max_claims:
            break

    log.info(
        "Claim extraction: %d claims from %r (%d chars in)",
        len(out),
        source_label,
        len(text),
    )
    return out


# ---------------------------------------------------------------------------
# Claim vs reality comparison
# ---------------------------------------------------------------------------
async def compare_claim_to_reality(
    claim: Dict[str, Any],
    actuals_context: str,
    *,
    evidence_char_budget: int = 6000,
) -> Dict[str, Any]:
    """LLM-diff a single claim against recent evidence.

    Args:
        claim: a dict returned by :func:`extract_forward_claims`.
        actuals_context: free-form text bundling the relevant actuals -
            typically the concatenation of recent news snippets,
            current fundamentals, and latest filing excerpts. Feel
            free to include a leading header like
            ``"LATEST 10-Q (filed 2026-02-25)\\n..."`` - the LLM will
            use that context to attribute evidence.
        evidence_char_budget: truncation cap on ``actuals_context``.

    Returns:
        Verdict dict with fields ``{verdict, variance_pct,
        variance_time, explanation, confidence, evidence_snippets,
        claim}``. On failure, returns ``{verdict: "unknowable", ...}``
        so the caller can carry on without special-casing errors.
    """
    if not claim or not actuals_context:
        return {
            "verdict": "unknowable",
            "explanation": "Missing claim or evidence input.",
            "confidence": "low",
            "claim": claim or {},
            "evidence_snippets": [],
        }

    llm = build_chat_model(
        temperature=0.1,
        max_tokens=900,
        streaming=False,
        response_format={"type": "json_object"},
    )

    user_msg = (
        f"ORIGINAL CLAIM:\n{json.dumps(claim, indent=2)}\n\n"
        f"RECENT EVIDENCE (may be truncated):\n\n"
        f"{actuals_context[:evidence_char_budget]}\n\n"
        "Produce the verdict JSON object per the system instructions. "
        "Cite concrete numbers / dates from the evidence wherever "
        "possible. Output the JSON object and nothing else."
    )
    messages = [
        SystemMessage(content=_COMPARE_SYSTEM),
        HumanMessage(content=user_msg),
    ]

    try:
        resp = await llm.ainvoke(messages)
    except Exception as e:
        log.warning("Claim-compare LLM failed: %s", e)
        return {
            "verdict": "unknowable",
            "explanation": f"LLM error: {e}",
            "confidence": "low",
            "claim": claim,
            "evidence_snippets": [],
        }

    content = getattr(resp, "content", "") or ""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        log.warning("Claim-compare returned non-JSON: %s", content[:240])
        return {
            "verdict": "unknowable",
            "explanation": "LLM did not return valid JSON.",
            "confidence": "low",
            "claim": claim,
            "evidence_snippets": [],
        }

    if not isinstance(data, dict):
        return {
            "verdict": "unknowable",
            "explanation": "LLM returned a non-object response.",
            "confidence": "low",
            "claim": claim,
            "evidence_snippets": [],
        }

    # Attach the original claim so downstream rendering has everything
    # it needs in a single dict.
    data["claim"] = claim
    # Normalise missing optional fields so downstream code doesn't need
    # a bunch of ``.get()`` fallbacks.
    data.setdefault("verdict", "unknowable")
    data.setdefault("variance_pct", None)
    data.setdefault("variance_time", None)
    data.setdefault("confidence", "low")
    data.setdefault("evidence_snippets", [])
    data.setdefault("explanation", "")
    return data
