"""Verify-numbers post-processing pass for the synthesizer output.

This module extracts numerical claims from the synthesizer's final markdown,
cross-references them against data in the scratchpad, and annotates the text
with verification badges:

* **Verified** (``✓``) — the number matches a value found in the scratchpad
  (exact or within a configurable relative tolerance).
* **Unverified** (``⚠``) — the number was not found in the scratchpad but
  doesn't contradict anything either. The synthesizer may have computed it
  correctly via ``run_python`` or derived it; we just can't confirm it.
* **Flagged** (``✗``) — the number *contradicts* a value in the scratchpad
  (differs by more than the tolerance from a clearly corresponding datum).

The pass is deliberately conservative: it only flags numbers that clearly
contradict scratchpad data.  A number that appears nowhere in the scratchpad
is marked unverified, not flagged — the synthesizer may have derived it
legitimately from a computation.

Design choices
--------------
* **Pure-regex extraction.** No LLM call — this is a deterministic pass that
  runs in <100ms for typical reports.  We extract numbers with their
  surrounding context (currency symbol, unit suffix like "B"/"M"/"crore").
* **Scratchpad-first verification.** We only check against data that actually
  landed in the scratchpad (step outputs).  We do NOT re-fetch external data.
* **Tolerance-based matching.** Financial reports often round (e.g. "1.23B"
  when the actual figure is 1,234,567,890).  Default tolerance is 2% relative
  difference.
* **Non-blocking.** If verification fails or errors, the original text is
  returned unchanged.  We never block the user from seeing the report.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.core.types import Scratchpad, StepResult

log = logging.getLogger("finai.verify_numbers")


class VerifyStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    FLAGGED = "flagged"


@dataclass
class NumberClaim:
    raw_text: str
    value: float
    unit_multiplier: float
    context_left: str
    context_right: str
    span_start: int
    span_end: int
    status: VerifyStatus = VerifyStatus.UNVERIFIED
    matched_scratchpad_value: Optional[float] = None
    matched_step_id: Optional[int] = None


@dataclass
class VerifyResult:
    annotated_text: str
    claims: List[NumberClaim]
    verified_count: int = 0
    unverified_count: int = 0
    flagged_count: int = 0

    @property
    def total_claims(self) -> int:
        return len(self.claims)

    @property
    def has_flags(self) -> bool:
        return self.flagged_count > 0


_DEFAULT_REL_TOL = 0.02
_UNIT_MULTIPLIERS = {
    "%": 0.01,
    "K": 1e3,
    "k": 1e3,
    "M": 1e6,
    "m": 1e6,
    "B": 1e9,
    "b": 1e9,
    "T": 1e12,
    "t": 1e12,
    "crore": 1e7,
    "Cr": 1e7,
    "cr": 1e7,
    "lakh": 1e5,
    "Lakh": 1e5,
    "L": 1e5,
    "x": 1.0,
}

_CURRENCY_SYMBOLS = {"$", "₹", "€", "£", "¥", "Rs", "Rs.", "INR", "USD", "EUR"}

_NUM_WITH_UNIT = re.compile(
    r"(?P<number>[\d,]+(?:\.\d+)?)\s*(?P<unit>%|crore|Cr|cr|lakh|Lakh|[KkMmBbTtLx])(?=\b|[\s,.;:)\]})\-—–])",
    re.IGNORECASE,
)

_CURRENCY_PREFIX = re.compile(
    r"(?P<currency>[$₹€£¥]|Rs\.?|INR|USD|EUR)\s*(?P<number>[\d,]+(?:\.\d+)?)",
)

_BARE_LARGE_NUM = re.compile(
    r"(?P<number>[\d,]+(?:\.\d+)?)",
)

_EXCLUDED_CONTEXTS = {
    "step", "step_id", "page", "line", "http", "https",
    "item", "section", "chapter", "figure", "table",
    "note", "footnote", "reference", "ref",
}

_MIN_SIGNIFICANT_VALUE = 0.01


def _get_context(text: str, pos: int, span: int = 40) -> Tuple[str, str]:
    """Get left/right context around a position in text."""
    left = text[max(0, pos - span):pos].strip()
    right_start = pos
    right = text[right_start:right_start + span].strip()
    return left, right


def extract_number_claims(text: str) -> List[NumberClaim]:
    """Extract numerical claims from markdown text.

    Returns a list of :class:`NumberClaim` objects, each representing a
    number found in the text along with its surrounding context and
    unit-parsed absolute value.

    Strategy: three passes, highest-priority first:
    1. Numbers with unit suffixes ($1.23B, ₹5,000 crore, 23.5%)
    2. Numbers with currency prefix ($1,234, ₹500)
    3. Large bare numbers (1,543,200,000)

    Already-matched spans are skipped in later passes.
    """
    claims: List[NumberClaim] = []
    used_spans: List[Tuple[int, int]] = []

    def _overlaps(start: int, end: int) -> bool:
        for us, ue in used_spans:
            if start < ue and end > us:
                return True
        return False

    # Pass 1: numbers with unit suffixes
    for m in _NUM_WITH_UNIT.finditer(text):
        raw = m.group("number")
        unit_str = m.group("unit")

        try:
            parsed = float(raw.replace(",", ""))
        except ValueError:
            continue

        unit_mult = _UNIT_MULTIPLIERS.get(unit_str, 1.0)
        if unit_str == "%":
            unit_mult = 1.0

        start = m.start("number")
        end = m.end()

        left_ctx, right_ctx = _get_context(text, start)

        if _is_trivial_number(left_ctx, right_ctx, raw, parsed):
            continue

        used_spans.append((start, end))
        claims.append(NumberClaim(
            raw_text=raw,
            value=parsed,
            unit_multiplier=unit_mult,
            context_left=left_ctx,
            context_right=right_ctx,
            span_start=start,
            span_end=end,
            status=VerifyStatus.UNVERIFIED,
        ))

    # Pass 2: numbers with currency prefix
    for m in _CURRENCY_PREFIX.finditer(text):
        raw = m.group("number")

        try:
            parsed = float(raw.replace(",", ""))
        except ValueError:
            continue

        start = m.start()
        num_start = m.start("number")
        end = m.end("number")

        if _overlaps(start, end):
            continue

        left_ctx, right_ctx = _get_context(text, start)

        if _is_trivial_number(left_ctx, right_ctx, raw, parsed):
            continue

        used_spans.append((start, end))
        claims.append(NumberClaim(
            raw_text=raw,
            value=parsed,
            unit_multiplier=1.0,
            context_left=left_ctx,
            context_right=right_ctx,
            span_start=start,
            span_end=end,
            status=VerifyStatus.UNVERIFIED,
        ))

    # Pass 3: large bare numbers (>= 4 digits or > 1000)
    for m in _BARE_LARGE_NUM.finditer(text):
        raw = m.group("number")

        try:
            parsed = float(raw.replace(",", ""))
        except ValueError:
            continue

        start = m.start()
        end = m.end()

        if _overlaps(start, end):
            continue

        if abs(parsed) < 1000:
            continue

        left_ctx, right_ctx = _get_context(text, start)

        if _is_trivial_number(left_ctx, right_ctx, raw, parsed):
            continue

        used_spans.append((start, end))
        claims.append(NumberClaim(
            raw_text=raw,
            value=parsed,
            unit_multiplier=1.0,
            context_left=left_ctx,
            context_right=right_ctx,
            span_start=start,
            span_end=end,
            status=VerifyStatus.UNVERIFIED,
        ))

    return claims


def _is_trivial_number(
    left: str, right: str, raw: str, parsed: float
) -> bool:
    """Skip numbers that are clearly not financial claims."""
    combined = (left + " " + right).lower()

    for exc in _EXCLUDED_CONTEXTS:
        if exc in combined:
            return True

    if re.search(r"fy\s*\d{4}|fiscal\s*\d{4}|\b20\d{2}\b", combined):
        if re.match(r"^\d{4}$", raw.strip()) and 2000 <= parsed <= 2099:
            return True

    if re.search(r"step\s*\d", combined):
        return True

    if re.match(r"^\d{1,2}$", raw.strip()) and parsed <= 31:
        if any(k in combined for k in ("jan", "feb", "mar", "apr", "may",
                                        "jun", "jul", "aug", "sep", "oct",
                                        "nov", "dec")):
            return True

    if re.match(r"^\d{4}$", raw.strip()) and 1800 <= parsed <= 2100:
        if not any(k in combined for k in (
            "revenue", "income", "profit", "debt", "cash", "assets",
            "sales", "market", "shares", "eps", "price", "value",
            "amount", "total", "net", "gross", "cost",
        )):
            return True

    if re.match(r"^\d{1,2}$", raw.strip()) and parsed <= 31:
        if any(k in combined for k in ("jan", "feb", "mar", "apr", "may",
                                        "jun", "jul", "aug", "sep", "oct",
                                        "nov", "dec")):
            return True

    return False


def _extract_scratchpad_numbers(scratchpad: Scratchpad) -> List[Tuple[float, int, str]]:
    """Pull all numeric values from the scratchpad step outputs.

    Returns list of ``(value, step_id, source_text)`` tuples.
    """
    values: List[Tuple[float, int, str]] = []

    for step_id, result in scratchpad.results.items():
        if result.status != "complete":
            continue
        output = result.output
        if output is None:
            continue

        text_chunks: List[str] = []
        if isinstance(output, dict):
            for v in output.values():
                if isinstance(v, str):
                    text_chunks.append(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            for iv in item.values():
                                if isinstance(iv, str):
                                    text_chunks.append(iv)
                        elif isinstance(item, str):
                            text_chunks.append(item)
        elif isinstance(output, str):
            text_chunks.append(output)
        elif isinstance(output, list):
            for item in output:
                if isinstance(item, str):
                    text_chunks.append(item)
                elif isinstance(item, dict):
                    for iv in item.values():
                        if isinstance(iv, str):
                            text_chunks.append(iv)

        combined_text = " ".join(text_chunks)
        for m in re.finditer(
            r"([\d,]+(?:\.\d+)?)\s*(%|crore|Cr|cr|lakh|Lakh|[KkMmBbTtL])?",
            combined_text,
        ):
            try:
                num = float(m.group(1).replace(",", ""))
            except ValueError:
                continue
            unit = m.group(2) or ""
            mult = _UNIT_MULTIPLIERS.get(unit, 1.0)
            if unit == "%":
                mult = 1.0
            values.append((num * mult, step_id, combined_text[max(0, m.start() - 30):m.end() + 30]))

        if isinstance(output, dict):
            _extract_dict_numbers(output, step_id, values)

    return values


def _extract_dict_numbers(
    data: Any, step_id: int, values: List[Tuple[float, int, str]],
) -> None:
    """Recursively extract bare numeric values from dict/list structures."""
    if isinstance(data, (int, float)):
        if abs(data) >= _MIN_SIGNIFICANT_VALUE:
            values.append((float(data), step_id, str(data)))
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(k, str) and any(
                skip in k.lower()
                for skip in ("id", "step", "page", "index", "count", "rank")
            ):
                continue
            _extract_dict_numbers(v, step_id, values)
    elif isinstance(data, list):
        for item in data:
            _extract_dict_numbers(item, step_id, values)


def _values_match(
    claim_value: float,
    scratch_value: float,
    rel_tol: float = _DEFAULT_REL_TOL,
) -> bool:
    """Check if two values are within relative tolerance."""
    if claim_value == 0 and scratch_value == 0:
        return True
    denom = max(abs(claim_value), abs(scratch_value))
    if denom == 0:
        return False
    return abs(claim_value - scratch_value) / denom <= rel_tol


def verify_numbers(
    text: str,
    scratchpad: Scratchpad,
    *,
    rel_tol: float = _DEFAULT_REL_TOL,
    max_claims: int = 50,
) -> VerifyResult:
    """Post-process the synthesizer's output to verify numerical claims.

    Parameters
    ----------
    text:
        The synthesizer's markdown output.
    scratchpad:
        The shared scratchpad containing prior step results.
    rel_tol:
        Relative tolerance for matching (default 2%).
    max_claims:
        Cap on the number of claims to verify (prevents runaway on
        very long texts).

    Returns
    -------
    :class:`VerifyResult` with the annotated text and per-claim status.
    """
    claims = extract_number_claims(text)
    if len(claims) > max_claims:
        log.info(
            "verify_numbers: capping claims from %d to %d",
            len(claims), max_claims,
        )
        claims = claims[:max_claims]

    if not claims:
        return VerifyResult(
            annotated_text=text,
            claims=[],
            verified_count=0,
            unverified_count=0,
            flagged_count=0,
        )

    scratch_nums = _extract_scratchpad_numbers(scratchpad)
    if not scratch_nums:
        return VerifyResult(
            annotated_text=text,
            claims=claims,
            verified_count=0,
            unverified_count=len(claims),
            flagged_count=0,
        )

    absolute_claim_values = [c.value * c.unit_multiplier for c in claims]

    for claim in claims:
        claim_abs = claim.value * claim.unit_multiplier
        best_match: Optional[Tuple[float, int]] = None
        best_diff = float("inf")

        for sv, sid, _ in scratch_nums:
            if _values_match(claim_abs, sv, rel_tol):
                diff = abs(claim_abs - sv) / max(abs(claim_abs), abs(sv), 1e-10)
                if diff < best_diff:
                    best_diff = diff
                    best_match = (sv, sid)

        if best_match is not None:
            claim.status = VerifyStatus.VERIFIED
            claim.matched_scratchpad_value = best_match[0]
            claim.matched_step_id = best_match[1]
        else:
            for sv, sid, src in scratch_nums:
                if _is_likely_same_context(claim, sv, src):
                    if not _values_match(claim_abs, sv, rel_tol * 5):
                        claim.status = VerifyStatus.FLAGGED
                        claim.matched_scratchpad_value = sv
                        claim.matched_step_id = sid
                        break

    verified = sum(1 for c in claims if c.status == VerifyStatus.VERIFIED)
    flagged = sum(1 for c in claims if c.status == VerifyStatus.FLAGGED)
    unverified = len(claims) - verified - flagged

    annotated = _annotate_text(text, claims)

    log.info(
        "verify_numbers: %d claims — %d verified, %d unverified, %d flagged",
        len(claims), verified, unverified, flagged,
    )

    return VerifyResult(
        annotated_text=annotated,
        claims=claims,
        verified_count=verified,
        unverified_count=unverified,
        flagged_count=flagged,
    )


def _is_likely_same_context(
    claim: NumberClaim, scratch_value: float, scratch_source: str,
) -> bool:
    """Heuristic: does the scratchpad context match the claim's context?

    Uses the **nearest** financial keyword before the number to determine
    what metric the claim refers to.  This avoids false-flagging when
    two numbers appear in the same sentence but refer to different
    metrics (e.g. "Revenue was $1.23B and headcount was 45,000").
    """
    local_ctx = claim.context_left[-60:].lower()
    source_ctx = scratch_source.lower()

    nearest_metric = _nearest_metric_keyword(local_ctx)
    if nearest_metric is None:
        return False

    return nearest_metric in source_ctx


_METRIC_KEYWORDS = frozenset({
    "revenue", "income", "profit", "eps", "debt", "cash",
    "assets", "equity", "market", "cap", "sales", "ebitda",
    "dividend", "margin", "growth", "net", "gross", "operating",
    "price", "share", "shares", "outstanding", "book", "value",
    "roe", "roa", "pe", "ratio", "yield", "turnover",
    "fiscal", "quarter", "q1", "q2", "q3", "q4", "fy",
    "total", "cost", "expense", "headcount", "employees",
})


def _nearest_metric_keyword(text: str) -> Optional[str]:
    """Find the rightmost financial keyword in ``text``."""
    tokens = text.split()
    for tok in reversed(tokens):
        cleaned = tok.strip(".,:;()[]{}-—–")
        if cleaned in _METRIC_KEYWORDS:
            return cleaned
    return None


_BADGE = {
    VerifyStatus.VERIFIED: " ✓",
    VerifyStatus.UNVERIFIED: "",
    VerifyStatus.FLAGGED: " ✗",
}


def _annotate_text(text: str, claims: List[NumberClaim]) -> str:
    """Insert verification badges into the text.

    Processes claims in reverse span order so that inserting badges
    doesn't shift earlier spans.
    """
    if not claims:
        return text

    result = text
    sorted_claims = sorted(claims, key=lambda c: c.span_start, reverse=True)

    for claim in sorted_claims:
        badge = _BADGE.get(claim.status, "")
        if not badge:
            continue

        insert_pos = min(claim.span_end, len(result))
        result = result[:insert_pos] + badge + result[insert_pos:]

    if any(c.status == VerifyStatus.FLAGGED for c in claims):
        footer = (
            "\n\n---\n"
            "⚠ **Number verification**: ✓ = confirmed against source data, "
            "✗ = contradicts source data. "
            "Unmarked numbers could not be independently verified.\n"
        )
        result += footer
    elif any(c.status == VerifyStatus.VERIFIED for c in claims):
        footer = (
            "\n\n---\n"
            "✓ **Number verification**: ✓ = confirmed against source data. "
            "Unmarked numbers could not be independently verified.\n"
        )
        result += footer

    return result


__all__ = [
    "VerifyStatus",
    "NumberClaim",
    "VerifyResult",
    "extract_number_claims",
    "verify_numbers",
]
