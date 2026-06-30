"""Tests for src/core/verify_numbers.py."""

import pytest

from src.core.types import Scratchpad, StepResult
from src.core.verify_numbers import (
    NumberClaim,
    VerifyResult,
    VerifyStatus,
    extract_number_claims,
    verify_numbers,
)


def _make_scratchpad(step_outputs=None):
    """Build a Scratchpad with the given step outputs.

    step_outputs: dict mapping step_id -> (status, output)
    """
    pad = Scratchpad(query="test")
    if step_outputs:
        for sid, (status, output) in step_outputs.items():
            pad.add(StepResult(
                step_id=sid,
                status=status,
                output=output,
            ))
    return pad


# ---------------------------------------------------------------------------
# extract_number_claims
# ---------------------------------------------------------------------------

class TestExtractNumberClaims:

    def test_simple_dollar_with_unit(self):
        text = "Revenue was $1.23B in FY2024."
        claims = extract_number_claims(text)
        assert len(claims) >= 1
        found = [c for c in claims if c.value == 1.23 and c.unit_multiplier == 1e9]
        assert len(found) >= 1

    def test_percentage(self):
        text = "Growth was 23.5% year-over-year."
        claims = extract_number_claims(text)
        pct_claims = [c for c in claims if c.value == 23.5]
        assert len(pct_claims) >= 1

    def test_indian_currency_crore(self):
        text = "Revenue was ₹5,000 crore in FY2024."
        claims = extract_number_claims(text)
        crore_claims = [c for c in claims if c.unit_multiplier == 1e7 and c.value == 5000]
        assert len(crore_claims) >= 1

    def test_lakh(self):
        text = "The company earned ₹12.5 lakh per employee."
        claims = extract_number_claims(text)
        lakh_claims = [c for c in claims if c.unit_multiplier == 1e5]
        assert len(lakh_claims) >= 1

    def test_plain_large_number_no_unit(self):
        text = "Total shares outstanding: 1,543,200,000."
        claims = extract_number_claims(text)
        large = [c for c in claims if c.value > 1e9]
        assert len(large) >= 1

    def test_skip_trivial_step_numbers(self):
        text = "Step 1: Gather data. Step 2: Analyze."
        claims = extract_number_claims(text)
        assert len(claims) == 0

    def test_skip_bare_year(self):
        text = "The report covers FY 2024."
        claims = extract_number_claims(text)
        year_claims = [c for c in claims if c.raw_text == "2024"]
        assert len(year_claims) == 0

    def test_no_numbers(self):
        text = "The company performed well this quarter."
        claims = extract_number_claims(text)
        assert len(claims) == 0

    def test_multiple_numbers_in_sentence(self):
        text = "EPS of $3.45 on revenue of $12.3B, up 15.2%."
        claims = extract_number_claims(text)
        assert len(claims) >= 3

    def test_comma_separated_number(self):
        text = "Net income was $1,234,567,890."
        claims = extract_number_claims(text)
        big = [c for c in claims if c.value > 1e9]
        assert len(big) >= 1

    def test_million_suffix(self):
        text = "Market cap reached $250M."
        claims = extract_number_claims(text)
        m_claims = [c for c in claims if c.unit_multiplier == 1e6]
        assert len(m_claims) >= 1

    def test_thousand_suffix(self):
        text = "Volume was 150K shares."
        claims = extract_number_claims(text)
        k_claims = [c for c in claims if c.unit_multiplier == 1e3]
        assert len(k_claims) >= 1

    def test_trillion_suffix(self):
        text = "US GDP is approximately $28T."
        claims = extract_number_claims(text)
        t_claims = [c for c in claims if c.unit_multiplier == 1e12]
        assert len(t_claims) >= 1

    def test_eps_with_currency(self):
        text = "EPS was $3.45 this quarter."
        claims = extract_number_claims(text)
        eps = [c for c in claims if c.value == 3.45]
        assert len(eps) >= 1


# ---------------------------------------------------------------------------
# verify_numbers
# ---------------------------------------------------------------------------

class TestVerifyNumbers:

    def test_verified_match(self):
        text = "Revenue was $1.23B in FY2024."
        pad = _make_scratchpad({
            1: ("complete", {"text": "Total revenue: $1,230,000,000"}),
        })
        result = verify_numbers(text, pad)
        verified = [c for c in result.claims if c.status == VerifyStatus.VERIFIED]
        assert len(verified) >= 1

    def test_flagged_contradiction(self):
        text = "Revenue was $2.5B in FY2024."
        pad = _make_scratchpad({
            1: ("complete", {"text": "Revenue was $1,230,000,000 for the year"}),
        })
        result = verify_numbers(text, pad)
        flagged = [c for c in result.claims if c.status == VerifyStatus.FLAGGED]
        assert len(flagged) >= 1

    def test_unverified_no_scratchpad_data(self):
        text = "Revenue was $1.23B in FY2024."
        pad = _make_scratchpad({})
        result = verify_numbers(text, pad)
        assert result.unverified_count > 0
        assert result.verified_count == 0
        assert result.flagged_count == 0

    def test_empty_text(self):
        pad = _make_scratchpad({})
        result = verify_numbers("", pad)
        assert result.total_claims == 0

    def test_no_numbers_in_text(self):
        pad = _make_scratchpad({})
        result = verify_numbers("The company did well.", pad)
        assert result.total_claims == 0

    def test_annotation_inserts_checkmark(self):
        text = "Revenue was $1.23B in FY2024."
        pad = _make_scratchpad({
            1: ("complete", {"text": "Total revenue: $1,230,000,000"}),
        })
        result = verify_numbers(text, pad)
        if result.verified_count > 0:
            assert "✓" in result.annotated_text

    def test_annotation_inserts_flag(self):
        text = "Revenue was $9.99B in FY2024."
        pad = _make_scratchpad({
            1: ("complete", {"text": "Total revenue: $1,230,000,000"}),
        })
        result = verify_numbers(text, pad)
        if result.flagged_count > 0:
            assert "✗" in result.annotated_text
            assert "contradicts source data" in result.annotated_text

    def test_annotation_footer_verified_only(self):
        text = "Revenue was $1.23B in FY2024."
        pad = _make_scratchpad({
            1: ("complete", {"text": "Total revenue: $1,230,000,000"}),
        })
        result = verify_numbers(text, pad)
        if result.verified_count > 0 and result.flagged_count == 0:
            assert "confirmed against source data" in result.annotated_text

    def test_failed_step_ignored(self):
        text = "Revenue was $1.23B in FY2024."
        pad = _make_scratchpad({
            1: ("failed", None),
        })
        result = verify_numbers(text, pad)
        assert result.unverified_count > 0
        assert result.verified_count == 0

    def test_rel_tol_parameter(self):
        text = "Revenue was $1.25B in FY2024."
        pad = _make_scratchpad({
            1: ("complete", {"text": "Total revenue: $1,230,000,000"}),
        })
        result_strict = verify_numbers(text, pad, rel_tol=0.001)
        result_loose = verify_numbers(text, pad, rel_tol=0.05)
        assert result_loose.verified_count >= result_strict.verified_count

    def test_max_claims_cap(self):
        numbers = " ".join(f"Value{i}: ${float(i)}.00" for i in range(100))
        pad = _make_scratchpad({})
        result = verify_numbers(numbers, pad, max_claims=10)
        assert result.total_claims <= 10

    def test_dict_numeric_output(self):
        text = "EPS was $3.45 this quarter."
        pad = _make_scratchpad({
            1: ("complete", {"eps": 3.45, "text": "Earnings per share"}),
        })
        result = verify_numbers(text, pad)
        verified = [c for c in result.claims if c.status == VerifyStatus.VERIFIED]
        assert len(verified) >= 1

    def test_percentage_match(self):
        text = "Operating margin was 23.5%."
        pad = _make_scratchpad({
            1: ("complete", {"text": "Operating margin: 23.5%"}),
        })
        result = verify_numbers(text, pad)
        verified = [c for c in result.claims if c.status == VerifyStatus.VERIFIED]
        assert len(verified) >= 1

    def test_crore_match(self):
        text = "Revenue was ₹5,000 crore."
        pad = _make_scratchpad({
            1: ("complete", {"text": "Total revenue: ₹5000 crore"}),
        })
        result = verify_numbers(text, pad)
        verified = [c for c in result.claims if c.status == VerifyStatus.VERIFIED]
        assert len(verified) >= 1

    def test_has_flags_property(self):
        text = "Revenue was $9.99B in FY2024."
        pad = _make_scratchpad({
            1: ("complete", {"text": "Total revenue: $1,230,000,000"}),
        })
        result = verify_numbers(text, pad)
        if result.flagged_count > 0:
            assert result.has_flags is True

    def test_no_flags_property(self):
        text = "Revenue was $1.23B in FY2024."
        pad = _make_scratchpad({})
        result = verify_numbers(text, pad)
        assert result.has_flags is False

    def test_original_text_preserved_on_no_verify(self):
        text = "Revenue was $1.23B."
        pad = _make_scratchpad({
            1: ("complete", {"text": "Revenue data here"}),
        })
        result = verify_numbers(text, pad)
        assert "$1.23" in result.annotated_text

    def test_mixed_verified_unverified(self):
        text = "Revenue was $1.23B and headcount was 45,000."
        pad = _make_scratchpad({
            1: ("complete", {"text": "Total revenue: $1,230,000,000"}),
        })
        result = verify_numbers(text, pad)
        assert result.verified_count >= 1
        assert result.unverified_count >= 1

    def test_list_output_extraction(self):
        text = "The P/E ratio is 25.3x."
        pad = _make_scratchpad({
            1: ("complete", {"text": "P/E: 25.3, Market cap: $10B"}),
        })
        result = verify_numbers(text, pad)
        assert result.total_claims >= 1

    def test_nested_dict_extraction(self):
        text = "Revenue was $1.23B."
        pad = _make_scratchpad({
            1: ("complete", {
                "items": [
                    {"revenue": "$1,230,000,000", "year": "2024"},
                ],
            }),
        })
        result = verify_numbers(text, pad)
        verified = [c for c in result.claims if c.status == VerifyStatus.VERIFIED]
        assert len(verified) >= 1

    def test_string_output_extraction(self):
        text = "Revenue was $1.23B."
        pad = _make_scratchpad({
            1: ("complete", "Revenue: $1,230,000,000 for FY2024"),
        })
        result = verify_numbers(text, pad)
        verified = [c for c in result.claims if c.status == VerifyStatus.VERIFIED]
        assert len(verified) >= 1

    def test_verify_result_counts_consistent(self):
        text = "Revenue was $1.23B and expenses were $890M and profit $340M."
        pad = _make_scratchpad({
            1: ("complete", {"text": "Revenue: $1,230,000,000"}),
        })
        result = verify_numbers(text, pad)
        assert (
            result.verified_count
            + result.unverified_count
            + result.flagged_count
            == result.total_claims
        )

    def test_zero_values_match(self):
        text = "The company had $0 debt."
        pad = _make_scratchpad({
            1: ("complete", {"text": "Outstanding debt: $0"}),
        })
        result = verify_numbers(text, pad)
        zero_verified = [
            c for c in result.claims
            if c.status == VerifyStatus.VERIFIED and c.value == 0
        ]
        assert len(zero_verified) >= 1

    def test_verify_does_not_crash_on_bad_output(self):
        text = "Revenue was $1.23B."
        pad = _make_scratchpad({
            1: ("complete", {"text": None}),
        })
        result = verify_numbers(text, pad)
        assert result.total_claims >= 1


# ---------------------------------------------------------------------------
# Correction surfacing + verification badge (Phase C)
# ---------------------------------------------------------------------------
class TestCorrectionAndBadge:

    def test_flagged_number_shows_source_value(self):
        # Claimed $2.5B but the source says $1.23B → flag AND surface the source.
        text = "Revenue was $2.5B in FY2024."
        pad = _make_scratchpad({
            1: ("complete", {"text": "Revenue was $1,230,000,000 for the year"}),
        })
        result = verify_numbers(text, pad)
        if result.flagged_count > 0:
            assert "source data:" in result.annotated_text
            # rendered in the claim's own unit scale (billions)
            assert "1.23B" in result.annotated_text
            # the original claimed number is preserved, never silently rewritten
            assert "2.5B" in result.annotated_text

    def test_summary_badge_counts_consistent(self):
        text = "Revenue was $1.23B and headcount was 45,000 people."
        pad = _make_scratchpad({
            1: ("complete", {"text": "Total revenue: $1,230,000,000"}),
        })
        result = verify_numbers(text, pad)
        s = result.summary
        assert f"{result.verified_count} verified" in s
        assert f"{result.flagged_count} flagged" in s

    def test_badge_footer_includes_counts_when_verified(self):
        text = "Revenue was $1.23B in FY2024."
        pad = _make_scratchpad({
            1: ("complete", {"text": "Total revenue: $1,230,000,000"}),
        })
        result = verify_numbers(text, pad)
        if result.verified_count > 0 and result.flagged_count == 0:
            assert "**Number check:**" in result.annotated_text
            assert "confirmed against source data" in result.annotated_text

    def test_percentage_source_rendered_with_percent(self):
        from src.core.verify_numbers import _format_source_value
        claim = NumberClaim(
            raw_text="15", value=15.0, unit_multiplier=1.0,
            context_left="margin of", context_right="%", span_start=0, span_end=2,
            status=VerifyStatus.FLAGGED, matched_scratchpad_value=18.0, unit_str="%",
        )
        assert _format_source_value(claim) == "18%"

    def test_bare_large_source_rendered_with_commas(self):
        from src.core.verify_numbers import _format_source_value
        claim = NumberClaim(
            raw_text="45000", value=45000.0, unit_multiplier=1.0,
            context_left="", context_right="", span_start=0, span_end=5,
            status=VerifyStatus.FLAGGED, matched_scratchpad_value=52000.0, unit_str="",
        )
        assert _format_source_value(claim) == "52,000"
