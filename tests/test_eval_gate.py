"""Offline eval gate — runs in CI with no API key / network.

Two things are guarded here:

1. ``evals.scorecard.compare_scorecards`` — the regression gate that the
   FinBen workflow uses to fail a build when model/prompt changes drop
   accuracy. We test the comparison logic directly with synthetic scorecards.
2. A small **numeric-accuracy** eval over ``src.core.verify_numbers``: numbers
   backed by scratchpad data must stay VERIFIED and hallucinated numbers must
   NOT be silently verified. This is the deterministic, always-on half of the
   numeric-reasoning gate (the full QRData/FinBen half runs nightly via NIM).
"""
from __future__ import annotations

import time
import unittest

from evals.scorecard import compare_scorecards, extract_letter, format_question
from src.core.types import Scratchpad, StepResult
from src.core.verify_numbers import verify_numbers


# ---------------------------------------------------------------------------
# compare_scorecards — the regression gate
# ---------------------------------------------------------------------------
_BASELINE = {
    "overall_accuracy": 0.71,
    "tasks": {
        "business_ethics": {"accuracy": 0.80},
        "econometrics": {"accuracy": 0.60},
        "professional_accounting": {"accuracy": 0.40},
    },
}


def _card(overall, **tasks):
    return {"overall_accuracy": overall, "tasks": {k: {"accuracy": v} for k, v in tasks.items()}}


class CompareScorecardTests(unittest.TestCase):
    def test_identical_passes(self):
        passed, _ = compare_scorecards(_BASELINE, _BASELINE)
        self.assertTrue(passed)

    def test_improvement_passes(self):
        better = _card(0.85, business_ethics=0.9, econometrics=0.8, professional_accounting=0.6)
        passed, _ = compare_scorecards(better, _BASELINE)
        self.assertTrue(passed)

    def test_small_drop_within_tolerance_passes(self):
        # overall -3pts, each task within 10pts → pass
        cand = _card(0.68, business_ethics=0.73, econometrics=0.55, professional_accounting=0.35)
        passed, _ = compare_scorecards(cand, _BASELINE)
        self.assertTrue(passed)

    def test_overall_regression_fails(self):
        cand = _card(0.60, business_ethics=0.80, econometrics=0.60, professional_accounting=0.40)
        passed, report = compare_scorecards(cand, _BASELINE)
        self.assertFalse(passed)
        self.assertIn("FAIL", "\n".join(report))

    def test_per_task_regression_fails(self):
        # overall barely moves but one task craters by 25pts
        cand = _card(0.69, business_ethics=0.55, econometrics=0.60, professional_accounting=0.40)
        passed, report = compare_scorecards(cand, _BASELINE)
        self.assertFalse(passed)
        self.assertTrue(any("business_ethics" in ln for ln in report))

    def test_missing_task_fails(self):
        cand = _card(0.75, business_ethics=0.80, econometrics=0.60)  # no professional_accounting
        passed, _ = compare_scorecards(cand, _BASELINE)
        self.assertFalse(passed)

    def test_custom_tolerance(self):
        cand = _card(0.66, business_ethics=0.80, econometrics=0.60, professional_accounting=0.40)
        # 5pt overall drop fails at default 5% (strictly >), passes at 10%
        self.assertFalse(compare_scorecards(cand, _BASELINE, overall_tol=0.04)[0])
        self.assertTrue(compare_scorecards(cand, _BASELINE, overall_tol=0.10)[0])


class ScorecardParsingTests(unittest.TestCase):
    def test_extract_letter_last_line(self):
        self.assertEqual(extract_letter("Reasoning...\nThe answer is clearly\nC"), "C")

    def test_extract_letter_answer_phrase(self):
        self.assertEqual(extract_letter("After analysis, answer: B because ..."), "B")

    def test_extract_letter_none(self):
        self.assertIsNone(extract_letter("no letter here"))
        self.assertIsNone(extract_letter(""))

    def test_format_question(self):
        s = {"question": "2+2?", "choices": ["3", "4", "5", "6"], "answer": 1}
        out = format_question(s)
        self.assertIn("Question: 2+2?", out)
        self.assertIn("A) 3", out)
        self.assertIn("D) 6", out)


# ---------------------------------------------------------------------------
# Numeric-accuracy eval over verify_numbers (deterministic, no LLM)
# ---------------------------------------------------------------------------
def _scratch_with_fundamentals() -> Scratchpad:
    sp = Scratchpad(query="analyze NVDA")
    sp.add(StepResult(
        step_id=1,
        status="complete",
        output={
            "ticker": "NVDA",
            "text": "NVDA last traded at $404.00 with a P/E of 38.2x.",
            "fundamentals": {"price": 404.0, "pe_ratio": 38.2},
        },
        started_at=time.time(),
        completed_at=time.time(),
    ))
    return sp


class NumericAccuracyGateTests(unittest.TestCase):
    def test_number_backed_by_scratchpad_is_verified(self):
        sp = _scratch_with_fundamentals()
        r = verify_numbers("NVDA is trading at $404.00 today.", sp)
        self.assertEqual(r.verified_count, 1)
        self.assertEqual(r.flagged_count, 0)

    def test_pe_ratio_is_verified(self):
        sp = _scratch_with_fundamentals()
        r = verify_numbers("Its P/E ratio sits at 38.2x.", sp)
        self.assertEqual(r.verified_count, 1)

    def test_hallucinated_number_is_not_verified(self):
        # A number absent from the scratchpad must never be marked verified —
        # this is the core hallucination guard.
        sp = _scratch_with_fundamentals()
        r = verify_numbers("Analysts set a price target of $625.00.", sp)
        self.assertEqual(r.verified_count, 0)
        self.assertGreaterEqual(r.total_claims, 1)

    def test_mixed_report_counts(self):
        sp = _scratch_with_fundamentals()
        text = "NVDA trades at $404.00 (P/E 38.2x); a bull case sees $625.00."
        r = verify_numbers(text, sp)
        # the two data-backed numbers verify; the novel target does not
        self.assertEqual(r.verified_count, 2)
        self.assertGreaterEqual(r.unverified_count, 1)

    def test_no_scratchpad_data_means_nothing_verified(self):
        empty = Scratchpad(query="x")
        r = verify_numbers("Revenue grew to $26B this year.", empty)
        self.assertEqual(r.verified_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
