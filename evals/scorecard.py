#!/usr/bin/env python3
"""FinBen-style scorecard runner + regression gate.

Two responsibilities, split so the *gate* can run anywhere while the
*benchmark* only runs where an API key + network are available:

1. ``run``   — evaluate the configured model on MMLU-Finance subsets via the
   NVIDIA API and write a scorecard JSON.
   Needs ``NVIDIA_API_KEY`` + network + the ``datasets`` package.

2. ``check`` — compare a candidate scorecard against the committed baseline
   and exit non-zero if quality regressed beyond tolerance.
   **Pure JSON diff** — no API key, no network — so CI / pre-commit can gate
   on it cheaply.

Usage::

    python evals/scorecard.py run   --out evals/results/latest.json --n 15
    python evals/scorecard.py check --candidate evals/results/latest.json \
                                    --baseline  evals/results/baseline.json

Exit codes: ``check`` returns 0 when the candidate holds the line, 1 on a
regression beyond tolerance, 2 on a usage/IO error.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

# Default MMLU subsets that stand in for FinBen's knowledge slice.
FINANCE_SUBSETS = [
    "business_ethics",
    "econometrics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "marketing",
    "professional_accounting",
]

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Regression tolerances (accuracy is a 0..1 fraction).
#   overall: a 5-point drop in aggregate accuracy fails the gate.
#   per-task: a 10-point drop on any single task fails (N is small per task,
#             so individual tasks are noisier than the aggregate).
DEFAULT_OVERALL_TOL = 0.05
DEFAULT_TASK_TOL = 0.10

SYSTEM_PROMPT = (
    "You are a financial knowledge expert. For each multiple-choice question, "
    "think step by step, then write ONLY the letter of the correct answer "
    "(A, B, C, or D) on the very last line of your response."
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested; no network)
# ---------------------------------------------------------------------------
def format_question(sample: Dict) -> str:
    q = sample["question"]
    opts = "\n".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(sample["choices"]))
    return f"Question: {q}\n\n{opts}"


def extract_letter(text: Optional[str]) -> Optional[str]:
    """Pull the answer letter (A-F) from a model response."""
    if not text:
        return None
    m = re.search(r"\b([A-F])\s*$", text.strip())
    if m:
        return m.group(1)
    m = re.search(r"(?:answer is|answer:)\s*([A-F])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def compare_scorecards(
    candidate: Dict,
    baseline: Dict,
    *,
    overall_tol: float = DEFAULT_OVERALL_TOL,
    task_tol: float = DEFAULT_TASK_TOL,
) -> Tuple[bool, List[str]]:
    """Compare a candidate scorecard against the baseline.

    Returns ``(passed, report_lines)``. ``passed`` is False when the overall
    accuracy drops by more than ``overall_tol``, when any baseline task drops
    by more than ``task_tol``, or when a baseline task is missing from the
    candidate (a structural regression).
    """
    lines: List[str] = []
    regressions: List[str] = []

    base_overall = float(baseline.get("overall_accuracy", 0.0))
    cand_overall = float(candidate.get("overall_accuracy", 0.0))
    overall_delta = cand_overall - base_overall

    lines.append(
        f"{'Task':<32} {'baseline':>9} {'candidate':>10} {'delta':>8}"
    )
    lines.append("-" * 62)

    base_tasks = baseline.get("tasks", {}) or {}
    cand_tasks = candidate.get("tasks", {}) or {}

    for task in sorted(base_tasks):
        b = float(base_tasks[task].get("accuracy", 0.0))
        if task not in cand_tasks:
            lines.append(f"{task:<32} {b:>8.1%} {'MISSING':>10} {'--':>8}")
            regressions.append(f"task '{task}' missing from candidate scorecard")
            continue
        c = float(cand_tasks[task].get("accuracy", 0.0))
        delta = c - b
        flag = "  <-- REGRESSION" if (b - c) > task_tol else ""
        lines.append(f"{task:<32} {b:>8.1%} {c:>9.1%} {delta:>+7.1%}{flag}")
        if (b - c) > task_tol:
            regressions.append(
                f"task '{task}' dropped {b - c:.1%} (> {task_tol:.0%} tolerance)"
            )

    lines.append("-" * 62)
    overall_flag = "  <-- REGRESSION" if -overall_delta > overall_tol else ""
    lines.append(
        f"{'OVERALL':<32} {base_overall:>8.1%} {cand_overall:>9.1%} "
        f"{overall_delta:>+7.1%}{overall_flag}"
    )
    if -overall_delta > overall_tol:
        regressions.append(
            f"overall accuracy dropped {-overall_delta:.1%} "
            f"(> {overall_tol:.0%} tolerance)"
        )

    lines.append("")
    if regressions:
        lines.append("RESULT: FAIL")
        for r in regressions:
            lines.append(f"  - {r}")
    else:
        lines.append("RESULT: PASS (within tolerance)")

    return (not regressions, lines)


# ---------------------------------------------------------------------------
# Benchmark runner (needs NVIDIA_API_KEY + network + `datasets`)
# ---------------------------------------------------------------------------
async def _eval_one(client, sample, sem) -> bool:
    import asyncio

    async with sem:
        true_idx = sample["answer"]
        prompt = format_question(sample)
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=256,
                    temperature=0,
                )
                msg = r.choices[0].message
                content = msg.content or ""
                reasoning = getattr(msg, "reasoning", "") or ""
                full = (reasoning + "\n" + content).strip() if reasoning else content
                pred = extract_letter(full)
                pred_idx = ord(pred) - 65 if pred else -1
                return pred_idx == true_idx
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(2)
                else:
                    return False
    return False


async def _run_async(n: int, model: str, concurrency: int) -> Dict:
    import asyncio

    from datasets import load_dataset
    from openai import AsyncOpenAI

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set; cannot run the benchmark")

    client = AsyncOpenAI(base_url=DEFAULT_BASE_URL, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    results: Dict[str, Dict] = {}
    total_c = total_n = 0
    for name in FINANCE_SUBSETS:
        ds = load_dataset("cais/mmlu", name, split="test")
        k = min(n, len(ds))
        outcomes = await asyncio.gather(
            *(_eval_one(client, ds[i], sem) for i in range(k))
        )
        correct = sum(outcomes)
        results[name] = {
            "correct": correct,
            "total": k,
            "accuracy": round(correct / k, 4) if k else 0.0,
        }
        total_c += correct
        total_n += k
        print(f"  {name}: {results[name]['accuracy']:.1%} ({correct}/{k})", flush=True)

    return {
        "model": model,
        "benchmark": "MMLU-Finance",
        "n_per_subset": n,
        "total_samples": total_n,
        "overall_accuracy": round(total_c / total_n, 4) if total_n else 0.0,
        "tasks": results,
    }


def run_benchmark(n: int = 15, model: str = DEFAULT_MODEL, concurrency: int = 5) -> Dict:
    import asyncio

    return asyncio.run(_run_async(n=n, model=model, concurrency=concurrency))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cmd_run(args: argparse.Namespace) -> int:
    print(f"Running MMLU-Finance scorecard: model={args.model} n={args.n}/subset")
    scorecard = run_benchmark(n=args.n, model=args.model, concurrency=args.concurrency)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    print(f"\nOVERALL: {scorecard['overall_accuracy']:.1%} "
          f"({scorecard['total_samples']} samples) -> {args.out}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    try:
        candidate = _load_json(args.candidate)
        baseline = _load_json(args.baseline)
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: could not load scorecards: {e}", file=sys.stderr)
        return 2

    passed, lines = compare_scorecards(
        candidate, baseline, overall_tol=args.overall_tol, task_tol=args.task_tol
    )
    print("\n".join(lines))
    return 0 if passed else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the benchmark and write a scorecard")
    p_run.add_argument("--out", default="evals/results/latest.json")
    p_run.add_argument("--n", type=int, default=15, help="samples per subset")
    p_run.add_argument("--model", default=DEFAULT_MODEL)
    p_run.add_argument("--concurrency", type=int, default=5)
    p_run.set_defaults(func=cmd_run)

    p_check = sub.add_parser("check", help="gate a candidate scorecard vs baseline")
    p_check.add_argument("--candidate", default="evals/results/latest.json")
    p_check.add_argument("--baseline", default="evals/results/baseline.json")
    p_check.add_argument("--overall-tol", type=float, default=DEFAULT_OVERALL_TOL,
                         dest="overall_tol")
    p_check.add_argument("--task-tol", type=float, default=DEFAULT_TASK_TOL,
                         dest="task_tol")
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
