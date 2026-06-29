# Evaluation & regression gate

Two layers protect quality so prompt/model/pipeline changes can't silently
regress:

| Layer | What | Where it runs | Needs a key? |
|---|---|---|---|
| **Offline gate** | `verify_numbers` numeric-accuracy cases + `compare_scorecards` regression logic | every push / PR (`.github/workflows/ci.yml`, via `tests/test_eval_gate.py`) | no |
| **FinBen gate** | MMLU-Finance accuracy vs committed baseline | nightly + on-demand (`.github/workflows/eval-finben.yml`) | yes (`NVIDIA_API_KEY`) |

## `scorecard.py`

```bash
# Run the benchmark (needs NVIDIA_API_KEY + network + `datasets`)
python evals/scorecard.py run --out evals/results/latest.json --n 15

# Gate a candidate scorecard against the baseline (pure JSON diff, no key)
python evals/scorecard.py check \
  --candidate evals/results/latest.json \
  --baseline  evals/results/baseline.json
```

`check` exits **1** on regression, **0** within tolerance. Tolerances
(overridable): overall `--overall-tol 0.05` (5 pts), per-task `--task-tol 0.10`
(10 pts — per-task N is small and noisier).

## The baseline

`evals/results/baseline.json` is the committed reference (currently **71.1%**
overall on `openai/gpt-oss-120b`, MMLU-Finance, 90 samples). To **intentionally**
move the baseline (e.g. after a model upgrade), run `run`, eyeball the scorecard,
then copy it over `baseline.json` in a dedicated commit so the change is
reviewable.

## CI behavior

- **CI** (`ci.yml`): installs the light deps and runs `pytest`. The retrieval
  tests auto-skip if `sentence-transformers` isn't present, keeping this lane
  fast. The offline eval gate (`tests/test_eval_gate.py`) runs here.
- **FinBen** (`eval-finben.yml`): nightly / manual. Skips cleanly if the
  `NVIDIA_API_KEY` secret isn't configured; otherwise runs the scorecard and
  fails the job on a regression, uploading the scorecard as an artifact.

> `evals/finlm_eval/` (gitignored) is a vendored full FinBen/FinLM harness kept
> for deeper, manual runs; the gate above intentionally uses the lightweight
> MMLU-Finance slice so it's cheap enough to run nightly.
