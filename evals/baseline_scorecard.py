#!/usr/bin/env python3
"""
FinBen baseline evaluation — MMLU Financial Subsets.
Evaluates openai/gpt-oss-120b via NVIDIA API with CoT prompting.
"""

import os, json, re, time, asyncio
from openai import AsyncOpenAI
from datasets import load_dataset

FINANCE_SUBSETS = [
    'business_ethics', 'econometrics',
    'high_school_macroeconomics', 'high_school_microeconomics',
    'marketing', 'professional_accounting'
]

SYSTEM_PROMPT = (
    "You are a financial knowledge expert. For each multiple-choice question, "
    "think step by step, then write ONLY the letter of the correct answer "
    "(A, B, C, or D) on the very last line of your response."
)

def format_question(sample):
    q = sample['question']
    opts = '\n'.join(f"{chr(65+i)}) {c}" for i, c in enumerate(sample['choices']))
    return f"Question: {q}\n\n{opts}"

def extract_letter(text):
    if not text:
        return None
    m = re.search(r'\b([A-F])\s*$', text.strip())
    if m:
        return m.group(1)
    m = re.search(r'(?:answer is|answer:)\s*([A-F])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None

async def eval_one(client, sample, sem):
    async with sem:
        true_idx = sample['answer']
        true_letter = chr(65 + true_idx)
        prompt = format_question(sample)
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=256,
                    temperature=0
                )
                content = r.choices[0].message.content or ""
                reasoning = r.choices[0].message.reasoning or ""
                full = (reasoning + "\n" + content).strip() if reasoning else content
                pred = extract_letter(full)
                pred_idx = ord(pred) - 65 if pred else -1
                return pred_idx == true_idx
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2)
                else:
                    return False

async def main():
    print("=" * 60)
    print("FinBen Baseline — MMLU Financial Subsets")
    print("Model: openai/gpt-oss-120b (NVIDIA API)")
    print("=" * 60)

    datasets = {}
    for s in FINANCE_SUBSETS:
        datasets[s] = load_dataset('cais/mmlu', s, split='test')
        print(f"  {s}: {len(datasets[s])} samples")

    api_key = os.getenv('NVIDIA_API_KEY')
    client = AsyncOpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    sem = asyncio.Semaphore(5)  # 5 concurrent requests

    N = 15  # per subset
    results = {}
    total_c, total_n = 0, 0

    for name, ds in datasets.items():
        n = min(N, len(ds))
        print(f"\n--- {name} ({n} samples) ---", flush=True)
        tasks = [eval_one(client, ds[i], sem) for i in range(n)]
        outcomes = await asyncio.gather(*tasks)
        correct = sum(outcomes)
        acc = correct / n
        results[name] = {'correct': correct, 'total': n, 'accuracy': round(acc, 4)}
        total_c += correct
        total_n += n
        print(f"  => {acc:.1%} ({correct}/{n})", flush=True)

    overall = total_c / total_n if total_n else 0
    output = {
        'model': 'openai/gpt-oss-120b',
        'benchmark': 'MMLU-Finance',
        'n_per_subset': N,
        'total_samples': total_n,
        'overall_accuracy': round(overall, 4),
        'tasks': results,
    }

    os.makedirs('evals/results', exist_ok=True)
    with open('evals/results/baseline.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("BASELINE SCORECARD")
    print("=" * 60)
    print(f"{'Task':<35} {'Acc':>8} {'N':>5}")
    print("-" * 50)
    for t, r in results.items():
        print(f"{t:<35} {r['accuracy']:>7.1%} {r['total']:>5}")
    print("-" * 50)
    print(f"{'OVERALL':<35} {overall:>7.1%} {total_n:>5}")
    print(f"\nSaved: evals/results/baseline.json")

if __name__ == "__main__":
    asyncio.run(main())
