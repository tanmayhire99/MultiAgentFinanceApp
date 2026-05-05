from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import statistics

# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

# =========================================================
# CONFIG
# =========================================================

MODEL = "deepseek-ai/deepseek-v4-pro"

PROMPT = """
Explain how a multi-agent orchestration system using MCP servers,
RAG memory, and tool routing works in practice.
Keep it concise.
"""

RUNS_PER_KEY = 3

API_KEYS = [
    os.getenv("NVIDIA_API_KEY"),
    os.getenv("NVIDIA_API_KEY_1"),
    os.getenv("NVIDIA_API_KEY_2"),
    os.getenv("NVIDIA_API_KEY_3"),
    os.getenv("NVIDIA_API_KEY_4"),
]

API_KEYS = [k for k in API_KEYS if k]

BASE_URL = "https://integrate.api.nvidia.com/v1"

# =========================================================
# SINGLE REQUEST
# =========================================================

def test_request(api_key, key_name, run_number):

    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key
    )

    try:

        start = time.perf_counter()

        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT
                }
            ],
            temperature=0.7,
            top_p=1,
            max_tokens=512,
            stream=True,
            extra_body={
                "chat_template_kwargs": {
                    "thinking": False
                    #"clear_thinking": False
                }
            }
        )

        first_token_time = None
        token_count = 0

        for chunk in stream:

            if not getattr(chunk, "choices", None):
                continue

            if len(chunk.choices) == 0:
                continue

            delta = chunk.choices[0].delta

            reasoning = getattr(delta, "reasoning_content", None)
            content = getattr(delta, "content", None)

            if reasoning or content:

                if first_token_time is None:
                    first_token_time = time.perf_counter()

                if reasoning:
                    token_count += len(reasoning.split())

                if content:
                    token_count += len(content.split())

        end = time.perf_counter()

        if first_token_time is None:
            return {
                "success": False,
                "key": key_name,
                "run": run_number,
                "error": "No tokens received"
            }

        ttft = first_token_time - start
        total = end - start
        gen_time = end - first_token_time

        tps = token_count / gen_time if gen_time > 0 else 0

        return {
            "success": True,
            "key": key_name,
            "run": run_number,
            "ttft": round(ttft, 3),
            "total": round(total, 3),
            "tps": round(tps, 2),
            "tokens": token_count
        }

    except Exception as e:

        return {
            "success": False,
            "key": key_name,
            "run": run_number,
            "error": str(e)
        }

# =========================================================
# MAIN
# =========================================================

def main():

    print("\n======================================================")
    print(f"PARALLEL TTFT TEST | MODEL: {MODEL}")
    print("======================================================\n")

    futures = []
    results = []

    # total parallel requests
    total_requests = len(API_KEYS) * RUNS_PER_KEY

    print(f"Running {total_requests} requests in parallel...\n")

    with ThreadPoolExecutor(max_workers=total_requests) as executor:

        for idx, key in enumerate(API_KEYS):

            key_name = f"KEY-{idx}"

            for run in range(1, RUNS_PER_KEY + 1):

                futures.append(
                    executor.submit(
                        test_request,
                        key,
                        key_name,
                        run
                    )
                )

        for future in as_completed(futures):

            result = future.result()
            results.append(result)

            if result["success"]:

                print(
                    f"[{result['key']} | RUN {result['run']}] "
                    f"TTFT={result['ttft']}s | "
                    f"TOTAL={result['total']}s | "
                    f"TPS={result['tps']}"
                )

            else:

                print(
                    f"[{result['key']} | RUN {result['run']}] "
                    f"ERROR: {result['error']}"
                )

    # =====================================================
    # SUMMARY
    # =====================================================

    successful = [r for r in results if r["success"]]

    if not successful:
        print("\nNo successful runs.")
        return

    ttfts = [r["ttft"] for r in successful]
    totals = [r["total"] for r in successful]
    tps_vals = [r["tps"] for r in successful]

    print("\n======================================================")
    print("SUMMARY")
    print("======================================================")

    print(f"Successful Requests : {len(successful)}")
    print(f"Average TTFT        : {statistics.mean(ttfts):.3f}s")
    print(f"Median TTFT         : {statistics.median(ttfts):.3f}s")
    print(f"Min TTFT            : {min(ttfts):.3f}s")
    print(f"Max TTFT            : {max(ttfts):.3f}s")
    print(f"Average Total Time  : {statistics.mean(totals):.3f}s")
    print(f"Average TPS         : {statistics.mean(tps_vals):.2f}")

    print("======================================================\n")

if __name__ == "__main__":
    main()