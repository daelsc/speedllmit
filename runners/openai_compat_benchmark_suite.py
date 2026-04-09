#!/usr/bin/env python3
"""Benchmark runner for OpenAI-compatible endpoints (vLLM, sglang, llama.cpp, Ollama, etc.)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from openai import AsyncOpenAI
except ImportError:
    print("openai package required: pip install openai", file=sys.stderr)
    sys.exit(1)

RUNNER_VERSION = "0.2.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark any OpenAI-compatible inference endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # endpoint
    parser.add_argument("--api-base", required=True, help="Base URL, e.g. http://localhost:8027/v1")

    # runtime metadata
    parser.add_argument("--runtime", required=True, help="Runtime name, e.g. vllm, sglang, llama.cpp, mlx, ollama")
    parser.add_argument("--runtime-version", default=None, help="Runtime version string, e.g. 0.19.0")

    # model metadata
    parser.add_argument("--model", required=True, help="Model name as served, e.g. gemma-4-31b")
    parser.add_argument("--model-dtype", default=None, help="Weight dtype, e.g. bfloat16, float16, int4")
    parser.add_argument("--model-quant", default=None, help="Quantization scheme, e.g. AWQ, GPTQ, Q4_K_M")
    parser.add_argument("--model-tp", type=int, default=None, help="Tensor parallel size")
    parser.add_argument("--model-max-ctx", type=int, default=None, help="Configured max context length")

    # hardware metadata
    parser.add_argument(
        "--hw-machine", default=None,
        help="Full machine accelerator inventory, e.g. '8x A100-SXM4-80GB'. "
             "Documents what the machine has, not what this run uses.",
    )
    parser.add_argument(
        "--hw-gpus-used", type=int, default=None,
        help="Number of GPUs actually used for this run. Defaults to --model-tp if set.",
    )

    # benchmark inputs
    parser.add_argument("--prompts", default="benchmark_prompts_50.json", help="Prompt corpus JSON")
    parser.add_argument("--spec", default="benchmark_spec.json", help="Benchmark spec JSON")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--max-per-category", type=int, default=3,
        help="Max prompts per category to run (default 3). Use 0 for all.",
    )
    parser.add_argument("--output-json", default="benchmark_results.json")
    parser.add_argument("--categories", nargs="*")
    parser.add_argument("--ids", nargs="*")

    # concurrency
    conc_group = parser.add_mutually_exclusive_group()
    conc_group.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of simultaneous requests (default 1 = serial)",
    )
    conc_group.add_argument(
        "--concurrency-sweep", default=None,
        help="Comma-separated concurrency levels to run in sequence, e.g. 1,4,8,16",
    )

    return parser.parse_args()


def load_prompts(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        prompts = json.load(f)
    for item in prompts:
        for key in ("id", "category", "max_tokens", "prompt"):
            if key not in item:
                raise ValueError(f"Missing key {key!r} in prompt item: {item}")
    return prompts


def load_spec(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def filter_prompts(prompts: list, categories, ids, max_per_category: int) -> list:
    filtered = []
    for item in prompts:
        if categories and item["category"] not in categories:
            continue
        if ids and item["id"] not in ids:
            continue
        filtered.append(item)

    if max_per_category:
        by_cat: dict = {}
        for item in filtered:
            by_cat.setdefault(item["category"], []).append(item)
        filtered = []
        for cat_items in by_cat.values():
            filtered.extend(cat_items[:max_per_category])

    return filtered


def get_git_commit():
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    except OSError:
        return None
    return proc.stdout.strip() or None


def get_host_metadata() -> dict:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cwd": str(Path.cwd()),
        "user": os.environ.get("USER"),
    }


def pct(data: list, p: int):
    """Return the p-th percentile of data (p in 0–100)."""
    if not data:
        return None
    s = sorted(data)
    idx = max(0, min(int(len(s) * p / 100), len(s) - 1))
    return round(s[idx], 3)


async def run_case(client: AsyncOpenAI, model: str, prompt_item: dict) -> dict:
    start = time.perf_counter()
    first_token_time = None
    end_time = None
    prompt_tokens = None
    generation_tokens = None
    error = None

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_item["prompt"]}],
            max_tokens=prompt_item["max_tokens"],
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            now = time.perf_counter()
            if first_token_time is None and chunk.choices and chunk.choices[0].delta.content:
                first_token_time = now
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                generation_tokens = chunk.usage.completion_tokens
        end_time = time.perf_counter()
    except Exception as exc:
        error = str(exc)
        end_time = time.perf_counter()

    elapsed = round(end_time - start, 3) if end_time else None
    ttft_s = None
    generation_tps = None
    prompt_tps_estimated = None

    if first_token_time is not None:
        ttft_s = round(first_token_time - start, 3)
        if end_time is not None and generation_tokens and (end_time - first_token_time) > 0:
            generation_tps = round(generation_tokens / (end_time - first_token_time), 3)
    if prompt_tokens and ttft_s and ttft_s > 0:
        prompt_tps_estimated = round(prompt_tokens / ttft_s, 3)

    return {
        "id": prompt_item["id"],
        "category": prompt_item["category"],
        "max_tokens": prompt_item["max_tokens"],
        "elapsed_s": elapsed,
        "prompt_tokens": prompt_tokens,
        "prompt_tps": prompt_tps_estimated,
        "prompt_tps_source": "estimated_from_ttft" if prompt_tps_estimated is not None else None,
        "generation_tokens": generation_tokens,
        "generation_tps": generation_tps,
        "ttft_s": ttft_s,
        "peak_memory_gb": None,  # not available via API
        "error": error,
    }


async def run_all(
    client: AsyncOpenAI, model: str, prompts: list, concurrency: int, repeats: int
) -> tuple:
    """Run all prompts at the given concurrency. Returns (results, wall_time_s, aggregate_tps)."""
    tasks = [(prompt_item, r) for prompt_item in prompts for r in range(1, repeats + 1)]
    total = len(tasks)
    results = [None] * total
    sem = asyncio.Semaphore(concurrency)

    async def bounded(idx: int, prompt_item: dict, repeat: int) -> None:
        async with sem:
            result = await run_case(client, model, prompt_item)
            result["repeat"] = repeat
            results[idx] = result
            print(
                f"[{idx + 1}/{total}] {prompt_item['id']} cat={prompt_item['category']} "
                f"max_tokens={prompt_item['max_tokens']} repeat={repeat} "
                f"gen_tps={result['generation_tps']} ttft_s={result['ttft_s']} "
                f"gen_tokens={result['generation_tokens']}"
                + (f" ERROR={result['error']}" if result["error"] else ""),
                flush=True,
            )

    wall_start = time.perf_counter()
    await asyncio.gather(*[bounded(i, p, r) for i, (p, r) in enumerate(tasks)])
    wall_time = round(time.perf_counter() - wall_start, 3)

    total_gen_tokens = sum(r["generation_tokens"] for r in results if r and r["generation_tokens"])
    aggregate_tps = round(total_gen_tokens / wall_time, 3) if wall_time > 0 else None

    return results, wall_time, aggregate_tps


def build_summary(results: list, concurrency: int, wall_time: float, aggregate_tps) -> dict:
    groups: dict = {}
    for r in results:
        key = (r["category"], r["max_tokens"])
        groups.setdefault(key, []).append(r)

    per_category = []
    for (category, max_tokens), rows in sorted(groups.items()):
        gen_tps = [r["generation_tps"] for r in rows if r["generation_tps"] is not None]
        ttft = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
        emitted = [r["generation_tokens"] for r in rows if r["generation_tokens"] is not None]

        entry: dict = {
            "category": category,
            "max_tokens": max_tokens,
            "runs": len(rows),
            "avg_generation_tps": round(statistics.mean(gen_tps), 3) if gen_tps else None,
            "avg_ttft_s": round(statistics.mean(ttft), 3) if ttft else None,
            "avg_generation_tokens": round(statistics.mean(emitted), 1) if emitted else None,
        }
        if concurrency > 1:
            entry.update({
                "p50_generation_tps": pct(gen_tps, 50),
                "p95_generation_tps": pct(gen_tps, 95),
                "p50_ttft_s": pct(ttft, 50),
                "p95_ttft_s": pct(ttft, 95),
                "p99_ttft_s": pct(ttft, 99),
            })
        per_category.append(entry)

    return {
        "concurrency": concurrency,
        "wall_time_s": wall_time,
        "aggregate_generation_tps": aggregate_tps,
        "per_category": per_category,
    }


def print_summary(s: dict) -> None:
    print(
        f"\nconcurrency={s['concurrency']}  "
        f"wall={s['wall_time_s']}s  "
        f"aggregate_tps={s['aggregate_generation_tps']}"
    )
    for row in s["per_category"]:
        line = (
            f"  {row['category']:>14} max_tokens={row['max_tokens']:<4} "
            f"avg_gen_tps={row['avg_generation_tps']} "
            f"avg_ttft_s={row['avg_ttft_s']} "
            f"avg_gen_tokens={row['avg_generation_tokens']}"
        )
        if s["concurrency"] > 1:
            line += (
                f"  p95_gen_tps={row.get('p95_generation_tps')} "
                f"p95_ttft_s={row.get('p95_ttft_s')}"
            )
        print(line)


async def main_async() -> int:
    args = parse_args()
    prompts = load_prompts(Path(args.prompts))
    spec = load_spec(Path(args.spec))
    prompts = filter_prompts(
        prompts,
        set(args.categories) if args.categories else None,
        set(args.ids) if args.ids else None,
        args.max_per_category,
    )
    if not prompts:
        print("No prompts selected.", file=sys.stderr)
        return 1

    gpus_used = args.hw_gpus_used if args.hw_gpus_used is not None else args.model_tp

    if args.concurrency_sweep:
        levels = [int(x.strip()) for x in args.concurrency_sweep.split(",")]
    else:
        levels = [args.concurrency]

    client = AsyncOpenAI(api_key="local", base_url=args.api_base)

    all_summaries = []
    all_results = []

    for level in levels:
        print(f"\n--- concurrency={level} ({len(prompts)} prompts × {args.repeats} repeat(s)) ---", flush=True)
        results, wall_time, aggregate_tps = await run_all(client, args.model, prompts, level, args.repeats)
        summary = build_summary(results, level, wall_time, aggregate_tps)
        all_summaries.append(summary)
        all_results.extend(results)

    payload = {
        "benchmark": spec,
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
            "host": get_host_metadata(),
            "api_base": args.api_base,
            "runner_version": RUNNER_VERSION,
        },
        "runtime": {
            "name": args.runtime,
            "version": args.runtime_version,
        },
        "model": {
            "served_name": args.model,
            "dtype": args.model_dtype,
            "quant": args.model_quant,
            "tensor_parallel": args.model_tp,
            "max_context": args.model_max_ctx,
        },
        "hardware": {
            "machine": args.hw_machine,
            "gpus_used": gpus_used,
            "hostname": platform.node(),
        },
        "prompt_file": str(Path(args.prompts).resolve()),
        "spec_file": str(Path(args.spec).resolve()),
        "prompts_per_category": args.max_per_category or "all",
        "repeats": args.repeats,
        "results": all_results,
        "summaries": all_summaries,
    }

    output_path = Path(args.output_json)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== Summary ===")
    print(
        f"runtime={args.runtime} {args.runtime_version}  "
        f"model={args.model}  dtype={args.model_dtype}  tp={args.model_tp}"
    )
    print(
        f"machine={args.hw_machine}  gpus_used={gpus_used}  hostname={platform.node()}"
    )
    print(
        f"prompts={len(prompts)} ({args.max_per_category or 'all'}/category)  repeats={args.repeats}"
    )
    for s in all_summaries:
        print_summary(s)

    print(f"\nWrote {output_path}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
