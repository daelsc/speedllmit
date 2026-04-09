#!/usr/bin/env python3
"""Benchmark runner for OpenAI-compatible endpoints (vLLM, sglang, llama.cpp, Ollama, etc.)."""

from __future__ import annotations

import argparse
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
    from openai import OpenAI
except ImportError:
    print("openai package required: pip install openai", file=sys.stderr)
    sys.exit(1)


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
    parser.add_argument("--model-dtype", default=None, help="Model dtype, e.g. bfloat16, float16, int4")
    parser.add_argument("--model-quant", default=None, help="Quantization scheme if applicable, e.g. AWQ, GPTQ, Q4_K_M")
    parser.add_argument("--model-tp", type=int, default=None, help="Tensor parallel size")
    parser.add_argument("--model-max-ctx", type=int, default=None, help="Configured max context length")

    # hardware metadata (free-form; auto-populated from platform but overridable)
    parser.add_argument("--hw-label", default=None, help="Human-readable hardware label, e.g. '8x A100-SXM4-80GB'")

    # benchmark inputs
    parser.add_argument("--prompts", default="benchmark_prompts_50.json", help="Prompt corpus JSON")
    parser.add_argument("--spec", default="benchmark_spec.json", help="Benchmark spec JSON")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-json", default="benchmark_results.json")
    parser.add_argument("--categories", nargs="*")
    parser.add_argument("--ids", nargs="*")
    return parser.parse_args()


def load_prompts(path: Path) -> list[dict]:
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


def filter_prompts(
    prompts: list[dict],
    categories: set[str] | None,
    ids: set[str] | None,
) -> list[dict]:
    filtered = []
    for item in prompts:
        if categories and item["category"] not in categories:
            continue
        if ids and item["id"] not in ids:
            continue
        filtered.append(item)
    return filtered


def get_git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
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


def run_case(client: OpenAI, model: str, prompt_item: dict) -> dict:
    start = time.perf_counter()
    first_token_time: float | None = None
    end_time: float | None = None
    prompt_tokens: int | None = None
    generation_tokens: int | None = None
    error: str | None = None

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_item["prompt"]}],
            max_tokens=prompt_item["max_tokens"],
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
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
    # prompt_tps is estimated from TTFT — less reliable than MLX's native measurement
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


def summarize(results: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for result in results:
        key = (result["category"], result["max_tokens"])
        groups.setdefault(key, []).append(result)

    summary = []
    for (category, max_tokens), rows in sorted(groups.items()):
        gen_tps = [r["generation_tps"] for r in rows if r["generation_tps"] is not None]
        ttft = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
        emitted = [r["generation_tokens"] for r in rows if r["generation_tokens"] is not None]
        summary.append({
            "category": category,
            "max_tokens": max_tokens,
            "runs": len(rows),
            "avg_generation_tps": round(statistics.mean(gen_tps), 3) if gen_tps else None,
            "avg_ttft_s": round(statistics.mean(ttft), 3) if ttft else None,
            "avg_generation_tokens": round(statistics.mean(emitted), 1) if emitted else None,
        })
    return summary


def main() -> int:
    args = parse_args()
    prompts = load_prompts(Path(args.prompts))
    spec = load_spec(Path(args.spec))
    prompts = filter_prompts(
        prompts,
        set(args.categories) if args.categories else None,
        set(args.ids) if args.ids else None,
    )
    if not prompts:
        print("No prompts selected.", file=sys.stderr)
        return 1

    client = OpenAI(api_key="local", base_url=args.api_base)

    results = []
    total = len(prompts) * args.repeats
    run_index = 0

    for prompt_item in prompts:
        for repeat in range(1, args.repeats + 1):
            run_index += 1
            print(
                f"[{run_index}/{total}] {prompt_item['id']} category={prompt_item['category']} "
                f"max_tokens={prompt_item['max_tokens']} repeat={repeat}",
                flush=True,
            )
            result = run_case(client, args.model, prompt_item)
            result["repeat"] = repeat
            results.append(result)
            print(
                f"  prompt_tps={result['prompt_tps']} generation_tps={result['generation_tps']} "
                f"generation_tokens={result['generation_tokens']} ttft_s={result['ttft_s']} "
                f"elapsed_s={result['elapsed_s']}"
                + (f" ERROR={result['error']}" if result["error"] else ""),
                flush=True,
            )

    payload = {
        "benchmark": spec,
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
            "host": get_host_metadata(),
            "api_base": args.api_base,
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
            "label": args.hw_label,
            "hostname": platform.node(),
        },
        "prompt_file": str(Path(args.prompts).resolve()),
        "spec_file": str(Path(args.spec).resolve()),
        "repeats": args.repeats,
        "results": results,
        "summary": summarize(results),
    }

    output_path = Path(args.output_json)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nSummary")
    print(
        f"benchmark_id={spec.get('benchmark_id')} "
        f"benchmark_version={spec.get('benchmark_version')} "
        f"runner=openai_compat-0.1.0"
    )
    print(
        f"runtime={args.runtime} version={args.runtime_version} "
        f"model={args.model} dtype={args.model_dtype} tp={args.model_tp}"
    )
    print(
        f"hardware={args.hw_label or platform.node()} "
        f"timestamp_utc={payload['run']['timestamp_utc']} "
        f"git_commit={payload['run']['git_commit']}"
    )
    for row in payload["summary"]:
        print(
            f"{row['category']:>14} max_tokens={row['max_tokens']:<4} runs={row['runs']:<2} "
            f"avg_generation_tps={row['avg_generation_tps']} avg_ttft_s={row['avg_ttft_s']} "
            f"avg_generation_tokens={row['avg_generation_tokens']}"
        )
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
