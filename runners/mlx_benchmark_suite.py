#!/usr/bin/env python3

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

PROMPT_RE = re.compile(r"^Prompt:\s+(\d+)\s+tokens,\s+([0-9.]+)\s+tokens-per-sec$")
GEN_RE = re.compile(r"^Generation:\s+(\d+)\s+tokens,\s+([0-9.]+)\s+tokens-per-sec$")
MEM_RE = re.compile(r"^Peak memory:\s+([0-9.]+)\s+GB$")
RUNNER_VERSION = "0.3.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an MLX benchmark corpus and summarize the results.")
    parser.add_argument("--model", required=True, help="Local model folder or remote model id.")
    parser.add_argument("--model-dtype", default="bfloat16", help="Weight dtype, e.g. bfloat16, float16.")
    parser.add_argument("--model-quant", default=None, help="Quantization scheme, e.g. mxfp4, 4bit.")
    parser.add_argument("--model-tp", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--model-max-ctx", type=int, default=None, help="Configured max context length.")
    parser.add_argument("--runtime", default="mlx", help="Runtime name.")
    parser.add_argument("--runtime-version", default=None, help="Runtime version string.")
    parser.add_argument("--hw-machine", default=None, help="Machine hardware inventory label.")
    parser.add_argument(
        "--hw-gpus-used",
        type=int,
        default=1,
        help="Number of GPUs/accelerators actually used for this run.",
    )
    parser.add_argument(
        "--prompts",
        default="benchmark_prompts_50.json",
        help="Path to benchmark prompt JSON file.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="How many times to run each prompt.")
    parser.add_argument(
        "--output-json",
        default="benchmark_results.json",
        help="Where to write the full result set as JSON.",
    )
    parser.add_argument(
        "--spec",
        default="benchmark_spec.json",
        help="Path to the benchmark spec JSON file.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Optional list of categories to include.",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        help="Optional list of prompt ids to include.",
    )
    conc_group = parser.add_mutually_exclusive_group()
    conc_group.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of simultaneous subprocesses (default 1 = serial).",
    )
    conc_group.add_argument(
        "--concurrency-sweep",
        default=None,
        help="Comma-separated concurrency levels to run in sequence, e.g. 1,2,4,8",
    )
    conc_group.add_argument(
        "--concurrency-auto",
        action="store_true",
        default=False,
        help="Auto-sweep concurrency doubling from --concurrency-start until aggregate TPS saturates",
    )
    parser.add_argument(
        "--concurrency-start",
        type=int,
        default=1,
        help="Starting concurrency for --concurrency-auto (default 1)",
    )
    parser.add_argument(
        "--saturation-threshold",
        type=float,
        default=0.05,
        help="--concurrency-auto stops when TPS improvement drops below this fraction (default 0.05 = 5%%)",
    )
    return parser.parse_args()


def load_prompts(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        prompts = json.load(f)
    for item in prompts:
        for key in ("id", "category", "max_tokens", "prompt"):
            if key not in item:
                raise ValueError(f"Missing key {key} in prompt item: {item}")
    return prompts


def load_spec(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None


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


def filter_prompts(prompts: list[dict], categories: set[str] | None, ids: set[str] | None) -> list[dict]:
    filtered = []
    for item in prompts:
        if categories and item["category"] not in categories:
            continue
        if ids and item["id"] not in ids:
            continue
        filtered.append(item)
    return filtered


def run_case(model: str, prompt_item: dict) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "mlx_vlm.generate",
        "--model",
        model,
        "--max-tokens",
        str(prompt_item["max_tokens"]),
        "--temperature",
        "0.0",
        "--prompt",
        prompt_item["prompt"],
    ]
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.time() - start
    text = proc.stdout + proc.stderr

    prompt_tokens = None
    prompt_tps = None
    generation_tokens = None
    generation_tps = None
    peak_memory_gb = None

    for line in text.splitlines():
        line = line.strip()
        m = PROMPT_RE.match(line)
        if m:
            prompt_tokens = int(m.group(1))
            prompt_tps = float(m.group(2))
            continue
        m = GEN_RE.match(line)
        if m:
            generation_tokens = int(m.group(1))
            generation_tps = float(m.group(2))
            continue
        m = MEM_RE.match(line)
        if m:
            peak_memory_gb = float(m.group(1))

    return {
        "id": prompt_item["id"],
        "category": prompt_item["category"],
        "max_tokens": prompt_item["max_tokens"],
        "elapsed_s": round(elapsed, 3),
        "returncode": proc.returncode,
        "prompt_tokens": prompt_tokens,
        "prompt_tps": prompt_tps,
        "prompt_tps_source": "native" if prompt_tps is not None else None,
        "generation_tokens": generation_tokens,
        "generation_tps": generation_tps,
        "ttft_s": None,
        "peak_memory_gb": peak_memory_gb,
        "error": None if proc.returncode == 0 else text,
        "raw_output": text,
    }


def build_summary(results: list[dict], wall_time_s: float, aggregate_tps: float | None) -> dict:
    groups: dict[tuple[str, int], list[dict]] = {}
    for result in results:
        key = (result["category"], result["max_tokens"])
        groups.setdefault(key, []).append(result)

    per_category = []
    for (category, max_tokens), rows in sorted(groups.items()):
        prompt_tps = [row["prompt_tps"] for row in rows if row["prompt_tps"] is not None]
        gen_tps = [row["generation_tps"] for row in rows if row["generation_tps"] is not None]
        mem = [row["peak_memory_gb"] for row in rows if row["peak_memory_gb"] is not None]
        emitted = [row["generation_tokens"] for row in rows if row["generation_tokens"] is not None]
        per_category.append(
            {
                "category": category,
                "max_tokens": max_tokens,
                "runs": len(rows),
                "avg_prompt_tps": round(statistics.mean(prompt_tps), 3) if prompt_tps else None,
                "avg_generation_tps": round(statistics.mean(gen_tps), 3) if gen_tps else None,
                "avg_peak_memory_gb": round(statistics.mean(mem), 3) if mem else None,
                "avg_generation_tokens": round(statistics.mean(emitted), 1) if emitted else None,
            }
        )
    return {
        "concurrency": 1,
        "wall_time_s": wall_time_s,
        "aggregate_generation_tps": aggregate_tps,
        "per_category": per_category,
    }


def run_all(model: str, prompts: list[dict], repeats: int, concurrency: int) -> tuple[list[dict], float, float | None]:
    tasks = [(prompt_item, repeat) for prompt_item in prompts for repeat in range(1, repeats + 1)]
    total = len(tasks)
    results: list[dict | None] = [None] * total
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_meta = {
            executor.submit(run_case, model, prompt_item): (idx, prompt_item, repeat)
            for idx, (prompt_item, repeat) in enumerate(tasks)
        }
        completed = 0
        for future in as_completed(future_to_meta):
            idx, prompt_item, repeat = future_to_meta[future]
            result = future.result()
            result["repeat"] = repeat
            results[idx] = result
            completed += 1
            print(
                f"[{completed}/{total}] {prompt_item['id']} category={prompt_item['category']} "
                f"max_tokens={prompt_item['max_tokens']} repeat={repeat} "
                f"prompt_tps={result['prompt_tps']} generation_tps={result['generation_tps']} "
                f"generation_tokens={result['generation_tokens']} peak_memory_gb={result['peak_memory_gb']} "
                f"elapsed_s={result['elapsed_s']} returncode={result['returncode']}",
                flush=True,
            )

    wall_time_s = round(time.perf_counter() - wall_start, 3)
    final_results = [row for row in results if row is not None]
    total_generation_tokens = sum(
        row["generation_tokens"] for row in final_results if row["generation_tokens"] is not None
    )
    aggregate_generation_tps = round(total_generation_tokens / wall_time_s, 3) if wall_time_s > 0 else None
    return final_results, wall_time_s, aggregate_generation_tps


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

    all_results = []
    all_summaries = []

    if args.concurrency_sweep:
        levels = [int(x.strip()) for x in args.concurrency_sweep.split(",")]
    elif args.concurrency_auto:
        levels = None
    else:
        levels = [args.concurrency]

    if levels is not None:
        for level in levels:
            print(f"\n--- concurrency={level} ({len(prompts)} prompts × {args.repeats} repeat(s)) ---", flush=True)
            results, wall_time_s, aggregate_generation_tps = run_all(args.model, prompts, args.repeats, level)
            summary = build_summary(results, wall_time_s, aggregate_generation_tps)
            summary["concurrency"] = level
            all_results.extend(results)
            all_summaries.append(summary)
    else:
        level = args.concurrency_start
        prev_tps = None
        confirmed_flat = False
        max_concurrency = 2048

        while level <= max_concurrency:
            print(
                f"\n--- concurrency={level} ({len(prompts)} prompts × {args.repeats} repeat(s)) [auto] ---",
                flush=True,
            )
            results, wall_time_s, aggregate_generation_tps = run_all(args.model, prompts, args.repeats, level)
            summary = build_summary(results, wall_time_s, aggregate_generation_tps)
            summary["concurrency"] = level
            all_results.extend(results)
            all_summaries.append(summary)

            if prev_tps is not None and aggregate_generation_tps is not None:
                improvement = (aggregate_generation_tps - prev_tps) / prev_tps
                print(
                    f"  [auto] improvement={improvement:.1%} vs previous "
                    f"(threshold={args.saturation_threshold:.0%})",
                    flush=True,
                )
                if improvement < args.saturation_threshold:
                    if confirmed_flat:
                        print(f"  [auto] saturated — stopping after c={level}", flush=True)
                        break
                    print(f"  [auto] below threshold — running one confirmation step", flush=True)
                    confirmed_flat = True
                else:
                    confirmed_flat = False

            prev_tps = aggregate_generation_tps
            level *= 2

    payload = {
        "benchmark": spec,
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
            "host": get_host_metadata(),
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
            "gpus_used": args.hw_gpus_used,
            "hostname": platform.node(),
        },
        "prompt_file": str(Path(args.prompts).resolve()),
        "spec_file": str(Path(args.spec).resolve()),
        "prompts_per_category": "all",
        "repeats": args.repeats,
        "concurrency_mode": "auto" if args.concurrency_auto else ("sweep" if args.concurrency_sweep else "fixed"),
        "concurrency_auto_config": {
            "start": args.concurrency_start,
            "saturation_threshold": args.saturation_threshold,
        } if args.concurrency_auto else None,
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
        f"machine={args.hw_machine}  gpus_used={args.hw_gpus_used}  hostname={payload['run']['host']['hostname']}"
    )
    print(
        f"prompts={len(prompts)} (all/category)  repeats={args.repeats}  "
        f"mode={'auto' if args.concurrency_auto else ('sweep' if args.concurrency_sweep else f'c={args.concurrency}')}"
    )
    for summary in all_summaries:
        print(
            f"\nconcurrency={summary['concurrency']}  "
            f"wall={summary['wall_time_s']}s  "
            f"aggregate_tps={summary['aggregate_generation_tps']}"
        )
        for row in summary["per_category"]:
            print(
                f"{row['category']:>14} max_tokens={row['max_tokens']:<4} runs={row['runs']:<2} "
                f"avg_prompt_tps={row['avg_prompt_tps']} avg_generation_tps={row['avg_generation_tps']} "
                f"avg_peak_memory_gb={row['avg_peak_memory_gb']} avg_generation_tokens={row['avg_generation_tokens']}"
            )
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
