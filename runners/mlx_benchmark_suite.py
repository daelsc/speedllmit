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
from datetime import datetime, timezone
from pathlib import Path

PROMPT_RE = re.compile(r"^Prompt:\s+(\d+)\s+tokens,\s+([0-9.]+)\s+tokens-per-sec$")
GEN_RE = re.compile(r"^Generation:\s+(\d+)\s+tokens,\s+([0-9.]+)\s+tokens-per-sec$")
MEM_RE = re.compile(r"^Peak memory:\s+([0-9.]+)\s+GB$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an MLX benchmark corpus and summarize the results.")
    parser.add_argument("--model", required=True, help="Local model folder or remote model id.")
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
        "generation_tokens": generation_tokens,
        "generation_tps": generation_tps,
        "peak_memory_gb": peak_memory_gb,
        "raw_output": text,
    }


def summarize(results: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for result in results:
        key = (result["category"], result["max_tokens"])
        groups.setdefault(key, []).append(result)

    summary = []
    for (category, max_tokens), rows in sorted(groups.items()):
        prompt_tps = [row["prompt_tps"] for row in rows if row["prompt_tps"] is not None]
        gen_tps = [row["generation_tps"] for row in rows if row["generation_tps"] is not None]
        mem = [row["peak_memory_gb"] for row in rows if row["peak_memory_gb"] is not None]
        emitted = [row["generation_tokens"] for row in rows if row["generation_tokens"] is not None]
        summary.append(
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
            result = run_case(args.model, prompt_item)
            result["repeat"] = repeat
            results.append(result)
            print(
                f"  prompt_tps={result['prompt_tps']} generation_tps={result['generation_tps']} "
                f"generation_tokens={result['generation_tokens']} peak_memory_gb={result['peak_memory_gb']} "
                f"elapsed_s={result['elapsed_s']} returncode={result['returncode']}",
                flush=True,
            )

    payload = {
        "benchmark": spec,
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
            "host": get_host_metadata(),
        },
        "model": args.model,
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
        f"prompt_corpus_version={spec.get('prompt_corpus_version')} "
        f"runner_version={spec.get('runner_version')}"
    )
    print(
        f"timestamp_utc={payload['run']['timestamp_utc']} "
        f"git_commit={payload['run']['git_commit']} "
        f"hostname={payload['run']['host']['hostname']}"
    )
    for row in payload["summary"]:
        print(
            f"{row['category']:>14} max_tokens={row['max_tokens']:<4} runs={row['runs']:<2} "
            f"avg_prompt_tps={row['avg_prompt_tps']} avg_generation_tps={row['avg_generation_tps']} "
            f"avg_peak_memory_gb={row['avg_peak_memory_gb']} avg_generation_tokens={row['avg_generation_tokens']}"
        )
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
