# speedllmit

Benchmark project for comparing local LLM inference performance across different hardware, models, quantizations, and runtimes.

Current focus:
- Apple Silicon local inference
- MLX runtime
- reproducible benchmark definitions
- machine-readable results with benchmark versioning and host metadata

## Layout

- `benchmark_spec.json`
  Versioned benchmark definition.
- `prompts/benchmark_prompts_50.json`
  Mixed 50-prompt corpus.
- `runners/mlx_benchmark_suite.py`
  MLX benchmark runner.
- `gemma31_benchmark_repro.md`
  Reproduction notes for the Gemma 4 31B benchmark.
- `results/`
  Recorded results in both Markdown and JSON.

## Benchmark Metadata

Every results file should identify:
- benchmark id
- benchmark version
- prompt corpus version
- runner version
- git commit hash
- run timestamp in UTC
- host metadata
- hardware metadata
- model metadata

## Quick Start

Run the MLX suite like this:

```bash
cd /path/to/speedllmit
python3 runners/mlx_benchmark_suite.py \
  --model /path/to/model-folder \
  --spec benchmark_spec.json \
  --prompts prompts/benchmark_prompts_50.json \
  --repeats 1 \
  --output-json benchmark_results.json
```

## Current Baseline

Recorded baseline:
- `results/speedfreak_gemma4_31b_it_bf16_2026-04-09.md`
- `results/speedfreak_gemma4_31b_it_bf16_2026-04-09.json`

Headline result on `speedfreak` for Gemma 4 31B bf16 in MLX:
- short and medium runs around `10.34-10.45 tok/s`
- one 256-token run at `5.676 tok/s`
- one longer natural-completion run at `7.019 tok/s`
- peak memory around `62.8-63.1 GB`

## Scope

This project is meant for practical benchmarking, not formal MLPerf submission. The aim is reproducible, comparable local measurements that are easy to rerun on different machines.
