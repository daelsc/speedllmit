#!/bin/bash
# Benchmark: gemma-4-31b-200k, vLLM, bfloat16, TP=4 (4x A100), auto saturation sweep
# Run from the speedllmit repo root on a100:
#   bash configs/a100_gemma4_31b_200k_tp4_auto.sh
set -euo pipefail
cd "$(dirname "$0")/.."

~/bench-venv/bin/python3 runners/openai_compat_benchmark_suite.py \
  --api-base http://localhost:8028/v1 \
  --runtime vllm \
  --runtime-version 0.19.0 \
  --model gemma-4-31b-200k \
  --model-dtype bfloat16 \
  --model-tp 4 \
  --model-max-ctx 200000 \
  --hw-machine "8x A100-SXM4-80GB" \
  --spec benchmark_spec.json \
  --prompts prompts/benchmark_prompts_50.json \
  --concurrency-auto \
  --concurrency-start 1 \
  --output-json "results/a100_gemma4_31b_200k_vllm_bf16_tp4_auto_$(date +%Y-%m-%d).json"
