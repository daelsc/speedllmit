#!/bin/bash
# Benchmark: gemma-4-31b, vLLM, bfloat16, TP=1 (1x A100), auto saturation sweep
# Run from the speedllmit repo root on a100:
#   bash configs/a100_gemma4_31b_tp1_auto.sh
set -euo pipefail
cd "$(dirname "$0")/.."

~/bench-venv/bin/python3 runners/openai_compat_benchmark_suite.py \
  --api-base http://localhost:8027/v1 \
  --runtime vllm \
  --runtime-version 0.19.0 \
  --model gemma-4-31b \
  --model-dtype bfloat16 \
  --model-tp 1 \
  --model-max-ctx 32768 \
  --hw-machine "8x A100-SXM4-80GB" \
  --spec benchmark_spec.json \
  --prompts prompts/benchmark_prompts_50.json \
  --concurrency-auto \
  --concurrency-start 1 \
  --output-json "results/a100_gemma4_31b_vllm_bf16_tp1_auto_$(date +%Y-%m-%d).json"
