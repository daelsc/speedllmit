# speedllmit

Benchmark project for comparing local LLM inference performance across different hardware,
models, quantizations, and runtimes.

## Design

Each benchmark result captures three dimensions:

- **Runtime** — the inference backend (vLLM, sglang, MLX, llama.cpp, Ollama, etc.) + version
- **Model** — served name, dtype, quantization, tensor parallel degree, max context
- **Hardware** — machine hostname + human-readable accelerator label

This lets results be compared across any axis: same model on different runtimes, same runtime
on different hardware, different quantizations of the same model.

## Layout

```
benchmark_spec.json             Versioned benchmark definition
prompts/benchmark_prompts_50.json   Mixed 50-prompt corpus
runners/
  mlx_benchmark_suite.py        MLX runner (Apple Silicon, subprocess-based)
  openai_compat_benchmark_suite.py  OpenAI-compatible API runner (vLLM, sglang, llama.cpp, etc.)
results/
  schema.md                     Canonical result JSON schema
  *.json / *.md                 Recorded results
gemma31_benchmark_repro.md      Reproduction notes for the Gemma 4 31B MLX baseline
```

## Runners

### `openai_compat_benchmark_suite.py`

Works with any OpenAI-compatible endpoint: vLLM, sglang, llama.cpp server, Ollama, etc.
Measures generation TPS and TTFT via streaming.

```bash
cd /path/to/speedllmit
python3 runners/openai_compat_benchmark_suite.py \
  --api-base http://localhost:8027/v1 \
  --runtime vllm \
  --runtime-version 0.19.0 \
  --model gemma-4-31b \
  --model-dtype bfloat16 \
  --model-tp 1 \
  --model-max-ctx 32768 \
  --hw-label "8x A100-SXM4-80GB" \
  --spec benchmark_spec.json \
  --prompts prompts/benchmark_prompts_50.json \
  --repeats 1 \
  --output-json results/a100_gemma4_31b_vllm_bf16_tp1_2026-04-09.json
```

For sglang, swap `--runtime sglang` and point `--api-base` at the sglang endpoint.
The runner is backend-agnostic as long as the endpoint is OpenAI-compatible.

### `mlx_benchmark_suite.py`

Apple Silicon only. Invokes `mlx_vlm.generate` as a subprocess and parses stdout.
Captures prompt TPS natively and peak memory usage.

```bash
cd /path/to/speedllmit
python3 runners/mlx_benchmark_suite.py \
  --model /path/to/model-folder \
  --spec benchmark_spec.json \
  --prompts prompts/benchmark_prompts_50.json \
  --repeats 1 \
  --output-json results/speedfreak_gemma4_31b_mlx_bf16_2026-04-09.json
```

## Versioning

| File | When to bump |
|------|-------------|
| `benchmark_spec.json` → `benchmark_version` | Benchmark behavior or reporting changes |
| `benchmark_spec.json` → `prompt_corpus_version` | Prompts, categories, or max_tokens change |
| `benchmark_spec.json` → `runner_version` | Runner parsing or summarization changes |

## Result Schema

See [`results/schema.md`](results/schema.md) for the full field reference and
notes on what can and cannot be compared across runners.

## Current Baselines

| File | Machine | Model | Runtime | dtype |
|------|---------|-------|---------|-------|
| `results/speedfreak_gemma4_31b_it_bf16_2026-04-09.json` | speedfreak (Apple Silicon) | Gemma 4 31B | MLX | bf16 |

Headline result on `speedfreak` for Gemma 4 31B bf16 in MLX:
- Generation: `10.34–10.45 tok/s` (short/medium), `5.676 tok/s` (256-token), `7.019 tok/s` (long natural)
- Peak memory: `62.8–63.1 GB`
