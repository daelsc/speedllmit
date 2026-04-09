# speedllmit

Benchmark project for comparing local LLM inference performance across different hardware,
models, quantizations, and runtimes.

## Design

Each benchmark result captures three dimensions:

- **Runtime** — the inference backend (vLLM, sglang, MLX, llama.cpp, Ollama, etc.) + version
- **Model** — served name, dtype, quantization, tensor parallel degree, max context
- **Hardware** — machine inventory (`--hw-machine`, what the box has) and GPUs actually
  used for this run (`gpus_used`, derived from `--model-tp`)

`hardware.machine` and `hardware.gpus_used` are kept separate because a machine with 8 GPUs
running a TP=1 model only uses 1 GPU — conflating them produces misleading results.

## Layout

```
benchmark_spec.json                    Versioned benchmark definition
prompts/benchmark_prompts_50.json      Mixed 50-prompt corpus
runners/
  mlx_benchmark_suite.py               MLX runner (Apple Silicon, subprocess-based)
  openai_compat_benchmark_suite.py     OpenAI-compatible API runner (vLLM, sglang, llama.cpp, etc.)
results/
  schema.md                            Canonical result JSON schema
  *.json / *.md                        Recorded results
gemma31_benchmark_repro.md             Reproduction notes for the Gemma 4 31B MLX baseline
```

## Runners

### `openai_compat_benchmark_suite.py`

Works with any OpenAI-compatible endpoint: vLLM, sglang, llama.cpp server, Ollama, etc.
Uses async streaming to measure TTFT and generation TPS. Supports concurrent load testing.

**Serial run (default — 3 prompts/category, 15 total):**
```bash
python3 runners/openai_compat_benchmark_suite.py \
  --api-base http://localhost:8027/v1 \
  --runtime vllm --runtime-version 0.19.0 \
  --model gemma-4-31b --model-dtype bfloat16 --model-tp 1 --model-max-ctx 32768 \
  --hw-machine "8x A100-SXM4-80GB" \
  --spec benchmark_spec.json \
  --prompts prompts/benchmark_prompts_50.json \
  --output-json results/a100_gemma4_31b_vllm_bf16_tp1_c1_2026-04-09.json
```

**Concurrent run (N simultaneous requests):**
```bash
  ... --concurrency 16 \
  --output-json results/a100_gemma4_31b_vllm_bf16_tp1_c16_2026-04-09.json
```

**Concurrency sweep (serial through high load in one invocation):**
```bash
  ... --concurrency-sweep 1,4,8,16,32 \
  --output-json results/a100_gemma4_31b_vllm_bf16_tp1_sweep_2026-04-09.json
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--max-per-category` | 3 | Prompts per category (15 total with 5 categories). Use 0 for all 50. |
| `--concurrency` | 1 | Simultaneous requests. Mutex with `--concurrency-sweep`. |
| `--concurrency-sweep` | — | Comma-separated levels, e.g. `1,4,8,16`. |
| `--hw-machine` | — | Machine hardware inventory, e.g. `8x A100-SXM4-80GB`. |
| `--hw-gpus-used` | `--model-tp` | Override GPUs used if different from TP size. |

### `mlx_benchmark_suite.py`

Apple Silicon only. Invokes `mlx_vlm.generate` as a subprocess and parses stdout.
Captures prompt TPS natively, peak memory, and run-level aggregate generation throughput.
Supports the same benchmark modes as the API runner:
- fixed concurrency: `--concurrency N`
- explicit sweep: `--concurrency-sweep 1,2,4,8`
- auto knee finding: `--concurrency-auto --concurrency-start 1 --saturation-threshold 0.05`

For MLX, concurrency means simultaneous local subprocesses, not requests into a shared server.

```bash
python3 runners/mlx_benchmark_suite.py \
  --model /path/to/model-folder \
  --spec benchmark_spec.json \
  --prompts prompts/benchmark_prompts_50.json \
  --repeats 1 \
  --output-json results/speedfreak_gemma4_31b_mlx_bf16_tp1_c1_2026-04-09.json
```

```bash
python3 runners/mlx_benchmark_suite.py \
  --model /path/to/model-folder \
  --spec benchmark_spec.json \
  --prompts prompts/benchmark_prompts_50.json \
  --concurrency-auto \
  --concurrency-start 1 \
  --saturation-threshold 0.05 \
  --output-json results/speedfreak_gemma4_31b_mlx_bf16_auto_2026-04-09.json
```

## Metrics

| Metric | Description | Runners |
|--------|-------------|---------|
| `aggregate_generation_tps` | Total tokens / wall time across the whole run | both |
| `avg_generation_tps` | Mean per-request generation speed | both |
| `avg_ttft_s` | Mean time to first token | openai_compat |
| `p95_ttft_s` | p95 TTFT under concurrent load | openai_compat (concurrency > 1) |
| `peak_memory_gb` | Peak memory usage | mlx only |

At `concurrency=1`, `aggregate_generation_tps` will usually be slightly below `avg_generation_tps`
because it includes full end-to-end wall time across the suite. The gap at higher concurrency
reflects batching efficiency for server runtimes.

## Versioning

| File | When to bump |
|------|-------------|
| `benchmark_spec.json` → `benchmark_version` | Benchmark behavior or reporting changes |
| `benchmark_spec.json` → `prompt_corpus_version` | Prompts, categories, or max_tokens change |
| `benchmark_spec.json` → `runner_version` | Runner parsing or summarization changes |

## Result Schema

See [`results/schema.md`](results/schema.md) for the full field reference and
notes on what can and cannot be compared across runners.

## Baselines

| File | Machine | Model | Runtime | dtype | TP | GPUs used |
|------|---------|-------|---------|-------|----|-----------|
| `results/speedfreak_gemma4_31b_it_bf16_2026-04-09.json` | speedfreak (Apple Silicon) | Gemma 4 31B | MLX | bf16 | 1 | 1 |
| `results/a100_gemma4_31b_vllm_bf16_tp1_2026-04-09.json` | a100 (8x A100-SXM4-80GB) | Gemma 4 31B | vLLM 0.19.0 | bf16 | 1 | 1 |

### Headline numbers — Gemma 4 31B, serial (concurrency=1)

| Machine | Runtime | GPUs used | avg gen TPS |
|---------|---------|-----------|-------------|
| speedfreak (Apple Silicon) | MLX bf16 | 1 | ~10.4 tok/s |
| a100 | vLLM bf16 TP1 | 1 | ~24.7–25.1 tok/s |
