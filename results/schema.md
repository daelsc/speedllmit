# Results Schema

Every result JSON file should contain the following top-level fields.
Runners populate all fields they can; use `null` for fields not available from their backend.

## Top-level structure

```json
{
  "benchmark":            { ... },
  "run":                  { ... },
  "runtime":              { ... },
  "model":                { ... },
  "hardware":             { ... },
  "prompt_file":          "...",
  "spec_file":            "...",
  "prompts_per_category": 3,
  "repeats":              1,
  "results":              [ ... ],
  "summaries":            [ ... ]
}
```

## `benchmark`

Copied verbatim from `benchmark_spec.json`.

| Field | Type | Description |
|-------|------|-------------|
| `benchmark_id` | string | Stable identifier for this benchmark definition |
| `benchmark_version` | string | Semver — bump when benchmark behavior changes |
| `prompt_corpus_version` | string | Semver — bump when prompts change |
| `runner_file` | string | Canonical runner filename |
| `runner_version` | string | Semver — bump when runner logic changes |

## `run`

Execution context.

| Field | Type | Description |
|-------|------|-------------|
| `timestamp_utc` | ISO 8601 string | Wall-clock start of run |
| `git_commit` | string \| null | Git HEAD of the speedllmit repo at run time |
| `runner_version` | string | Version of the runner script used |
| `host` | object | Full `platform` metadata (hostname, OS, arch, Python version, etc.) |
| `api_base` | string \| null | Endpoint URL (openai_compat runner only) |

## `runtime`

The inference backend being tested.

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | e.g. `vllm`, `sglang`, `llama.cpp`, `mlx`, `ollama` |
| `version` | string \| null | Runtime version string, e.g. `0.19.0` |

## `model`

What model is loaded and how.

| Field | Type | Description |
|-------|------|-------------|
| `served_name` | string | Name as returned by the API, e.g. `gemma-4-31b` |
| `dtype` | string \| null | Weight dtype, e.g. `bfloat16`, `float16`, `int4` |
| `quant` | string \| null | Quantization scheme, e.g. `AWQ`, `GPTQ`, `Q4_K_M` |
| `tensor_parallel` | int \| null | Tensor parallel degree |
| `max_context` | int \| null | Configured max context length (tokens) |

## `hardware`

The machine and how many accelerators this run actually used.

| Field | Type | Description |
|-------|------|-------------|
| `machine` | string \| null | Full machine accelerator inventory, e.g. `8x A100-SXM4-80GB`. Documents what the machine *has*, not what this run *uses*. |
| `gpus_used` | int \| null | Number of GPUs actually used for this run. Derived from `tensor_parallel` unless explicitly overridden. |
| `hostname` | string | Machine hostname |

**Important:** `machine` and `gpus_used` serve different purposes. A machine with 8 GPUs running a
TP=1 model uses 1 GPU. Always record both so results are unambiguous.

## Per-run result fields (`results[]`)

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Prompt id |
| `category` | string | Prompt category |
| `max_tokens` | int | Max tokens requested |
| `repeat` | int | Repeat index (1-based) |
| `elapsed_s` | float | Total wall time for this call |
| `prompt_tokens` | int \| null | Tokens in the prompt |
| `prompt_tps` | float \| null | Prompt processing speed (tokens/sec) |
| `prompt_tps_source` | string \| null | `native` (runtime reported) or `estimated_from_ttft` |
| `generation_tokens` | int \| null | Tokens generated |
| `generation_tps` | float \| null | Generation throughput (tokens/sec) — single-request view |
| `ttft_s` | float \| null | Time to first token (seconds); openai_compat runner only |
| `peak_memory_gb` | float \| null | Peak memory (GB); MLX runner only |
| `error` | string \| null | Error message if the call failed |

## Summary fields (`summaries[]`)

One entry per concurrency level. For a serial run (`--concurrency 1`) there is one entry.
For a sweep (`--concurrency-sweep 1,4,8,16`) there is one entry per level.

### Top-level summary fields

| Field | Type | Description |
|-------|------|-------------|
| `concurrency` | int | Number of simultaneous requests for this entry |
| `wall_time_s` | float | Total wall time from first dispatch to last completion |
| `aggregate_generation_tps` | float \| null | Total tokens generated / wall time — true server throughput |

### `per_category[]` fields (always present)

| Field | Type | Description |
|-------|------|-------------|
| `category` | string | |
| `max_tokens` | int | |
| `runs` | int | Number of calls in this group |
| `avg_generation_tps` | float \| null | Mean per-request generation throughput |
| `avg_ttft_s` | float \| null | Mean time to first token |
| `avg_generation_tokens` | float \| null | Mean tokens actually generated |

### `per_category[]` fields (concurrency > 1 only)

| Field | Type | Description |
|-------|------|-------------|
| `p50_generation_tps` | float \| null | Median per-request generation throughput |
| `p95_generation_tps` | float \| null | p95 per-request generation throughput |
| `p50_ttft_s` | float \| null | Median TTFT |
| `p95_ttft_s` | float \| null | p95 TTFT |
| `p99_ttft_s` | float \| null | p99 TTFT |

## Notes on cross-runner comparison

- **`prompt_tps`**: MLX reports this natively; the openai_compat runner estimates it
  from TTFT and marks it `estimated_from_ttft`. Do not compare these directly.
- **`peak_memory_gb`**: Only available from the MLX runner. Not queryable via API.
- **`ttft_s`**: Only the openai_compat runner measures this.
- **`aggregate_generation_tps`**: The most useful throughput metric for run-level comparison.
  For serial MLX runs it is total emitted tokens divided by full benchmark wall time.
  For API runners at higher concurrency it also reflects batching efficiency.
- For apples-to-apples comparison across runtimes, use `aggregate_generation_tps` and
  `avg_generation_tps`. Everything else is backend-specific.

## Result filename convention

```
results/<hostname>_<model>_<runtime>_<dtype>_tp<N>_c<concurrency>_<date>.json
```

Examples:
- `results/a100_gemma4_31b_vllm_bf16_tp1_c1_2026-04-09.json`   ← serial, 1 GPU
- `results/a100_gemma4_31b_vllm_bf16_tp1_c16_2026-04-09.json`  ← 16 concurrent, 1 GPU
- `results/a100_gemma4_31b_sglang_bf16_tp2_c8_2026-04-09.json` ← 8 concurrent, 2 GPUs
- `results/speedfreak_gemma4_31b_mlx_bf16_tp1_c1_2026-04-09.json`
