# Results Schema

Every result JSON file should contain the following top-level fields.
Runners are expected to populate all fields they can; use `null` for fields
that are not available from their backend.

## Top-level structure

```json
{
  "benchmark":   { ... },
  "run":         { ... },
  "runtime":     { ... },
  "model":       { ... },
  "hardware":    { ... },
  "prompt_file": "...",
  "spec_file":   "...",
  "repeats":     1,
  "results":     [ ... ],
  "summary":     [ ... ]
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

The machine and accelerator.

| Field | Type | Description |
|-------|------|-------------|
| `label` | string \| null | Human-readable description, e.g. `8x A100-SXM4-80GB` or `M4 Max 128GB` |
| `hostname` | string | Machine hostname |

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
| `prompt_tps_source` | string \| null | How prompt_tps was measured: `native` (runtime reported) or `estimated_from_ttft` |
| `generation_tokens` | int \| null | Tokens generated |
| `generation_tps` | float \| null | Generation throughput (tokens/sec) |
| `ttft_s` | float \| null | Time to first token (seconds); openai_compat runner only |
| `peak_memory_gb` | float \| null | Peak memory (GB); MLX runner only |
| `error` | string \| null | Error message if the call failed |

## Summary fields (`summary[]`)

Grouped by `(category, max_tokens)`.

| Field | Type | Description |
|-------|------|-------------|
| `category` | string | |
| `max_tokens` | int | |
| `runs` | int | Number of calls in this group |
| `avg_generation_tps` | float \| null | Mean generation throughput |
| `avg_ttft_s` | float \| null | Mean time to first token |
| `avg_generation_tokens` | float \| null | Mean tokens actually generated |

## Notes on cross-runner comparison

- **`prompt_tps`**: MLX reports this natively; the openai_compat runner estimates it
  from TTFT and marks it `estimated_from_ttft`. Do not compare these directly.
- **`peak_memory_gb`**: Only available from the MLX runner. Not queryable via API.
- **`ttft_s`**: Only the openai_compat runner measures this. MLX subprocess calls
  do not isolate TTFT.
- For apples-to-apples comparison across runtimes, use `avg_generation_tps` and
  `avg_generation_tokens`. Everything else is backend-specific.

## Result filename convention

```
results/<hostname>_<model>_<runtime>_<dtype>_<date>.json
```

Examples:
- `results/speedfreak_gemma4_31b_mlx_bf16_2026-04-09.json`
- `results/a100_gemma4_31b_vllm_bf16_tp1_2026-04-09.json`
- `results/a100_gemma4_31b_sglang_bf16_tp2_2026-04-09.json`
