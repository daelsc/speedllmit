# Speedfreak Gemma 4 31B bf16 Results

Recorded on April 9, 2026.

## Hardware

- Host label: `speedfreak`
- Hostname: `SpeedfracStudio.lan`
- Machine class: Mac Studio
- SoC: Apple `M3 Ultra`
- Memory: `512 GB` unified memory
- OS: macOS `26.3`
- Kernel: `Darwin 25.3.0`

## Runtime

- Python env: `/Users/speedfreak/llm/mlx-gemma31/.venv`
- Model path: `/Users/speedfreak/llm/models/gemma-4-31b-it-bf16`
- Model source: `mlx-community/gemma-4-31b-it-bf16`
- Runtime family: `mlx-vlm`
- Benchmark id: `mlx-local-llm-benchmark`
- Benchmark version: `0.1.0`
- Prompt corpus file: `benchmark_prompts_50.json`
- Prompt corpus version: `0.1.0`
- Runner file: `mlx_benchmark_suite.py`
- Runner version: `0.1.0`

Installed runtime versions used for setup on this machine:

- `mlx 0.31.1`
- `mlx-lm 0.31.2`
- `mlx-vlm 0.4.4`

## Workload

Primary benchmark prompt:

```text
Write a short explanation of how a heat pump works.
```

Temperature:

```text
0.0
```

## Direct Single Run

Command shape:

```bash
python -m mlx_vlm.generate \
  --model /Users/speedfreak/llm/models/gemma-4-31b-it-bf16 \
  --max-tokens 128 \
  --temperature 0.0 \
  --prompt "Write a short explanation of how a heat pump works."
```

Observed result:

```text
Prompt: 69 tokens, 49.201 tokens-per-sec
Generation: 128 tokens, 10.359 tokens-per-sec
Peak memory: 62.853 GB
```

## Repeated Short And Medium Runs

64-token run 1:

```text
Prompt: 24 tokens, 34.660 tokens-per-sec
Generation: 64 tokens, 10.450 tokens-per-sec
Peak memory: 62.813 GB
```

64-token run 2:

```text
Prompt: 24 tokens, 35.655 tokens-per-sec
Generation: 64 tokens, 10.444 tokens-per-sec
Peak memory: 62.813 GB
```

128-token run 1:

```text
Prompt: 24 tokens, 35.715 tokens-per-sec
Generation: 128 tokens, 10.350 tokens-per-sec
Peak memory: 62.813 GB
```

128-token run 2:

```text
Prompt: 24 tokens, 36.124 tokens-per-sec
Generation: 128 tokens, 10.344 tokens-per-sec
Peak memory: 62.813 GB
```

## Longer Runs

256-token run:

```text
Prompt: 24 tokens, 33.303 tokens-per-sec
Generation: 256 tokens, 5.676 tokens-per-sec
Peak memory: 62.826 GB
```

Longer run requested at 1000 tokens, but the model stopped naturally at 447 tokens:

```text
Prompt: 24 tokens, 14.029 tokens-per-sec
Generation: 447 tokens, 7.019 tokens-per-sec
Peak memory: 63.057 GB
```

## Interpretation

- Short and medium completions were very consistent at about `10.34–10.45 tok/s`.
- Prompt throughput was much higher than generation throughput.
- Peak memory stayed near `63 GB`.
- Longer generations did not stay at the short-run rate in this setup.
- The `1000-token` request did not force a full 1000-token output, so it should be treated as a longer natural-completion sample, not a strict 1000-token benchmark.

## Reproduction

See:

- [`gemma31_benchmark_repro.md`](/home/dave/git/agent-tasks/aerae-agent/gemma31_benchmark_repro.md)
- [`benchmark_prompts_50.json`](/home/dave/git/agent-tasks/aerae-agent/benchmark_prompts_50.json)
- [`mlx_benchmark_suite.py`](/home/dave/git/agent-tasks/aerae-agent/mlx_benchmark_suite.py)
