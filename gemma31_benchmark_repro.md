# Gemma 4 31B MLX Benchmark Repro

This note describes how to reproduce the local MLX benchmark that was run on `speedfreak`.

## Target setup

- Hardware class: Apple Silicon Mac
- Runtime env: `mlx-vlm` in a Python virtualenv
- Model format: local MLX folder, not a remote HF model id at benchmark time
- Model used on `speedfreak`: `gemma-4-31b-it-bf16`
- Local model path used on `speedfreak`: `/Users/speedfreak/llm/models/gemma-4-31b-it-bf16`
- Virtualenv used on `speedfreak`: `/Users/speedfreak/llm/mlx-gemma31/.venv`
- Benchmark spec file: `benchmark_spec.json`
- Benchmark version: `0.1.0`
- Prompt corpus version: `0.1.0`
- Runner version: `0.1.0`

## Benchmark goal

Capture these metrics from MLX:

- prompt token count
- prompt tokens/sec
- generation token count
- generation tokens/sec
- peak memory

## Exact benchmark command

Run from the MLX env:

```bash
cd ~/llm/mlx-gemma31
source .venv/bin/activate
python -m mlx_vlm.generate \
  --model /path/to/gemma-4-31b-it-bf16 \
  --max-tokens 128 \
  --temperature 0.0 \
  --prompt "Write a short explanation of how a heat pump works."
```

Replace `/path/to/gemma-4-31b-it-bf16` with the local model folder on the target machine.

## More thorough benchmark loop

This is the broader benchmark used to get a more stable picture:

```bash
cd ~/llm/mlx-gemma31
source .venv/bin/activate

for t in 64 128 256; do
  for r in 1 2; do
    echo "__RUN__ max_tokens=$t repeat=$r"
    python -m mlx_vlm.generate \
      --model /path/to/gemma-4-31b-it-bf16 \
      --max-tokens "$t" \
      --temperature 0.0 \
      --prompt "Write a short explanation of how a heat pump works." \
      2>&1 | egrep '^(Prompt:|Generation:|Peak memory:)'
  done
done
```

## Baseline result from `speedfreak`

Single 128-token benchmark:

```text
Prompt: 69 tokens, 49.201 tokens-per-sec
Generation: 128 tokens, 10.359 tokens-per-sec
Peak memory: 62.853 GB
```

Thorough benchmark samples observed on `speedfreak`:

```text
64 tokens:
- Prompt: 24 tokens, 34.660 tokens-per-sec
- Generation: 64 tokens, 10.450 tokens-per-sec
- Peak memory: 62.813 GB

64 tokens:
- Prompt: 24 tokens, 35.655 tokens-per-sec
- Generation: 64 tokens, 10.444 tokens-per-sec
- Peak memory: 62.813 GB

128 tokens:
- Prompt: 24 tokens, 35.715 tokens-per-sec
- Generation: 128 tokens, 10.350 tokens-per-sec
- Peak memory: 62.813 GB

128 tokens:
- Prompt: 24 tokens, 36.124 tokens-per-sec
- Generation: 128 tokens, 10.344 tokens-per-sec
- Peak memory: 62.813 GB
```

The loop was interrupted before the 256-token runs completed, so rerun those on the target machine for a complete table.

Updated results later captured on `speedfreak`:

```text
256 tokens:
- Prompt: 24 tokens, 33.303 tokens-per-sec
- Generation: 256 tokens, 5.676 tokens-per-sec
- Peak memory: 62.826 GB

Longer run requested at 1000 tokens:
- Prompt: 24 tokens, 14.029 tokens-per-sec
- Generation: 447 tokens, 7.019 tokens-per-sec
- Peak memory: 63.057 GB
```

The nominal 1000-token run stopped naturally at 447 generated tokens, so it should be treated as a longer natural-completion sample rather than a strict 1000-token completion benchmark.

## Larger 50-prompt suite

This repo also contains:

- `prompts/benchmark_prompts_50.json`
- `runners/mlx_benchmark_suite.py`

Use them like this:

```bash
cd /path/to/this/repo
python3 mlx_benchmark_suite.py \
  --model /path/to/gemma-4-31b-it-bf16 \
  --spec benchmark_spec.json \
  --prompts prompts/benchmark_prompts_50.json \
  --repeats 1 \
  --output-json benchmark_results.json
```

The prompt set includes about 50 prompts across:

- short explanations
- structured outputs
- code generation
- longer explanations
- forced-long workloads

This is a better fit than MLPerf for practical local-machine comparison because it exercises several workload shapes while still using a simple reproducible harness.

Results should always record:

- benchmark id
- benchmark version
- prompt corpus version
- runner version
- git commit hash
- run timestamp in UTC
- host metadata
- model id or local path
- hardware metadata

## Notes

- `python -m mlx_vlm.generate` prints a deprecation warning recommending `mlx_vlm generate`. That warning did not block the benchmark.
- The generated text output looked slightly odd in this setup because of prompt formatting, but the throughput and memory figures were still usable.
- For apples-to-apples comparison, keep the same:
  - model
  - token counts
  - temperature
  - prompt text
