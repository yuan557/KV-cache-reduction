# KV Cache Compression Benchmark

A reproducible benchmark for comparing KV cache compression strategies during LLM inference. Evaluates memory savings, perplexity impact, and decode throughput on Qwen3 models with statistical rigor.

## Motivation

KV cache compression research is fragmented — papers test one strategy on one model with no shared evaluation protocol. This project provides a standardized framework to swap strategies, sweep memory budgets, and get statistically grounded comparisons with one command.

## Compression Strategies

| Strategy | Description | Status |
|---|---|---|
| FP16 Baseline | No compression (control group) | Done |
| INT8 Quantization | Per-channel symmetric quantization of K and V | Planned |
| INT4 Quantization | Group-wise asymmetric quantization (group size 128) | Planned |
| SVD Truncation | Per-head SVD, keep top-r singular components | Planned |
| H2O Token Eviction | Score by cumulative attention mass, evict lowest | Planned |
| Hybrid (per-head) | Auto-assign strategy per head based on spectral profile | Planned |

## Evaluation Metrics

- **Perplexity** — Measured on WikiText-2 with per-chunk NLL collection for bootstrap variance estimation
- **Memory** — KV cache bytes, peak GPU allocation, and model-only footprint
- **Throughput** — Autoregressive decode tokens/sec with CUDA-synchronized timing

## Project Structure

```
├── run_benchmark.py          # Main benchmark entry point
├── run_profiling.py          # Spectral profiling (WIP)
├── config/
│   └── default.yaml          # Default benchmark configuration
├── strategies/
│   ├── base.py               # CompressedKVCache abstract interface
│   └── baseline.py           # FP16 no-compression baseline
├── eval/
│   ├── perplexity.py         # WikiText-2 perplexity evaluation
│   ├── memory.py             # KV cache memory measurement
│   └── throughput.py         # Decode throughput benchmarking
├── profiling/                # Attention & spectral analysis (WIP)
├── analysis/                 # Statistics & Pareto plots (WIP)
└── results/                  # Benchmark output (JSON)
```

## Setup

Requires Python 3.9+ and a CUDA-capable GPU.

```bash
git clone https://github.com/yuan557/KV-cache-reduction.git
cd KV-cache-reduction
pip install -r requirements.txt
```

## Usage

Run the full benchmark with default config (Qwen3-0.6B, baseline strategy):

```bash
python run_benchmark.py
```

Override model, strategy, or sequence lengths from the CLI:

```bash
python run_benchmark.py --model Qwen/Qwen3-4B --strategy baseline --seq-lengths 512 2048 8192
```

Use a custom config:

```bash
python run_benchmark.py --config config/default.yaml --output-dir results/raw
```

Results are saved as timestamped JSON files in `results/raw/`.

## Design

`CompressedKVCache` subclasses HuggingFace's `DynamicCache`, so compressed caches plug directly into `model(past_key_values=...)` without modifying model internals. New strategies implement `compress()`, `decompress()`, `memory_bytes()`, and `reset()` and register themselves in the strategy registry.

## Target Hardware

Single GPU, 24 GB VRAM. Default model is Qwen3-0.6B for development; Qwen3-4B is the intended full-scale target.
