# KV Cache Compression Benchmark

A rigorous, reproducible benchmark comparing KV cache compression strategies for LLM inference. Built around Qwen3 4B with statistical testing, per-head profiling, and Pareto analysis.

## Why This Exists

KV cache compression research is fragmented. Papers test one strategy on one model with no shared evaluation protocol. There is no single repo that lets you swap strategies, sweep memory budgets, and get statistically grounded comparisons with one command.

This project fills that gap.

## Model

**Qwen3 4B** (single model, single GPU). Chosen for:

- Modern GQA architecture, representative of current LLMs
- Hybrid thinking/non-thinking mode produces variable-length KV caches
- Small enough to run on consumer hardware (24GB VRAM) with room for profiling
- Understudied for KV compression compared to Llama

## Compression Strategies

| Strategy | Description | Sweep Parameters |
|---|---|---|
| **FP16 Baseline** | No compression. Control group. | None |
| **INT8 Quantization** | Per-channel symmetric quantization of K and V | None |
| **INT4 Quantization** | Group-wise asymmetric quantization, group size 128 | None |
| **SVD Truncation** | Per-head SVD, keep top-r singular components | Rank ratio: 25%, 50%, 75% |
| **H2O Token Eviction** | Score tokens by cumulative attention mass, evict lowest | Retention ratio: 25%, 50%, 75% |
| **Hybrid (per-head)** | Assign best strategy per head based on spectral profile | Auto-assigned |

All strategies implement the same `CompressedKVCache` interface and are swappable via config.

## Evaluation

### Quality Metrics

- **Perplexity** on WikiText-2 and C4 validation splits
- **Downstream accuracy** via lm-eval-harness:
  - MMLU (5-shot)
  - HellaSwag (10-shot)
  - ARC-Challenge (25-shot)

### Efficiency Metrics

- Peak KV cache memory (bytes) at sequence lengths 512, 2048, 8192
- Decode throughput (tokens/sec)

### Statistical Rigor

- Bootstrap perplexity over 100 evaluation chunks for variance estimation
- Paired t-test per strategy vs baseline for perplexity
- McNemar's test for downstream task accuracy (paired, per-example)
- Benjamini-Hochberg correction for multiple comparisons
- Effect sizes (Cohen's d) reported alongside p-values
- Quality vs memory Pareto frontier plots

## Per-Head Spectral Profiling

Before running compression, the benchmark profiles every attention head:

- Run 256 calibration sequences through the model
- Capture K/V activations per layer per head
- Compute singular value decay curves
- Classify heads by compressibility (fast decay = easy to compress, flat spectrum = keep full precision)
- Output: decay plots, per-head rank recommendations, compressibility scores

This profiling drives the hybrid strategy and provides standalone analysis of Qwen3 4B's internal structure.

## Project Structure

```
kv-cache-bench/
    config/
        default.yaml              # Default benchmark config
        strategies/               # Per-strategy config overrides
    strategies/
        base.py                   # CompressedKVCache interface
        baseline.py               # FP16, no compression
        quantize.py               # INT8, INT4 variants
        svd.py                    # Per-head SVD truncation
        eviction.py               # H2O token eviction
        hybrid.py                 # Per-head strategy assignment
    profiling/
        spectral.py               # Singular value analysis per head
        attention.py              # Attention entropy and sparsity stats
    eval/
        perplexity.py             # Perplexity on WikiText-2, C4
        downstream.py             # lm-eval-harness wrapper
        memory.py                 # Peak KV cache measurement
        throughput.py             # Decode tokens/sec
    analysis/
        statistics.py             # Bootstrap, McNemar, BH correction
        pareto.py                 # Quality-memory frontier plots
        plots.py                  # Spectral decay and comparison charts
    results/                      # Auto-generated benchmark outputs
        figures/
        tables/
        raw/
    run_benchmark.py              # Single entry point
    run_profiling.py              # Spectral profiling entry point
    requirements.txt
    README.md
```

## Usage

### Setup

```bash
git clone https://github.com/<you>/kv-cache-bench.git
cd kv-cache-bench
pip install -r requirements.txt
```

### Run spectral profiling

```bash
python run_profiling.py --model Qwen/Qwen3-4B --num-calibration 256
```

Outputs per-head singular value plots to `results/figures/spectral/`.

### Run full benchmark

```bash
# All strategies, all evaluations
python run_benchmark.py --strategy all --model Qwen/Qwen3-4B

# Single strategy
python run_benchmark.py --strategy int8 --model Qwen/Qwen3-4B

# Custom sequence lengths
python run_benchmark.py --strategy all --seq-lengths 512 2048 8192
```

### Run statistical analysis

```bash
python -m analysis.statistics --results-dir results/raw/
python -m analysis.pareto --results-dir results/raw/
```

Outputs p-value tables to `results/tables/` and Pareto plots to `results/figures/`.

## Dependencies

```
torch>=2.1
transformers>=4.40
accelerate
datasets
lm-eval>=0.4
scipy
statsmodels
matplotlib
seaborn
pyyaml
```

### Hardware

- 1x GPU with 24GB VRAM (RTX 3090, 4090, A5000, or equivalent)
- 32GB system RAM recommended

## Timeline

| Phase | Days | Deliverable |
|---|---|---|
| Base cache interface + baseline eval | 1-3 | Working generation with intercepted KV cache |
| INT8 + INT4 quantization | 4-7 | Two quantization strategies, numerical verification |
| SVD + spectral profiling | 8-10 | SVD strategy, per-head decay plots |
| H2O eviction + hybrid | 11-13 | Eviction strategy, auto-assigned hybrid |
| Full benchmark sweep | 14-17 | All strategies x all configs x all evals |
| Analysis + writeup | 18-21 | p-values, Pareto plots, final README |

## What This Project Does NOT Include

- No serving infrastructure (no vLLM, no API server)
- No training or finetuning
- No multi-GPU support
- No custom CUDA/Triton kernels (that's the next project)
- No multi-model comparison (Qwen3 4B only, for now)

## Future Work

- **Triton kernel for learned low-rank projections.** If SVD or hybrid shows promise, build a fused attention kernel that compresses and decompresses K/V without materializing full tensors in HBM.
- **Multi-model validation.** Extend to Llama 3.1 8B, Mistral 7B to test generalization.
- **Integration with vLLM/SGLang.** Port the winning strategy into a production serving stack.
- **Learned projections.** Replace post-hoc SVD with trained projection matrices optimized for attention fidelity rather than reconstruction error.

## License

MIT