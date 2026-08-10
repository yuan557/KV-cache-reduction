import argparse
import json
import os
from datetime import datetime

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.memory import measure_memory
from eval.perplexity import evaluate_perplexity
from eval.throughput import measure_throughput
from strategies import get_strategy


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def print_summary(results):
    for strat, data in results.items():
        print(f"\n{'=' * 50}")
        print(f"Strategy: {strat}")
        print(f"{'=' * 50}")
        if "throughput" in data:
            tp = data["throughput"]
            print(f"  Throughput: {tp['tokens_per_second']:.1f} tok/s")
        for key, val in data.items():
            if key == "throughput":
                continue
            print(f"  seq_length={key}:")
            print(f"    Perplexity: {val['perplexity']['perplexity']:.2f}")
            print(f"    Cache memory: {val['memory']['cache_bytes'] / 1e6:.1f} MB")
            print(f"    Peak GPU: {val['memory']['peak_gpu_bytes'] / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="KV Cache Compression Benchmark")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--strategy", nargs="+", default=None)
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", default="results/raw")
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = args.model or config["model"]["name"]
    dtype = getattr(torch, config["model"]["dtype"])
    device = config["model"]["device"]
    seq_lengths = args.seq_lengths or config["eval"]["sequence_lengths"]
    strategy_names = args.strategy or config["strategies"]

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype
    ).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    for strat_name in strategy_names:
        print(f"\nRunning strategy: {strat_name}")
        cache = get_strategy(strat_name)
        results[strat_name] = {}

        for seq_len in seq_lengths:
            print(f"  seq_length={seq_len}")
            ppl = evaluate_perplexity(model, tokenizer, cache, seq_len)
            mem = measure_memory(model, tokenizer, cache, seq_len)
            results[strat_name][seq_len] = {"perplexity": ppl, "memory": mem}

        print("  Measuring throughput...")
        tp = measure_throughput(model, tokenizer, cache)
        results[strat_name]["throughput"] = tp
        print(f"  Throughput: {tp['tokens_per_second']:.1f} tok/s")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"benchmark_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
