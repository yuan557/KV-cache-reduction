import time

import torch


def _generate(model, input_ids, cache, max_new_tokens):
    """Manual autoregressive loop so we can compress after each step."""
    cur = input_ids
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(cur, past_key_values=cache, use_cache=True)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for layer_idx in range(len(cache.key_cache)):
            cache.compress(layer_idx)
        cur = next_token


def measure_throughput(model, tokenizer, cache, num_tokens=128, warmup_runs=3):
    """Measure decode throughput in tokens/second."""
    prompt_ids = tokenizer(
        "The quick brown fox jumps over the lazy dog",
        return_tensors="pt",
    ).input_ids.to(model.device)

    for _ in range(warmup_runs):
        cache.reset()
        _generate(model, prompt_ids, cache, num_tokens)

    cache.reset()
    torch.cuda.synchronize()
    start = time.perf_counter()
    _generate(model, prompt_ids, cache, num_tokens)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "tokens_per_second": num_tokens / elapsed,
        "num_tokens": num_tokens,
        "elapsed_seconds": round(elapsed, 3),
    }
