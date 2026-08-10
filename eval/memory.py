import torch


def measure_memory(model, tokenizer, cache, seq_length=2048):
    """Measure KV cache memory and peak GPU memory at a given sequence length."""
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (1, seq_length), device=model.device
    )

    cache.reset()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True)
        for layer_idx in range(len(cache.key_cache)):
            cache.compress(layer_idx)

    cache_bytes = cache.memory_bytes()
    peak_bytes = torch.cuda.max_memory_allocated()

    return {
        "cache_bytes": cache_bytes,
        "peak_gpu_bytes": peak_bytes,
        "model_bytes": mem_before,
        "seq_length": seq_length,
    }
