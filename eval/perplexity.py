import torch
import torch.nn.functional as F
from datasets import load_dataset


def evaluate_perplexity(model, tokenizer, cache, seq_length=2048):
    """Evaluate perplexity with KV cache compression.

    Splits each chunk into prefix (builds cache) and suffix (measures loss).
    Returns per-chunk NLLs for later bootstrap analysis.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in dataset["text"] if t.strip())
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    mid = seq_length // 2
    n_chunks = input_ids.size(1) // seq_length
    nlls = []

    for i in range(n_chunks):
        chunk = input_ids[:, i * seq_length : (i + 1) * seq_length]
        prefix = chunk[:, :mid]
        suffix = chunk[:, mid:]

        cache.reset()

        with torch.no_grad():
            model(prefix, past_key_values=cache, use_cache=True)

            for layer_idx in range(len(cache.key_cache)):
                cache.compress(layer_idx)

            outputs = model(suffix, past_key_values=cache, use_cache=True)

        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = suffix[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.item())

        if (i + 1) % 10 == 0:
            print(f"    chunk {i + 1}/{n_chunks}, running ppl: {torch.exp(torch.tensor(nlls).mean()).item():.2f}")

    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    return {
        "perplexity": ppl,
        "mean_nll": sum(nlls) / len(nlls),
        "per_chunk_nll": nlls,
        "n_chunks": n_chunks,
        "seq_length": seq_length,
    }
