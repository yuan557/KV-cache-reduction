from abc import ABC, abstractmethod

import torch
from transformers import DynamicCache


class CompressedKVCache(DynamicCache, ABC):
    """Base class for all KV cache compression strategies.

    Subclasses DynamicCache so it can be passed directly to HuggingFace
    model forward passes via past_key_values.
    """

    @abstractmethod
    def compress(self, layer_idx: int):
        """Compress the KV cache for the given layer in-place."""

    @abstractmethod
    def decompress(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return decompressed (key, value) for the given layer."""

    def memory_bytes(self) -> int:
        """Total GPU memory used by the cache."""
        total = 0
        for layer_keys, layer_values in zip(self.key_cache, self.value_cache):
            total += layer_keys.nelement() * layer_keys.element_size()
            total += layer_values.nelement() * layer_values.element_size()
        return total

    def reset(self):
        """Clear all cached state."""
        self.key_cache.clear()
        self.value_cache.clear()
        self._seen_tokens = 0
