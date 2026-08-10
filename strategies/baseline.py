import torch

from strategies.base import CompressedKVCache


class BaselineCache(CompressedKVCache):
    """FP16 passthrough — no compression. Control group."""

    def compress(self, layer_idx: int):
        pass

    def decompress(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
