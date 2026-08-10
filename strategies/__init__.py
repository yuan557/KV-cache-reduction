from strategies.base import CompressedKVCache
from strategies.baseline import BaselineCache

STRATEGY_REGISTRY = {
    "baseline": BaselineCache,
}


def get_strategy(name, **kwargs):
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name](**kwargs)
