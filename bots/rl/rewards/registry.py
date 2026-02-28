# bots/rl/rewards/registry.py
from typing import Callable

# Tipus d'una reward function:
# rep (prev_value, curr_value, action, in_position) i retorna un float
RewardFn = Callable[[float, float, int, bool], float]

_REGISTRY: dict[str, RewardFn] = {}


def register(name: str):
    """Decorator per registrar una reward function."""
    def decorator(fn: RewardFn) -> RewardFn:
        _REGISTRY[name] = fn
        return fn
    return decorator


def get(name: str) -> RewardFn:
    if name not in _REGISTRY:
        raise ValueError(
            f"Reward function '{name}' no trobada. "
            f"Disponibles: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]