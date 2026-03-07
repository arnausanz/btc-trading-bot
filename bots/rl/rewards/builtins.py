# bots/rl/rewards/builtins.py
import numpy as np
from bots.rl.rewards.registry import register


@register("simple")
def simple_reward(
    prev_value: float,
    curr_value: float,
    action: int,
    in_position: bool,
    history: list[float],
    scaling: float = 100.0,
) -> float:
    """
    Basic reward: percentage change of the portfolio.
    Simple and direct — the agent maximizes total return.
    """
    return (curr_value - prev_value) / prev_value * scaling


# bots/rl/rewards/builtins.py — canvia sharpe_reward i sortino_reward

@register("sharpe")
def sharpe_reward(
    prev_value: float,
    curr_value: float,
    action: int,
    in_position: bool,
    history: list[float],
    scaling: float = 100.0,
) -> float:
    if prev_value <= 0:
        return 0.0
    ret = (curr_value - prev_value) / prev_value
    if len(history) < 20:
        return ret * scaling
    returns = np.diff(history[-20:]) / np.maximum(history[-20:-1], 1e-8)
    std = returns.std()
    if std == 0:
        return 0.0
    return float((ret / std) * scaling * 0.1)


@register("sortino")
def sortino_reward(
    prev_value: float,
    curr_value: float,
    action: int,
    in_position: bool,
    history: list[float],
    scaling: float = 100.0,
) -> float:
    if prev_value <= 0:
        return 0.0
    ret = (curr_value - prev_value) / prev_value
    if len(history) < 20:
        return ret * scaling
    returns = np.diff(history[-20:]) / np.maximum(history[-20:-1], 1e-8)
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return ret * scaling
    return float((ret / downside.std()) * scaling * 0.1)


@register("penalize_inaction")
def penalize_inaction_reward(
    prev_value: float,
    curr_value: float,
    action: int,
    in_position: bool,
    history: list[float],
    scaling: float = 100.0,
) -> float:
    """
    Like simple but penalizes extended periods without a position.
    Prevents the agent from learning to always HOLD.
    """
    ret = (curr_value - prev_value) / prev_value * scaling
    if action == 0 and not in_position:
        ret -= 0.001  # small penalty for prolonged inaction
    return ret