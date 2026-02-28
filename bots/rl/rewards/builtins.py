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
    Reward bàsic: canvi percentual del portfolio.
    Simple i directe — l'agent maximitza el retorn total.
    """
    return (curr_value - prev_value) / prev_value * scaling


@register("sharpe")
def sharpe_reward(
    prev_value: float,
    curr_value: float,
    action: int,
    in_position: bool,
    history: list[float],
    scaling: float = 100.0,
) -> float:
    """
    Reward basat en Sharpe incremental.
    Premia retorn consistent i penalitza volatilitat.
    Necessita almenys 20 steps d'historial per ser significatiu.
    """
    ret = (curr_value - prev_value) / prev_value
    if len(history) < 20:
        return ret * scaling
    returns = np.diff(history[-20:]) / history[-20:-1]
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
    """
    Reward basat en Sortino — com Sharpe però només penalitza
    la volatilitat negativa. Més adequat per trading asimètric.
    """
    ret = (curr_value - prev_value) / prev_value
    if len(history) < 20:
        return ret * scaling
    returns = np.diff(history[-20:]) / history[-20:-1]
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
    Com simple però penalitza estar massa temps sense posició.
    Evita que l'agent aprengui a fer HOLD sempre.
    """
    ret = (curr_value - prev_value) / prev_value * scaling
    if action == 0 and not in_position:
        ret -= 0.001  # petita penalització per inacció prolongada
    return ret