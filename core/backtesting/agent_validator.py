# core/backtesting/agent_validator.py
"""
Shared validation utility for RL agents (Proposal B).

Extracted from the duplicated _validate() methods in PPOAgent, SACAgent
and TD3Agent. Any new agent using this function gets identical, consistent
validation metrics without copying code.

Usage:
    from core.backtesting.agent_validator import validate_agent

    def _validate(self, env) -> dict:
        return validate_agent(self, env)
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def validate_agent(agent, env) -> dict:
    """
    Runs a full deterministic validation episode and returns standard metrics.

    Args:
        agent:  Any BaseRLAgent with an act(obs, deterministic=True) method.
        env:    A Gymnasium trading environment with initial_capital and trades attrs.

    Returns:
        {
            "val_return_pct":        float,  # total return over the episode
            "val_max_drawdown_pct":  float,  # peak-to-trough drawdown
            "val_trades":            int,    # number of trades executed
            "val_final_capital":     float,  # final portfolio value
        }
    """
    obs, _ = env.reset()
    done = False
    portfolio_values = [env.initial_capital]

    while not done:
        action = agent.act(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])

    values = np.array(portfolio_values)
    total_return = (values[-1] - env.initial_capital) / env.initial_capital * 100
    peak = np.maximum.accumulate(values)
    drawdown = float(((values - peak) / peak * 100).min())

    agent_name = type(agent).__name__
    logger.info(
        f"  {agent_name} Validation → return={total_return:.2f}% "
        f"drawdown={drawdown:.2f}% trades={env.trades}"
    )
    return {
        "val_return_pct": round(total_return, 2),
        "val_max_drawdown_pct": round(drawdown, 2),
        "val_trades": env.trades,
        "val_final_capital": round(float(values[-1]), 2),
    }
