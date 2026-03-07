# core/interfaces/base_rl_agent.py
from abc import ABC, abstractmethod
import numpy as np


class BaseRLAgent(ABC):
    """
    Base interface for all RL agents in the project.
    Parallel to BaseMLModel but for reinforcement learning.

    Key difference vs BaseMLModel:
      - train(env) instead of train(X, y) → learns through interaction
      - act(obs) instead of predict(X) → continuous or discrete action
      - No threshold — the agent decides directly

    Adding a new agent = inherit BaseRLAgent
    + add to the registry in train_rl.py
    """

    is_trained: bool = False

    @abstractmethod
    def train(self, train_env, val_env, total_timesteps: int) -> dict:
        """
        Trains the agent and returns validation metrics:
        {val_return_pct, val_max_drawdown_pct, val_trades, val_final_capital}
        """
        ...

    @abstractmethod
    def act(self, observation: np.ndarray, deterministic: bool = True) -> int | np.ndarray:
        """
        Returns the action for a given observation.
        deterministic=True for production, False for exploration.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Saves the agent to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Loads the agent from disk."""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "BaseRLAgent":
        """
        Constructor from YAML config.
        Same pattern as BaseMLModel.from_config().
        """
        ...