# core/interfaces/base_rl_agent.py
from abc import ABC, abstractmethod
import numpy as np


class BaseRLAgent(ABC):
    """
    Interfície base per a tots els agents RL del projecte.
    Paral·lel a BaseMLModel però per a reinforcement learning.

    La diferència clau vs BaseMLModel:
      - train(env) en comptes de train(X, y)  → aprèn per interacció
      - act(obs) en comptes de predict(X)     → acció contínua o discreta
      - No té threshold — l'agent decideix directament

    Afegir un agent nou = crear classe que hereti BaseRLAgent
    + afegir al registry de train_rl.py
    """

    is_trained: bool = False

    @abstractmethod
    def train(self, train_env, val_env, total_timesteps: int) -> dict:
        """
        Entrena l'agent i retorna mètriques de validació:
        {val_return_pct, val_max_drawdown_pct, val_trades, val_final_capital}
        """
        ...

    @abstractmethod
    def act(self, observation: np.ndarray, deterministic: bool = True) -> int | np.ndarray:
        """
        Retorna l'acció donada una observació.
        deterministic=True per a producció, False per a exploració.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Guarda l'agent a disc."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Carrega l'agent des de disc."""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "BaseRLAgent":
        """
        Constructor des de config YAML.
        Mateix patró que BaseMLModel.from_config().
        """
        ...