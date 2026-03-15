# bots/rl/agents/ppo_agent.py
import logging
import numpy as np
from stable_baselines3 import PPO
from bots.rl.callbacks import ProgressCallback
from core.interfaces.base_rl_agent import BaseRLAgent
from core.backtesting.agent_validator import validate_agent

logger = logging.getLogger(__name__)


class PPOAgent(BaseRLAgent):
    """
    Agent PPO (Proximal Policy Optimization).
    Bon baseline: estable, ben documentat, funciona amb accions discretes.
    Usat com a comparativa contra SAC.
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        n_steps: int = 2048,
        n_epochs: int = 10,
        gamma: float = 0.99,
        policy: str = "MlpPolicy",
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.policy = policy
        self.model = None
        self.is_trained = False

    @classmethod
    def from_config(cls, config: dict) -> "PPOAgent":
        m = config["model"]
        return cls(
            learning_rate=m["learning_rate"],
            batch_size=m["batch_size"],
            n_steps=m.get("n_steps", 2048),
            n_epochs=m.get("n_epochs", 10),
            gamma=m.get("gamma", 0.99),
            policy=m.get("policy", "MlpPolicy"),
        )

    def train(self, train_env, val_env, total_timesteps: int) -> dict:
        self.model = PPO(
            self.policy,
            train_env,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            n_steps=self.n_steps,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            verbose=0,
        )
        logger.info(f"  Training PPO — {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=ProgressCallback(total_timesteps),
        )
        self.is_trained = True
        return self._validate(val_env)

    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Agent not trained. Call train() first.")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        self.model.save(path)
        logger.info(f"PPOAgent saved to {path}")

    def load(self, path: str) -> None:
        self.model = PPO.load(path)
        self.is_trained = True
        logger.info(f"PPOAgent loaded from {path}")

    def _validate(self, env) -> dict:
        return validate_agent(self, env)