# bots/rl/agents/sac_agent.py
import logging
import numpy as np
from stable_baselines3 import SAC
from bots.rl.callbacks import ProgressCallback
from core.interfaces.base_rl_agent import BaseRLAgent

logger = logging.getLogger(__name__)


class SACAgent(BaseRLAgent):
    """
    Agent SAC (Soft Actor-Critic).
    Agent principal per a trading amb accions contínues.

    Why SAC for trading:
    - Maximizes return + entropy → robust exploration in noisy markets
    - Off-policy: leverages past experiences (replay buffer)
    - Continuous actions [-1, 1]: natural position sizing (short ↔ long)
    - More stable than DDPG in non-stationary financial environments
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        buffer_size: int = 100_000,
        learning_starts: int = 1000,
        gamma: float = 0.99,
        tau: float = 0.005,
        train_freq: int = 1,
        gradient_steps: int = 1,
        policy: str = "MlpPolicy",
        net_arch: list[int] | None = None,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.gamma = gamma
        self.tau = tau
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.policy = policy
        self.net_arch = net_arch or [256, 256]
        self.model = None
        self.is_trained = False

    @classmethod
    def from_config(cls, config: dict) -> "SACAgent":
        m = config["model"]
        return cls(
            learning_rate=m["learning_rate"],
            batch_size=m["batch_size"],
            buffer_size=m.get("buffer_size", 100_000),
            learning_starts=m.get("learning_starts", 1000),
            gamma=m.get("gamma", 0.99),
            tau=m.get("tau", 0.005),
            train_freq=m.get("train_freq", 1),
            gradient_steps=m.get("gradient_steps", 1),
            policy=m.get("policy", "MlpPolicy"),
            net_arch=m.get("net_arch", [256, 256]),
        )

    def train(self, train_env, val_env, total_timesteps: int) -> dict:
        policy_kwargs = {"net_arch": self.net_arch}

        self.model = SAC(
            self.policy,
            train_env,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            gamma=self.gamma,
            tau=self.tau,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )
        logger.info(f"  Training SAC — {total_timesteps} timesteps...")
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
        logger.info(f"SACAgent saved to {path}")

    def load(self, path: str) -> None:
        self.model = SAC.load(path)
        self.is_trained = True
        logger.info(f"SACAgent loaded from {path}")

    def _validate(self, env) -> dict:
        obs, _ = env.reset()
        done = False
        portfolio_values = [env.initial_capital]

        while not done:
            action = self.act(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            portfolio_values.append(info["portfolio_value"])

        values = np.array(portfolio_values)
        total_return = (values[-1] - env.initial_capital) / env.initial_capital * 100
        peak = np.maximum.accumulate(values)
        drawdown = float(((values - peak) / peak * 100).min())

        logger.info(
            f"  SAC Validation → return={total_return:.2f}% "
            f"drawdown={drawdown:.2f}% trades={env.trades}"
        )
        return {
            "val_return_pct": round(total_return, 2),
            "val_max_drawdown_pct": round(drawdown, 2),
            "val_trades": env.trades,
            "val_final_capital": round(float(values[-1]), 2),
        }