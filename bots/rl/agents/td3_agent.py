# bots/rl/agents/td3_agent.py
"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) agent.

Why TD3 over SAC for noisy financial markets:
─────────────────────────────────────────────
  Twin Q-networks    Maintains two independent critic networks and uses the
                     minimum of their estimates → eliminates SAC's tendency to
                     overestimate Q-values in stochastic reward environments.

  Delayed updates    Policy (actor) and target networks are updated only every
                     `policy_delay` critic steps (default 2). This decouples
                     the noisy critic gradient signal from the policy gradient,
                     reducing variance in non-stationary financial time series.

  Target smoothing   Adds Gaussian noise to target actions when computing the
                     Bellman target: Q(s', π(s') + ε).  This smooths the value
                     landscape and prevents the policy from exploiting sharp
                     reward spikes common in financial data (e.g. sudden BTC
                     price moves).

When to prefer TD3 vs SAC:
  - SAC:  better when exploration is critical; entropy bonus helps escape local
          optima in long training runs.
  - TD3:  better when reward signal is noisy; twin critics give more stable
          training in high-variance environments.  Lower hyperparameter
          sensitivity than SAC.

References:
  Fujimoto et al., 2018 — "Addressing Function Approximation Error in
  Actor-Critic Methods" (https://arxiv.org/abs/1802.09477)
  Stable-Baselines3: stable-baselines3.readthedocs.io/en/master/modules/td3.html
"""
import logging
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from bots.rl.callbacks import ProgressCallback
from core.interfaces.base_rl_agent import BaseRLAgent

logger = logging.getLogger(__name__)


class TD3Agent(BaseRLAgent):
    """
    TD3 (Twin Delayed DDPG) agent.

    Wraps Stable-Baselines3 TD3 with the same interface as SACAgent /
    PPOAgent so it can be plugged into RLTrainer without changes.

    Key TD3 hyperparameters:
      policy_delay         Frequency ratio of critic vs actor updates.
                           Default 2: actor updates every 2nd critic step.
      target_policy_noise  Std dev of noise added to target actions (smoothing).
      target_noise_clip    Hard clip of target noise (prevents extreme values).
      action_noise_sigma   Std dev of exploration noise at training time.
                           Scaled to action space size [0, 1].

    Recommended starting points for BTC swing trading:
      learning_rate = 1e-3 to 3e-4
      batch_size    = 256  (replay buffer allows larger batches than PPO)
      buffer_size   = 200_000
      policy_delay  = 2 (default)
      target_policy_noise = 0.2  (high enough to smooth BTC noise)
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        buffer_size: int = 200_000,
        learning_starts: int = 1000,
        gamma: float = 0.99,
        tau: float = 0.005,
        train_freq: int = 1,
        gradient_steps: int = 1,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        action_noise_sigma: float = 0.1,
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
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.action_noise_sigma = action_noise_sigma
        self.policy = policy
        self.net_arch = net_arch or [256, 256]
        self.model = None
        self.is_trained = False

    @classmethod
    def from_config(cls, config: dict) -> "TD3Agent":
        """
        Builds TD3Agent from the `training` section of a unified YAML.

        Reads from config["model"], with sensible defaults for all TD3-
        specific hyperparameters so existing YAML schemas remain compatible.
        """
        m = config["model"]
        return cls(
            learning_rate=m["learning_rate"],
            batch_size=m["batch_size"],
            buffer_size=m.get("buffer_size", 200_000),
            learning_starts=m.get("learning_starts", 1000),
            gamma=m.get("gamma", 0.99),
            tau=m.get("tau", 0.005),
            train_freq=m.get("train_freq", 1),
            gradient_steps=m.get("gradient_steps", 1),
            policy_delay=m.get("policy_delay", 2),
            target_policy_noise=m.get("target_policy_noise", 0.2),
            target_noise_clip=m.get("target_noise_clip", 0.5),
            action_noise_sigma=m.get("action_noise_sigma", 0.1),
            policy=m.get("policy", "MlpPolicy"),
            net_arch=m.get("net_arch", [256, 256]),
        )

    def train(self, train_env, val_env, total_timesteps: int) -> dict:
        """
        Trains the TD3 model and validates on val_env.

        Action noise (NormalActionNoise) is added during training to encourage
        exploration. At inference time (deterministic=True in act()), no noise
        is applied.

        TD3 requires a 1D continuous action space (Box). If the environment
        has multi-dimensional actions, n_actions is set accordingly.
        """
        policy_kwargs = {"net_arch": self.net_arch}

        # Action noise for exploration during training
        n_actions = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=self.action_noise_sigma * np.ones(n_actions),
        )

        self.model = TD3(
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
            policy_delay=self.policy_delay,
            target_policy_noise=self.target_policy_noise,
            target_noise_clip=self.target_noise_clip,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )
        logger.info(f"  Training TD3 — {total_timesteps} timesteps...")
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
        logger.info(f"TD3Agent saved to {path}")

    def load(self, path: str) -> None:
        self.model = TD3.load(path)
        self.is_trained = True
        logger.info(f"TD3Agent loaded from {path}")

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
            f"  TD3 Validation → return={total_return:.2f}% "
            f"drawdown={drawdown:.2f}% trades={env.trades}"
        )
        return {
            "val_return_pct": round(total_return, 2),
            "val_max_drawdown_pct": round(drawdown, 2),
            "val_trades": env.trades,
            "val_final_capital": round(float(values[-1]), 2),
        }
