# bots/rl/trainer.py
import logging
import mlflow
import yaml
from bots.rl.agents import SACAgent, PPOAgent
from bots.rl.environment import BtcTradingEnvDiscrete, BtcTradingEnvContinuous
from bots.rl.rewards import builtins  # registra les builtins
from core.interfaces.base_rl_agent import BaseRLAgent
from data.processing.technical import compute_features
from core.config import MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)

# Adding a new agent = one line here
_AGENT_REGISTRY: dict[str, type[BaseRLAgent]] = {
    "sac": SACAgent,
    "ppo": PPOAgent,
}

# Agent → environment map
_ENV_REGISTRY: dict[str, type] = {
    "sac": BtcTradingEnvContinuous,
    "ppo": BtcTradingEnvDiscrete,
}


class RLTrainer:
    """
    Trains any RL agent registered in _AGENT_REGISTRY.
    Same pattern as train_models.py for supervised models.
    Complete config in a single YAML: environment + agent + data + output.
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def run(self) -> dict:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.config["experiment_name"])
        mlflow.end_run()

        with mlflow.start_run():
            agent_type = self.config["model_type"]
            if agent_type not in _AGENT_REGISTRY:
                raise ValueError(
                    f"Unknown agent: '{agent_type}'. "
                    f"Available: {list(_AGENT_REGISTRY.keys())}"
                )

            mlflow.log_params({
                "model_type": agent_type,
                "total_timesteps": self.config["model"]["total_timesteps"],
                "lookback": self.config["environment"]["lookback"],
                "reward_type": self.config["environment"]["reward_type"],
                "timeframe": self.config["data"]["timeframe"],
            })

            # Data
            df = compute_features(
                symbol=self.config["data"]["symbol"],
                timeframe=self.config["data"]["timeframe"],
            )
            split = int(len(df) * self.config["data"]["train_pct"])
            df_train = df.iloc[:split]
            df_val = df.iloc[split:]
            logger.info(f"Train: {len(df_train)} rows | Val: {len(df_val)} rows")

            # Environments
            env_cfg = self.config["environment"]  # ← this line was missing
            env_class = _ENV_REGISTRY[agent_type]
            train_env = env_class(df=df_train, **env_cfg)
            val_env = env_class(df=df_val, **env_cfg)

            # Agent via from_config — same pattern as supervised models
            agent = _AGENT_REGISTRY[agent_type].from_config(self.config)
            metrics = agent.train(
                train_env=train_env,
                val_env=val_env,
                total_timesteps=self.config["model"]["total_timesteps"],
            )

            mlflow.log_metrics(metrics)
            agent.save(self.config["output"]["model_path"])

            logger.info(
                f"  ✓ return={metrics['val_return_pct']}% "
                f"drawdown={metrics['val_max_drawdown_pct']}% "
                f"trades={metrics['val_trades']}"
            )
            return metrics