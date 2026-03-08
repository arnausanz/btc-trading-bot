# bots/rl/trainer.py
import logging
import mlflow
import yaml
from bots.rl.agents import SACAgent, PPOAgent
from bots.rl.environment import BtcTradingEnvDiscrete, BtcTradingEnvContinuous
from bots.rl.rewards import builtins  # registers builtin reward functions
from core.interfaces.base_rl_agent import BaseRLAgent
from data.processing.feature_builder import FeatureBuilder
from core.config import MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)

# Adding a new agent = one line in each registry
_AGENT_REGISTRY: dict[str, type[BaseRLAgent]] = {
    "sac": SACAgent,
    "ppo": PPOAgent,
}

_ENV_REGISTRY: dict[str, type] = {
    "sac": BtcTradingEnvContinuous,
    "ppo": BtcTradingEnvDiscrete,
}


class RLTrainer:
    """
    Trains any RL agent registered in _AGENT_REGISTRY.

    ── FEATURE CONSISTENCY ────────────────────────────────────────────────────
    The training config's data.features.select determines EXACTLY which columns
    are fed to the environment observation. This must match the bot deployment
    config's features list to ensure identical obs_shape at train and inference:

        obs_shape = len(data.features.select) × environment.lookback

    If data.features.select is null/omitted, ALL columns from FeatureBuilder
    are used (old behavior, not recommended for RL).

    ── EXTERNAL FEATURES ──────────────────────────────────────────────────────
    Configure external data sources in data.features.external (see FeatureBuilder
    docstring). The matching bot config must declare the same external section
    so ObservationBuilder loads the same data at inference time.
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

            env_cfg = self.config["environment"]
            features_cfg = self.config["data"].get("features", {})
            select = features_cfg.get("select")

            # Build feature DataFrame via FeatureBuilder (consistent with deployment)
            fb = FeatureBuilder.from_config(self.config["data"])
            df = fb.build()

            n_features = len(df.columns)
            obs_shape = env_cfg["lookback"] * n_features
            logger.info(
                f"Features: {n_features} columns, lookback={env_cfg['lookback']}, "
                f"obs_shape={obs_shape}"
            )
            if select:
                logger.info(f"  Selected: {select}")
            else:
                logger.warning(
                    "  data.features.select is not configured — using ALL columns. "
                    "This may cause obs_shape mismatch at inference if the bot config "
                    "specifies a feature subset. Consider adding features.select."
                )

            mlflow.log_params({
                "model_type": agent_type,
                "total_timesteps": self.config["model"]["total_timesteps"],
                "lookback": env_cfg["lookback"],
                "reward_type": env_cfg["reward_type"],
                "timeframe": self.config["data"]["timeframe"],
                "n_features": n_features,
                "obs_shape": obs_shape,
            })

            split = int(len(df) * self.config["data"]["train_pct"])
            df_train = df.iloc[:split]
            df_val = df.iloc[split:]
            logger.info(f"Train: {len(df_train)} rows | Val: {len(df_val)} rows")

            env_class = _ENV_REGISTRY[agent_type]
            train_env = env_class(df=df_train, **env_cfg)
            val_env = env_class(df=df_val, **env_cfg)

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
