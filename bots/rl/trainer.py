# bots/rl/trainer.py
import logging
import yaml
from bots.rl.agents import SACAgent, PPOAgent, TD3Agent
from bots.rl.environment import BtcTradingEnvDiscrete, BtcTradingEnvContinuous
from bots.rl.environment_professional import (
    BtcTradingEnvProfessionalDiscrete,
    BtcTradingEnvProfessionalContinuous,
)
from bots.rl.rewards import builtins, professional  # registers all reward functions  # noqa: F401
from bots.rl.rewards import advanced                # registers regime_adaptive        # noqa: F401
from core.interfaces.base_rl_agent import BaseRLAgent
from data.processing.feature_builder import FeatureBuilder

logger = logging.getLogger(__name__)

# Adding a new agent = one line in each registry
_AGENT_REGISTRY: dict[str, type[BaseRLAgent]] = {
    "sac":              SACAgent,
    "ppo":              PPOAgent,
    # Professional variants (same agent algorithm, new env + reward)
    "ppo_professional": PPOAgent,
    "sac_professional": SACAgent,
    # C3: TD3 variants
    "td3_professional": TD3Agent,
    "td3_multiframe":   TD3Agent,
}

_ENV_REGISTRY: dict[str, type] = {
    "sac":              BtcTradingEnvContinuous,
    "ppo":              BtcTradingEnvDiscrete,
    "ppo_professional": BtcTradingEnvProfessionalDiscrete,
    "sac_professional": BtcTradingEnvProfessionalContinuous,
    # C3: TD3 uses professional continuous env (continuous action space)
    "td3_professional": BtcTradingEnvProfessionalContinuous,
    "td3_multiframe":   BtcTradingEnvProfessionalContinuous,
}

# Model types that use MultiFrameFeatureBuilder instead of FeatureBuilder
_MULTIFRAME_TYPES = {"td3_multiframe"}


class RLTrainer:
    """
    Trains any RL agent registered in _AGENT_REGISTRY.

    -- FEATURE CONSISTENCY ----------------------------------------------------
    The training config's data.features.select determines EXACTLY which columns
    are fed to the environment observation. This must match the bot deployment
    config's features list to ensure identical obs_shape at train and inference:

        obs_shape = len(data.features.select) x environment.lookback

    If data.features.select is null/omitted, ALL columns from FeatureBuilder
    are used (old behavior, not recommended for RL).

    -- EXTERNAL FEATURES -------------------------------------------------------
    Configure external data sources in data.features.external (see FeatureBuilder
    docstring). The matching bot config must declare the same external section
    so ObservationBuilder loads the same data at inference time.

    -- MULTI-TIMEFRAME --------------------------------------------------------
    For model_types in _MULTIFRAME_TYPES (e.g. td3_multiframe), the trainer
    uses MultiFrameFeatureBuilder instead of FeatureBuilder. The YAML must
    include aux_timeframes: [4h] and features.select must list auxiliary
    features with the appropriate suffix (e.g. rsi_14_4h).
    """

    def __init__(self, config_path: str):
        from core.config_utils import apply_best_params
        with open(config_path) as f:
            self.config = apply_best_params(yaml.safe_load(f))

    def run(self) -> dict:
        train_cfg = self.config["training"]
        features_cfg = self.config["features"]

        agent_type = self.config["model_type"]
        if agent_type not in _AGENT_REGISTRY:
            raise ValueError(
                f"Unknown agent: '{agent_type}'. "
                f"Available: {list(_AGENT_REGISTRY.keys())}"
            )

        env_cfg = train_cfg["environment"]
        env_cfg = {**env_cfg, "lookback": features_cfg["lookback"]}

        # Build feature DataFrame
        if agent_type in _MULTIFRAME_TYPES:
            from data.processing.multiframe_builder import MultiFrameFeatureBuilder
            df = MultiFrameFeatureBuilder.from_config(self.config).build()
        else:
            df = FeatureBuilder.from_config(self.config).build()

        n_features = len(df.columns)
        lookback = features_cfg["lookback"]
        # Professional envs add 4 position-state dims: obs_shape = n_feats×lookback + 4
        _PROFESSIONAL_ENV_TYPES = {"ppo_professional", "sac_professional", "td3_professional", "td3_multiframe"}
        pos_state_dims = 4 if agent_type in _PROFESSIONAL_ENV_TYPES else 0
        obs_shape = lookback * n_features + pos_state_dims
        logger.info(
            f"Features: {n_features} columns, lookback={lookback}, "
            f"obs_shape={obs_shape}"
            + (f" ({n_features}×{lookback} + {pos_state_dims} pos state)" if pos_state_dims else "")
        )
        logger.info(f"  Selected: {features_cfg.get('select')}")
        logger.info(
            f"  model_type={agent_type} | "
            f"total_timesteps={train_cfg['model']['total_timesteps']} | "
            f"reward_type={env_cfg['reward_type']} | "
            f"timeframe={self.config['timeframe']}"
        )

        split = int(len(df) * train_cfg["train_pct"])
        df_train = df.iloc[:split]
        df_val = df.iloc[split:]
        logger.info(f"Train: {len(df_train)} rows | Val: {len(df_val)} rows")

        # Minimum data guard
        lookback = features_cfg["lookback"]
        min_rows = lookback + 10
        if len(df_train) < min_rows:
            model_type = self.config.get("model_type", "?")
            raise ValueError(
                f"Insufficient training data: {len(df_train)} rows but "
                f"lookback={lookback} requires at least {min_rows} rows.\n"
                f"Likely cause: an external data source has limited history "
                f"and causes most rows to be dropped via dropna().\n"
                f"Fix: remove the problematic source from features.external in "
                f"config/models/{model_type}.yaml, or collect more historical data."
            )
        if len(df_val) < lookback + 2:
            raise ValueError(
                f"Insufficient validation data: {len(df_val)} rows but "
                f"lookback={lookback} requires at least {lookback + 2} rows.\n"
                f"Consider reducing train_pct (currently {train_cfg['train_pct']}) "
                f"or collecting more data."
            )

        env_class = _ENV_REGISTRY[agent_type]
        train_env = env_class(df=df_train, **env_cfg)
        val_env = env_class(df=df_val, **env_cfg)

        agent = _AGENT_REGISTRY[agent_type].from_config(train_cfg)
        metrics = agent.train(
            train_env=train_env,
            val_env=val_env,
            total_timesteps=train_cfg["model"]["total_timesteps"],
        )

        agent.save(train_cfg["model_path"])

        logger.info(
            f"  OK return={metrics['val_return_pct']}% "
            f"drawdown={metrics['val_max_drawdown_pct']}% "
            f"trades={metrics['val_trades']}"
        )
        return metrics
