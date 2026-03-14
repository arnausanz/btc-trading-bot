# bots/rl/rl_bot.py
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.interfaces.base_rl_agent import BaseRLAgent
from core.models import Signal, Action
from bots.rl.agents import SACAgent, PPOAgent, TD3Agent

logger = logging.getLogger(__name__)

_AGENT_REGISTRY: dict[str, type[BaseRLAgent]] = {
    "sac":              SACAgent,
    "ppo":              PPOAgent,
    "ppo_professional": PPOAgent,
    "sac_professional": SACAgent,
    # C3: TD3 variants
    "td3_professional": TD3Agent,
    "td3_multiframe":   TD3Agent,
}


class RLBot(BaseBot):
    """
    RL agent-agnostic bot.
    Parallel to MLBot but for reinforcement learning agents.
    Adding a new agent = adding an entry to _AGENT_REGISTRY.
    """

    def __init__(self, config_path: str = "config/models/ppo.yaml"):
        from core.config_utils import apply_best_params
        with open(config_path) as f:
            raw = apply_best_params(yaml.safe_load(f))

        features_cfg = raw["features"]
        config = {
            **raw,
            **raw.get("bot", {}),
            "model_path": raw["training"]["model_path"],
            "features": features_cfg["select"],
            "lookback": features_cfg["lookback"],
            "external": features_cfg.get("external", {}),
        }
        super().__init__(bot_id=config["bot_id"], config=config)
        self._agent = self._load_agent()
        self._observation_buffer: list[np.ndarray] = []

    def _load_agent(self) -> BaseRLAgent:
        agent_type = self.config["model_type"]
        model_path = self.config["model_path"]

        if agent_type not in _AGENT_REGISTRY:
            raise ValueError(
                f"Unknown agent: '{agent_type}'. "
                f"Available: {list(_AGENT_REGISTRY.keys())}"
            )

        agent = _AGENT_REGISTRY[agent_type]()
        agent.load(model_path)
        logger.info(f"RLBot loaded: {agent_type} from {model_path}")
        return agent

    def observation_schema(self) -> ObservationSchema:
        return ObservationSchema(
            features=self.config["features"],
            timeframes=[self.config["timeframe"]],
            lookback=self.config["lookback"],
            extras={"external": self.config.get("external", {})},
        )

    def on_observation(self, observation: dict) -> Signal:
        timeframe = self.config["timeframe"]
        features: pd.DataFrame = observation[timeframe]["features"]

        # Construct flattened observation same as BtcTradingEnv._get_observation()
        window = features[self.config["features"]].iloc[-self.config["lookback"]:]
        obs = window.values.astype(np.float32)
        col_std = obs.std(axis=0)
        col_std[col_std == 0] = 1.0
        obs = (obs - obs.mean(axis=0)) / col_std
        obs_flat = obs.flatten()

        # Act: PPO -> discrete action; SAC/TD3 -> continuous fraction [0,1]
        raw_action = self._agent.act(obs_flat, deterministic=True)
        agent_type = self.config["model_type"]

        ts = datetime.now(timezone.utc)

        if agent_type == "ppo_professional":
            # Discrete(5): 0=HOLD, 1=BUY_FULL, 2=BUY_PARTIAL, 3=SELL_PARTIAL, 4=SELL_FULL
            action = int(np.squeeze(raw_action))
            if action == 1:
                return Signal(bot_id=self.bot_id, timestamp=ts,
                              action=Action.BUY,  size=1.0,  confidence=1.0,
                              reason="PPO-pro: BUY_FULL")
            elif action == 2:
                return Signal(bot_id=self.bot_id, timestamp=ts,
                              action=Action.BUY,  size=0.5,  confidence=1.0,
                              reason="PPO-pro: BUY_PARTIAL")
            elif action == 3:
                return Signal(bot_id=self.bot_id, timestamp=ts,
                              action=Action.SELL, size=0.5,  confidence=1.0,
                              reason="PPO-pro: SELL_PARTIAL")
            elif action == 4:
                return Signal(bot_id=self.bot_id, timestamp=ts,
                              action=Action.SELL, size=1.0,  confidence=1.0,
                              reason="PPO-pro: SELL_FULL")
            else:  # HOLD
                return Signal(bot_id=self.bot_id, timestamp=ts,
                              action=Action.HOLD, size=0.0,  confidence=1.0,
                              reason="PPO-pro: HOLD")

        elif agent_type in ("sac", "sac_professional", "td3_professional", "td3_multiframe"):
            # Continuous [0,1]: target fraction of portfolio in BTC
            fraction = float(np.squeeze(raw_action))
            agent_prefix = agent_type.upper()
            if fraction > 0.6:
                return Signal(bot_id=self.bot_id, timestamp=ts,
                              action=Action.BUY,  size=fraction, confidence=1.0,
                              reason=f"{agent_prefix}: BUY {fraction:.2f}")
            elif fraction < 0.4:
                return Signal(bot_id=self.bot_id, timestamp=ts,
                              action=Action.SELL, size=1.0,      confidence=1.0,
                              reason=f"{agent_prefix}: SELL (target={fraction:.2f})")
            else:
                return Signal(bot_id=self.bot_id, timestamp=ts,
                              action=Action.HOLD, size=0.0,      confidence=1.0,
                              reason=f"{agent_prefix}: HOLD (target={fraction:.2f})")

        else:
            # Discrete (PPO baseline): 0=HOLD, 1=BUY, 2=SELL
            action = int(np.squeeze(raw_action))
            trade_size = self.config["trade_size"]
            if action == 1:
                return Signal(bot_id=self.bot_id, timestamp=ts,
                              action=Action.BUY,  size=trade_size, confidence=1.0,
                              reason="RL agent: BUY")
            elif action == 2:
                return Signal(bot_id=self.bot_id, timestamp=ts,
                              action=Action.SELL, size=1.0,        confidence=1.0,
                              reason="RL agent: SELL")

        return Signal(bot_id=self.bot_id, timestamp=ts,
                      action=Action.HOLD, size=0.0, confidence=1.0,
                      reason="RL agent: HOLD")

    def on_start(self) -> None:
        logger.info(f"RLBot initialized with agent: {self.config['model_type']}")
