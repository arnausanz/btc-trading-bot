# bots/rl/rl_bot.py
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.interfaces.base_rl_agent import BaseRLAgent
from core.models import Signal, Action
from bots.rl.agents import SACAgent, PPOAgent

logger = logging.getLogger(__name__)

_AGENT_REGISTRY: dict[str, type[BaseRLAgent]] = {
    "sac": SACAgent,
    "ppo": PPOAgent,
}


class RLBot(BaseBot):
    """
    RL agent-agnostic bot.
    Parallel to MLBot but for reinforcement learning agents.
    Adding a new agent = adding an entry to _AGENT_REGISTRY.
    """

    def __init__(self, config_path: str = "config/models/ppo.yaml"):
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        # Construïm config plana des del YAML unificat.
        # features.select i features.lookback es mapegen a top-level per
        # compatibilitat amb observation_schema() i on_observation().
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
            # Pass external config so ObservationBuilder loads the same data
            # sources as used during training. Must match data.features.external
            # in the corresponding training config.
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

        # Act: PPO → acció discreta (int 0/1/2); SAC → fracció contínua [0,1]
        raw_action = self._agent.act(obs_flat, deterministic=True)
        agent_type = self.config["model_type"]

        if agent_type == "sac":
            # Continuous: fracció desitjada de portfolio en BTC
            fraction = float(np.squeeze(raw_action))
            if fraction > 0.6:
                action, trade_size = 1, fraction      # BUY proporcional
            elif fraction < 0.4:
                action, trade_size = 2, 1.0           # SELL total
            else:
                action, trade_size = 0, 0.0           # HOLD
        else:
            # Discrete (PPO): 0=HOLD, 1=BUY, 2=SELL
            action = int(np.squeeze(raw_action))
            trade_size = self.config["trade_size"]

        if action == 1:
            return Signal(
                bot_id=self.bot_id,
                timestamp=datetime.now(timezone.utc),
                action=Action.BUY,
                size=trade_size,
                confidence=1.0,
                reason="RL agent: BUY",
            )
        elif action == 2:
            return Signal(
                bot_id=self.bot_id,
                timestamp=datetime.now(timezone.utc),
                action=Action.SELL,
                size=1.0,
                confidence=1.0,
                reason="RL agent: SELL",
            )

        return Signal(
            bot_id=self.bot_id,
            timestamp=datetime.now(timezone.utc),
            action=Action.HOLD,
            size=0.0,
            confidence=1.0,
            reason="RL agent: HOLD",
        )

    def on_start(self) -> None:
        logger.info(f"RLBot initialized with agent: {self.config['model_type']}")