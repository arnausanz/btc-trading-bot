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

    def __init__(self, config_path: str = "config/bots/rl_bot.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
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

        # Act: acció discreta (0=hold, 1=buy, 2=sell)
        action = int(self._agent.act(obs_flat, deterministic=True))

        if action == 1:
            return Signal(
                bot_id=self.bot_id,
                timestamp=datetime.now(timezone.utc),
                action=Action.BUY,
                size=self.config["trade_size"],
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