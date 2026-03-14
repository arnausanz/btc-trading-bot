# core/interfaces/base_bot.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from core.models import Signal


@dataclass
class ObservationSchema:
    """
    Declares the data a bot needs to function.
    The Runner uses this schema to build the observation automatically.
    """
    features: list[str]                    # DataFrame columns required: 'close', 'rsi_14', ...
    timeframes: list[str]                  # timeframes required: '1h', '4h', ...
    lookback: int                          # number of historical candles the bot needs
    extras: dict[str, Any] = field(default_factory=dict)  # external data sources: sentiment, onchain, ...


class BaseBot(ABC):
    """
    Interface that every bot must implement.
    The Runner is bot-agnostic — it only calls observation_schema and on_observation.
    """

    def __init__(self, bot_id: str, config: dict):
        self.bot_id = bot_id
        # For classical bots (flat config, category == "classic"), apply any
        # best_params overrides persisted by BotOptimizer.save_best_config().
        # RL / ML bots apply best_params earlier (in RLBot / MLBot __init__)
        # where the full nested YAML structure is still intact.
        if config.get("best_params") and config.get("category", "classic").lower() == "classic":
            from core.config_utils import apply_best_params
            config = apply_best_params(config)
        self.config = config

    @abstractmethod
    def observation_schema(self) -> ObservationSchema:
        """
        Declares what data this bot needs.
        Called once during bot initialization.
        """
        ...

    @abstractmethod
    def on_observation(self, observation: dict) -> Signal:
        """
        Receives the observation and returns a trading decision.
        This is the bot's core decision-making logic.
        """
        ...

    def on_start(self) -> None:
        """Optional hook: called when the bot starts."""
        pass

    def on_stop(self) -> None:
        """Optional hook: called when the bot stops."""
        pass