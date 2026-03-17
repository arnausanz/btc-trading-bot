# bots/classical/grid_bot.py
import yaml
import logging
import pandas as pd
from datetime import datetime, timezone
from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.models import Signal, Action

logger = logging.getLogger(__name__)


class GridBot(BaseBot):
    """
    Grid Trading Bot that calculates Bollinger Bands dynamically.
    """

    def __init__(self, config_path: str = "config/models/grid.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        super().__init__(bot_id=config["bot_id"], config=config)
        self._in_position = False

    def observation_schema(self) -> ObservationSchema:
        return ObservationSchema(
            features=["close", "rsi_14"],
            timeframes=[self.config["timeframe"]],
            lookback=self.config["lookback"],
        )

    def on_observation(self, observation: dict) -> Signal:
        timeframe = self.config["timeframe"]
        features: pd.DataFrame = observation[timeframe]["features"].copy()

        close = features["close"]
        bb_window = self.config.get("bb_window", 20)
        bb_std = self.config.get("bb_std", 2.0)
        middle = close.rolling(window=bb_window).mean()
        std = close.rolling(window=bb_window).std()
        bb_upper = middle + bb_std * std
        bb_lower = middle - bb_std * std

        price = close.iloc[-1]
        rsi = features["rsi_14"].iloc[-1]

        if price <= bb_lower.iloc[-1] and not self._in_position and rsi < self.config["rsi_filter_high"]:
            self._in_position = True
            return Signal(
                bot_id=self.bot_id,
                timestamp=datetime.now(timezone.utc),
                action=Action.BUY,
                size=self.config["level_size"],
                confidence=min(1.0, 1 - rsi / 100),
                reason=f"Price {price:.0f} touches BB lower. RSI: {rsi:.1f}",
            )

        if price >= bb_upper.iloc[-1] and self._in_position and rsi > self.config["rsi_filter_low"]:
            self._in_position = False
            return Signal(
                bot_id=self.bot_id,
                timestamp=datetime.now(timezone.utc),
                action=Action.SELL,
                size=1.0,
                confidence=min(1.0, rsi / 100),
                reason=f"Price {price:.0f} touches BB upper. RSI: {rsi:.1f}",
            )

        return Signal(
            bot_id=self.bot_id,
            timestamp=datetime.now(timezone.utc),
            action=Action.HOLD,
            size=0.0,
            confidence=1.0,
            reason=f"Price {price:.0f} within bands",
        )

    def on_start(self) -> None:
        self._in_position = False
        logger.info("GridBot initialized")