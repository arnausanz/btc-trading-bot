# bots/classical/grid_bot.py
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.models import Signal, Action

logger = logging.getLogger(__name__)


class GridBot(BaseBot):
    """
    Grid Trading Bot que calcula Bollinger Bands dinàmicament.
    """

    def __init__(self, config_path: str = "config/bots/grid.yaml"):
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

        # Calcula Bollinger Bands dinàmicament
        close = features["close"]
        middle = close.rolling(window=20).mean()
        std = close.rolling(window=20).std()
        bb_upper = middle + 2 * std
        bb_lower = middle - 2 * std

        price = close.iloc[-1]
        rsi = features["rsi_14"].iloc[-1]

        if (
            price <= bb_lower.iloc[-1]
            and not self._in_position
            and rsi < self.config["rsi_filter_high"]
        ):
            self._in_position = True
            return Signal(
                bot_id=self.bot_id,
                timestamp=datetime.now(timezone.utc),
                action=Action.BUY,
                size=self.config["level_size"],
                confidence=min(1.0, 1 - rsi / 100),
                reason=f"Preu {price:.0f} toca BB lower. RSI: {rsi:.1f}",
            )

        if (
            price >= bb_upper.iloc[-1]
            and self._in_position
            and rsi > self.config["rsi_filter_low"]
        ):
            self._in_position = False
            return Signal(
                bot_id=self.bot_id,
                timestamp=datetime.now(timezone.utc),
                action=Action.SELL,
                size=1.0,
                confidence=min(1.0, rsi / 100),
                reason=f"Preu {price:.0f} toca BB upper. RSI: {rsi:.1f}",
            )

        return Signal(
            bot_id=self.bot_id,
            timestamp=datetime.now(timezone.utc),
            action=Action.HOLD,
            size=0.0,
            confidence=1.0,
            reason=f"Preu {price:.0f} dins les bandes",
        )

    def on_start(self) -> None:
        self._in_position = False
        logger.info("GridBot iniciat")