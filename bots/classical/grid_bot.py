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
    Grid Trading Bot basat en Bollinger Bands.
    Compra quan el preu toca la banda inferior (sobrevenut)
    i ven quan toca la banda superior (sobrecomprat).
    Funciona millor en mercats laterals amb oscil·lacions regulars.
    """

    def __init__(self, config_path: str = "config/bots/grid.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        super().__init__(bot_id=config["bot_id"], config=config)
        self._in_position = False

    def observation_schema(self) -> ObservationSchema:
        return ObservationSchema(
            features=self.config["features"],
            timeframes=[self.config["timeframe"]],
            lookback=self.config["lookback"],
        )

    def on_observation(self, observation: dict) -> Signal:
        timeframe = self.config["timeframe"]
        features: pd.DataFrame = observation[timeframe]["features"]
        curr = features.iloc[-1]

        price = curr["close"]
        bb_lower = curr["bb_lower_20"]
        bb_upper = curr["bb_upper_20"]
        rsi = curr["rsi_14"]

        # Compra quan el preu toca o trenca la banda inferior
        if (
            price <= bb_lower
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
                reason=f"Preu {price:.0f} toca BB lower {bb_lower:.0f}. RSI: {rsi:.1f}",
            )

        # Ven quan el preu toca o trenca la banda superior
        if (
            price >= bb_upper
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
                reason=f"Preu {price:.0f} toca BB upper {bb_upper:.0f}. RSI: {rsi:.1f}",
            )

        return Signal(
            bot_id=self.bot_id,
            timestamp=datetime.now(timezone.utc),
            action=Action.HOLD,
            size=0.0,
            confidence=1.0,
            reason=f"Preu {price:.0f} dins les bandes [{bb_lower:.0f}, {bb_upper:.0f}]",
        )

    def on_start(self) -> None:
        self._in_position = False
        logger.info("GridBot iniciat")