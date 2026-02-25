# bots/classical/trend_bot.py
import yaml
import logging
import pandas as pd
from datetime import datetime, timezone
from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.models import Signal, Action

logger = logging.getLogger(__name__)


class TrendBot(BaseBot):
    """
    Trend Following Bot basat en EMA crossover + filtre RSI.
    Compra quan EMA ràpida creua EMA lenta cap amunt (i RSI no és sobrecomprat).
    Ven quan EMA ràpida creua EMA lenta cap avall (i RSI no és sobrevenut).
    """

    def __init__(self, config_path: str = "config/bots/trend.yaml"):
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

        ema_fast_col = f"ema_{self.config['ema_fast']}"
        ema_slow_col = f"ema_{self.config['ema_slow']}"

        # Últimes dues files per detectar el creuament
        prev = features.iloc[-2]
        curr = features.iloc[-1]

        ema_fast_crossed_up = (
            prev[ema_fast_col] <= prev[ema_slow_col] and
            curr[ema_fast_col] > curr[ema_slow_col]
        )
        ema_fast_crossed_down = (
            prev[ema_fast_col] >= prev[ema_slow_col] and
            curr[ema_fast_col] < curr[ema_slow_col]
        )

        rsi = curr["rsi_14"]
        size = self.config["trade_size"]

        if ema_fast_crossed_up and not self._in_position and rsi < self.config["rsi_overbought"]:
            self._in_position = True
            return Signal(
                bot_id=self.bot_id,
                timestamp=datetime.now(timezone.utc),
                action=Action.BUY,
                size=size,
                confidence=min(1.0, (rsi / 100)),
                reason=f"EMA crossover alcista. RSI: {rsi:.1f}",
            )

        if ema_fast_crossed_down and self._in_position and rsi > self.config["rsi_oversold"]:
            self._in_position = False
            return Signal(
                bot_id=self.bot_id,
                timestamp=datetime.now(timezone.utc),
                action=Action.SELL,
                size=1.0,  # ven tot
                confidence=min(1.0, (1 - rsi / 100)),
                reason=f"EMA crossover baixista. RSI: {rsi:.1f}",
            )

        return Signal(
            bot_id=self.bot_id,
            timestamp=datetime.now(timezone.utc),
            action=Action.HOLD,
            size=0.0,
            confidence=1.0,
            reason="Sense senyal",
        )

    def on_start(self) -> None:
        self._in_position = False
        logger.info("TrendBot iniciat")