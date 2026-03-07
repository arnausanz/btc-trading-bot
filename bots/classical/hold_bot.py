# bots/classical/hold_bot.py
import yaml
from datetime import datetime, timezone
from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.models import Signal, Action


class HoldBot(BaseBot):
    """
    Buy & Hold: buys BTC at the first tick and never touches it again.
    Is the minimum benchmark — any active strategy must outperform it.

    Logic:
      - First tick: BUY with all capital (size=1.0)
      - Remaining ticks: HOLD indefinitely

    If restored from DB and already has BTC (btc_balance > 0), does not buy again.
    """

    def __init__(self, config_path: str = "config/bots/hold.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        super().__init__(bot_id=config["bot_id"], config=config)
        self._bought: bool = False

    def on_start(self) -> None:
        self._bought = False

    def observation_schema(self) -> ObservationSchema:
        return ObservationSchema(
            features=self.config["features"],
            timeframes=[self.config["timeframe"]],
            lookback=self.config["lookback"],
        )

    def on_observation(self, observation: dict) -> Signal:
        # Check if we already have BTC (session restoration)
        portfolio = observation.get("portfolio", {})
        if not self._bought and portfolio.get("btc_balance", 0) > 1e-6:
            self._bought = True  # already have BTC from previous session

        if not self._bought:
            self._bought = True
            return Signal(
                bot_id=self.bot_id,
                timestamp=datetime.now(timezone.utc),
                action=Action.BUY,
                size=1.0,          # buy with all capital
                confidence=1.0,
                reason="Buy & Hold: initial buy",
            )

        return Signal(
            bot_id=self.bot_id,
            timestamp=datetime.now(timezone.utc),
            action=Action.HOLD,
            size=0.0,
            confidence=1.0,
            reason="Buy & Hold: holding position",
        )
