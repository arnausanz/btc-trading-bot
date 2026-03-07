# bots/classical/dca_bot.py
import yaml
import logging
from datetime import datetime, timezone
from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.models import Signal, Action

logger = logging.getLogger(__name__)


class DCABot(BaseBot):
    """
    Dollar Cost Averaging Bot.
    Buys a fixed fraction of capital every N ticks, ignoring price.
    Optimal strategy for markets with bullish long-term trend.
    """

    def __init__(self, config_path: str = "config/bots/dca.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        super().__init__(bot_id=config["bot_id"], config=config)
        self._tick_count = 0

    def observation_schema(self) -> ObservationSchema:
        return ObservationSchema(
            features=self.config["features"],
            timeframes=[self.config["timeframe"]],
            lookback=self.config["lookback"],
        )

    def on_observation(self, observation: dict) -> Signal:
        self._tick_count += 1
        every_n = self.config["buy_every_n_ticks"]

        if self._tick_count % every_n == 0:
            action = Action.BUY
            size = self.config["buy_size"]
            reason = f"DCA tick {self._tick_count}: scheduled buy every {every_n} ticks"
        else:
            action = Action.HOLD
            size = 0.0
            reason = f"DCA tick {self._tick_count}: waiting"

        return Signal(
            bot_id=self.bot_id,
            timestamp=datetime.now(timezone.utc),
            action=action,
            size=size,
            confidence=1.0,
            reason=reason,
        )

    def on_start(self) -> None:
        self._tick_count = 0
        logger.info(f"DCABot initialized: buying every {self.config['buy_every_n_ticks']} ticks")