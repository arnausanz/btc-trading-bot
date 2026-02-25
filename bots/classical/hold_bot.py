# bots/classical/hold_bot.py
import yaml
from datetime import datetime, timezone
from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.models import Signal, Action


class HoldBot(BaseBot):
    """
    Bot trivial que sempre fa HOLD.
    Útil per validar el Runner i com a baseline — qualsevol estratègia
    ha de batre aquest bot per tenir valor.
    """

    def __init__(self, config_path: str = "config/bots/hold.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        super().__init__(bot_id=config["bot_id"], config=config)

    def observation_schema(self) -> ObservationSchema:
        return ObservationSchema(
            features=self.config["features"],
            timeframes=[self.config["timeframe"]],
            lookback=self.config["lookback"],
        )

    def on_observation(self, observation: dict) -> Signal:
        return Signal(
            bot_id=self.bot_id,
            timestamp=datetime.now(timezone.utc),
            action=Action.HOLD,
            size=0.0,
            confidence=1.0,
            reason="sempre HOLD",
        )