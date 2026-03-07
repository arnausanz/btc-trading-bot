# bots/classical/hold_bot.py
import yaml
from datetime import datetime, timezone
from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.models import Signal, Action


class HoldBot(BaseBot):
    """
    Buy & Hold: compra BTC al primer tick i no toca res mai més.
    És el benchmark mínim — qualsevol estratègia activa ha de superar-lo.

    Lògica:
      - Primer tick: BUY amb tot el capital (size=1.0)
      - Resta de ticks: HOLD perpetuu

    Si es restaura des de DB i ja té BTC (btc_balance > 0), no torna a comprar.
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
        # Comprova si ja tenim BTC (restauració de sessió anterior)
        portfolio = observation.get("portfolio", {})
        if not self._bought and portfolio.get("btc_balance", 0) > 1e-6:
            self._bought = True  # ja tenim BTC des d'una sessió anterior

        if not self._bought:
            self._bought = True
            return Signal(
                bot_id=self.bot_id,
                timestamp=datetime.now(timezone.utc),
                action=Action.BUY,
                size=1.0,          # compra amb tot el capital
                confidence=1.0,
                reason="Buy & Hold: compra inicial",
            )

        return Signal(
            bot_id=self.bot_id,
            timestamp=datetime.now(timezone.utc),
            action=Action.HOLD,
            size=0.0,
            confidence=1.0,
            reason="Buy & Hold: mantenint posició",
        )
