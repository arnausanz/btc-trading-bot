# core/engine/runner.py
import logging
from core.interfaces.base_bot import BaseBot
from core.interfaces.base_exchange import BaseExchange
from exchanges.paper import PaperExchange
from data.observation.builder import ObservationBuilder

logger = logging.getLogger(__name__)


class Runner:
    """
    Connecta un bot amb un exchange i executa el loop de trading.
    Delega la construcció d'observacions a l'ObservationBuilder.
    """

    def __init__(self, bot: BaseBot, exchange: BaseExchange):
        self.bot = bot
        self.exchange = exchange
        self.builder = ObservationBuilder()

    def run(self, symbol: str, timeframe: str) -> list[dict]:
        schema = self.bot.observation_schema()

        # Precarrega totes les dades necessàries
        self.builder.load(schema=schema, symbol=symbol)
        df = self.builder.get_dataframe(symbol=symbol, timeframe=timeframe)

        if len(df) < schema.lookback:
            raise ValueError(
                f"No hi ha prou dades. Necessites {schema.lookback} candles, "
                f"tens {len(df)}."
            )

        self.bot.on_start()
        history = []

        for i in range(schema.lookback, len(df)):
            current_price = float(df.iloc[i]["close"])

            if isinstance(self.exchange, PaperExchange):
                self.exchange.set_current_price(current_price)

            observation = self.builder.build(
                schema=schema,
                symbol=symbol,
                index=i,
            )
            observation["portfolio"] = self.exchange.get_portfolio()

            signal = self.bot.on_observation(observation)
            order = self.exchange.send_order(signal)

            history.append({
                "timestamp": df.index[i],
                "price": current_price,
                "signal": signal.action,
                "order_status": order.status,
                "portfolio_value": (
                    self.exchange.get_portfolio_value()
                    if isinstance(self.exchange, PaperExchange)
                    else None
                ),
            })

        self.bot.on_stop()
        logger.info(f"Runner completat: {len(history)} ticks processats")
        return history