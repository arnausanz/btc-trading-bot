# core/engine/runner.py
import logging
import yaml
import pandas as pd
from datetime import datetime, timezone
from core.interfaces.base_bot import BaseBot
from core.interfaces.base_exchange import BaseExchange
from exchanges.paper import PaperExchange
from data.processing.technical import compute_features

logger = logging.getLogger(__name__)


class Runner:
    """
    Connecta un bot amb un exchange i executa el loop de trading.
    Donada una config, carrega el bot, construeix les observacions
    i processa cada candle cronològicament.
    """

    def __init__(self, bot: BaseBot, exchange: BaseExchange):
        self.bot = bot
        self.exchange = exchange

    def run(self, symbol: str, timeframe: str) -> list[dict]:
        """
        Executa el bot sobre totes les candles disponibles a la DB.
        Retorna l'historial de decisions i l'estat del portfolio.
        """
        schema = self.bot.observation_schema()
        df = compute_features(symbol=symbol, timeframe=timeframe)

        if len(df) < schema.lookback:
            raise ValueError(
                f"No hi ha prou dades. Necessites {schema.lookback} candles, "
                f"tens {len(df)}."
            )

        # Filtrem només les features que el bot necessita
        missing = [f for f in schema.features if f not in df.columns]
        if missing:
            raise ValueError(f"Features no trobades al DataFrame: {missing}")

        self.bot.on_start()
        history = []

        for i in range(schema.lookback, len(df)):
            window = df.iloc[i - schema.lookback:i]
            current_price = float(df.iloc[i]["close"])

            # Actualitza el preu a l'exchange
            if isinstance(self.exchange, PaperExchange):
                self.exchange.set_current_price(current_price)

            # Construeix l'observació per al bot
            observation = {
                "features": window[schema.features],
                "current_price": current_price,
                "timestamp": df.index[i],
                "portfolio": self.exchange.get_portfolio(),
            }

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