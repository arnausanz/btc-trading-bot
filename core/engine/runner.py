# core/engine/runner.py
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from core.interfaces.base_bot import BaseBot
from core.interfaces.base_exchange import BaseExchange
from data.observation.builder import ObservationBuilder

logger = logging.getLogger(__name__)


def _to_utc_timestamp(date_str: str) -> pd.Timestamp:
    """Converteix un string ISO (p.ex. '2025-01-01') a Timestamp UTC."""
    ts = pd.Timestamp(date_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


class Runner:
    def __init__(self, bot: BaseBot, exchange: BaseExchange):
        self.bot = bot
        self.exchange = exchange
        self.builder = ObservationBuilder()

    def run(
        self,
        symbol: str,
        timeframe: str,
        start_date: str | None = None,
        end_date: str | None = None,
        desc: str | None = None,
    ) -> list[dict]:
        """
        Executa el bot tick a tick sobre les dades del symbol/timeframe.

        start_date / end_date (strings ISO opcionals) permeten limitar l'interval
        d'iteració per al walk-forward backtesting.
        desc: etiqueta per a la barra de progrés (p.ex. 'train' o 'test').
        """
        schema = self.bot.observation_schema()
        self.builder.load(schema=schema, symbol=symbol)
        df = self.builder.get_dataframe(symbol=symbol, timeframe=timeframe)

        # ─── Determina els índexs d'iteració ──────────────────────────────────
        iter_start = schema.lookback
        iter_end = len(df)

        if start_date is not None:
            ts = _to_utc_timestamp(start_date)
            positions = np.where(df.index >= ts)[0]
            if len(positions) > 0:
                first_pos = int(positions[0])
                iter_start = max(schema.lookback, first_pos)
            else:
                logger.warning(f"start_date={start_date} és posterior a totes les dades. Retornem buit.")
                return []

        if end_date is not None:
            ts = _to_utc_timestamp(end_date)
            positions = np.where(df.index <= ts)[0]
            if len(positions) > 0:
                last_pos = int(positions[-1])
                iter_end = min(len(df), last_pos + 1)
            else:
                logger.warning(f"end_date={end_date} és anterior a totes les dades. Retornem buit.")
                return []

        if iter_start >= iter_end:
            logger.warning(f"Interval buit: iter_start={iter_start} >= iter_end={iter_end}. Retornem buit.")
            return []

        # ─── Bucle principal amb barra de progrés ─────────────────────────────
        bar_desc = f"  Backtest [{desc}]" if desc else "  Backtest"
        self.bot.on_start()
        history = []

        with tqdm(
            total=iter_end - iter_start,
            desc=bar_desc,
            unit="tick",
            leave=False,
            dynamic_ncols=True,
        ) as pbar:
            for i in range(iter_start, iter_end):
                current_price = float(df.iloc[i]["close"])
                self.exchange.set_current_price(current_price)

                observation = self.builder.build(schema=schema, symbol=symbol, index=i)
                observation["portfolio"] = self.exchange.get_portfolio()

                signal = self.bot.on_observation(observation)
                order = self.exchange.send_order(signal)

                history.append({
                    "timestamp": df.index[i],
                    "price": current_price,
                    "signal": signal.action,
                    "order_status": order.status,
                    "portfolio_value": self.exchange.get_portfolio_value(),
                })
                pbar.update(1)

        self.bot.on_stop()
        return history
