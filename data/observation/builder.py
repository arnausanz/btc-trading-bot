# data/observation/builder.py
import logging
import pandas as pd
from core.interfaces.base_bot import ObservationSchema
from data.processing.technical import compute_features

logger = logging.getLogger(__name__)


class ObservationBuilder:
    """
    Construeix l'observació que cada bot necessita a partir del seu ObservationSchema.
    El Runner no sap com es construeix l'observació — delega al Builder.
    Això permet afegir nous tipus de dades (sentiment, onchain) sense tocar el Runner.
    """

    def __init__(self):
        self._cache: dict[str, pd.DataFrame] = {}

    def load(self, schema: ObservationSchema, symbol: str) -> None:
        """
        Precarrega totes les dades necessàries per al schema donat.
        Detecta automàticament quins períodes d'EMA necessita el bot.
        """
        for timeframe in schema.timeframes:
            key = f"{symbol}_{timeframe}"
            if key not in self._cache:
                logger.info(f"Carregant features per {symbol} {timeframe}...")

                # Detecta períodes d'EMA a partir dels noms de features
                ema_periods = [
                    int(f.split("_")[1])
                    for f in schema.features
                    if f.startswith("ema_")
                ]

                self._cache[key] = compute_features(
                    symbol=symbol,
                    timeframe=timeframe,
                    ema_periods=ema_periods,
                )
                logger.info(f"  {len(self._cache[key])} files carregades")

    def build(
        self,
        schema: ObservationSchema,
        symbol: str,
        index: int,
    ) -> dict:
        """
        Construeix l'observació per a un instant de temps concret.
        index és la posició actual al DataFrame principal (timeframe[0]).
        """
        observation = {}

        for timeframe in schema.timeframes:
            key = f"{symbol}_{timeframe}"
            df = self._cache[key]

            if index < schema.lookback:
                raise ValueError(
                    f"Index {index} massa petit per a lookback {schema.lookback}"
                )

            window = df.iloc[index - schema.lookback:index]
            missing = [f for f in schema.features if f not in window.columns]
            if missing:
                raise ValueError(f"Features no trobades: {missing}")

            observation[timeframe] = {
                "features": window[schema.features].copy(),
                "current_price": float(df.iloc[index]["close"]),
                "timestamp": df.index[index],
            }

        return observation

    def get_dataframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Retorna el DataFrame complet d'un symbol i timeframe."""
        key = f"{symbol}_{timeframe}"
        if key not in self._cache:
            raise KeyError(f"Dades no carregades per {symbol} {timeframe}. Crida load() primer.")
        return self._cache[key]