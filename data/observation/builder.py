# data/observation/builder.py
import logging
import pandas as pd
from core.interfaces.base_bot import ObservationSchema
from data.processing.technical import compute_features

logger = logging.getLogger(__name__)


class ObservationBuilder:
    """
    Builds the observation that each bot needs from its ObservationSchema.
    The Runner doesn't know how observation is built — delegates to Builder.
    This allows adding new data types (sentiment, onchain) without touching Runner.
    """

    def __init__(self):
        self._cache: dict[str, pd.DataFrame] = {}

    def load(self, schema: ObservationSchema, symbol: str) -> None:
        """
        Preloads all data necessary for the given schema.
        Automatically detects which EMA periods the bot needs.
        """
        for timeframe in schema.timeframes:
            key = f"{symbol}_{timeframe}"
            if key not in self._cache:
                logger.info(f"Loading features for {symbol} {timeframe}...")

                # Detects EMA periods from feature names
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
                logger.info(f"  {len(self._cache[key])} rows loaded")

    def build(
        self,
        schema: ObservationSchema,
        symbol: str,
        index: int,
    ) -> dict:
        """
        Builds the observation for a specific point in time.
        index is the current position in the main DataFrame (timeframe[0]).
        """
        observation = {}

        for timeframe in schema.timeframes:
            key = f"{symbol}_{timeframe}"
            df = self._cache[key]

            if index < schema.lookback:
                raise ValueError(
                    f"Index {index} too small for lookback {schema.lookback}"
                )

            window = df.iloc[index - schema.lookback:index]
            missing = [f for f in schema.features if f not in window.columns]
            if missing:
                raise ValueError(f"Features not found: {missing}")

            observation[timeframe] = {
                "features": window[schema.features].copy(),
                "current_price": float(df.iloc[index]["close"]),
                "timestamp": df.index[index],
            }

        return observation

    def get_dataframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Returns the complete DataFrame for a symbol and timeframe."""
        key = f"{symbol}_{timeframe}"
        if key not in self._cache:
            raise KeyError(f"Data not loaded for {symbol} {timeframe}. Call load() first.")
        return self._cache[key]