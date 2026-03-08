# data/observation/builder.py
import logging
import pandas as pd
from core.interfaces.base_bot import ObservationSchema
from data.processing.feature_builder import FeatureBuilder

logger = logging.getLogger(__name__)


class ObservationBuilder:
    """
    Builds the observation that each bot needs from its ObservationSchema.
    The Runner doesn't know how observations are built — delegates to Builder.

    Uses FeatureBuilder to load technical + external features consistently
    with how data is built during training. External data config is read from
    schema.extras['external'] (set by the bot via its YAML config).

    This allows adding new data types (sentiment, onchain) without touching Runner.
    """

    def __init__(self):
        self._cache: dict[str, pd.DataFrame] = {}

    def load(self, schema: ObservationSchema, symbol: str) -> None:
        """
        Preloads all data necessary for the given schema.

        External features config is read from schema.extras['external'].
        EMA periods are auto-detected from schema.features names.
        """
        external = schema.extras.get("external", {})

        for timeframe in schema.timeframes:
            key = f"{symbol}_{timeframe}"
            if key not in self._cache:
                logger.info(f"Loading features for {symbol} {timeframe}...")

                # Auto-detect EMA periods from feature names (e.g. "ema_9" → 9)
                ema_periods = [
                    int(f.split("_")[1])
                    for f in schema.features
                    if f.startswith("ema_")
                ] or None

                fb = FeatureBuilder(
                    symbol=symbol,
                    timeframe=timeframe,
                    external=external,
                    select=None,  # Load all; build() will filter to schema.features
                    ema_periods=ema_periods,
                )
                self._cache[key] = fb.build()
                df = self._cache[key]
                logger.info(f"  {len(df)} rows, {len(df.columns)} columns loaded")

                # Validate that all required features exist in the loaded DataFrame
                missing = [f for f in schema.features if f not in df.columns]
                if missing:
                    raise ValueError(
                        f"[ObservationBuilder] Features required by bot not found "
                        f"in loaded data: {missing}\n"
                        f"Available columns: {sorted(df.columns.tolist())}\n"
                        f"Check that external config matches training config."
                    )

    def build(
        self,
        schema: ObservationSchema,
        symbol: str,
        index: int,
    ) -> dict:
        """
        Builds the observation for a specific point in time.
        index is the current position in the primary timeframe DataFrame.
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
                raise ValueError(f"Features not found in cache: {missing}")

            observation[timeframe] = {
                "features": window[schema.features].copy(),
                "current_price": float(df.iloc[index]["close"]),
                "timestamp": df.index[index],
            }

        return observation

    def get_dataframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Returns the complete cached DataFrame for a symbol and timeframe."""
        key = f"{symbol}_{timeframe}"
        if key not in self._cache:
            raise KeyError(
                f"Data not loaded for {symbol} {timeframe}. Call load() first."
            )
        return self._cache[key]
