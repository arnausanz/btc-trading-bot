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

        If schema.extras contains 'aux_timeframes', MultiFrameFeatureBuilder is used
        to merge auxiliary timeframes (e.g. 4h) into the base timeframe DataFrame.
        This matches the same builder used during training for multi-TF RL bots.
        """
        external        = schema.extras.get("external", {})
        aux_timeframes  = schema.extras.get("aux_timeframes", [])

        for timeframe in schema.timeframes:
            key = f"{symbol}_{timeframe}"
            if key not in self._cache:
                logger.info(f"Loading features for {symbol} {timeframe}...")

                if aux_timeframes:
                    # Multi-timeframe bot: use MultiFrameFeatureBuilder so columns
                    # like 'rsi_14_4h' are created by the same merging logic as
                    # during training. select=None → load all, validate below.
                    from data.processing.multiframe_builder import MultiFrameFeatureBuilder
                    fb = MultiFrameFeatureBuilder(
                        symbol=symbol,
                        base_timeframe=timeframe,
                        aux_timeframes=aux_timeframes,
                        external=external,
                        select=None,
                    )
                else:
                    # Single-timeframe bot: regular FeatureBuilder.
                    # Auto-detect EMA periods from base (non-suffixed) feature names.
                    ema_periods = [
                        int(f.split("_")[1])
                        for f in schema.features
                        if f.startswith("ema_") and "_" not in f[4:]
                    ] or None

                    fb = FeatureBuilder(
                        symbol=symbol,
                        timeframe=timeframe,
                        external=external,
                        select=None,  # Load all; validate below
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
        index is the current position in the PRIMARY timeframe DataFrame.

        Multi-timeframe support (backward-compatible):
        For each secondary timeframe, we find the correct row via searchsorted
        on the primary timestamp. For bots with a single timeframe, searchsorted
        returns exactly `index` — identical behaviour to before.
        """
        observation = {}

        # Resolve the timestamp from the primary (first) timeframe
        primary_tf  = schema.timeframes[0]
        primary_df  = self._cache[f"{symbol}_{primary_tf}"]

        if index < schema.lookback:
            raise ValueError(
                f"Index {index} too small for lookback {schema.lookback}"
            )

        # The timestamp of the current candle in the primary timeframe
        primary_ts = primary_df.index[index]

        for timeframe in schema.timeframes:
            key = f"{symbol}_{timeframe}"
            df  = self._cache[key]

            if timeframe == primary_tf:
                # Primary TF: use index directly (no searchsorted needed)
                tf_idx = index
            else:
                # Secondary TF: find the last candle whose timestamp <= primary_ts
                # searchsorted(..., side="right") - 1 gives the last idx <= ts
                tf_idx = int(df.index.searchsorted(primary_ts, side="right")) - 1
                if tf_idx < schema.lookback:
                    raise ValueError(
                        f"Lookback insuficient [{timeframe}]: "
                        f"tf_idx={tf_idx} < lookback={schema.lookback}. "
                        f"Carrega més historial."
                    )

            window  = df.iloc[tf_idx - schema.lookback : tf_idx]
            missing = [f for f in schema.features if f not in window.columns]
            if missing:
                raise ValueError(
                    f"Features not found in cache [{timeframe}]: {missing}"
                )

            observation[timeframe] = {
                "features":      window[schema.features].copy(),
                "current_price": float(df.iloc[tf_idx]["close"]),
                "timestamp":     df.index[tf_idx],
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
