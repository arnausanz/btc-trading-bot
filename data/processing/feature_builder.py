# data/processing/feature_builder.py
"""
FeatureBuilder: single class to build a complete feature DataFrame
from a YAML-driven config. Supports technical indicators + external data sources.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIG STRUCTURE (data.features section in training/bot YAML):

  features:
    external:                             # all optional, default: disabled
      fear_greed: true                    # adds: fear_greed_value, fear_greed_class
      funding_rate: true                  # adds: funding_rate
      funding_rate_symbol: BTC/USDT:USDT # optional, overrides default symbol
      open_interest:                      # list of {symbol, timeframe} pairs
        - symbol: BTC/USDT:USDT
          timeframe: 1h                   # adds: oi_btc_1h, oi_usdt_1h
        - symbol: BTC/USDT:USDT
          timeframe: 5m                   # adds: oi_btc_5m, oi_usdt_5m
      blockchain:                         # list of blockchain.com metric names
        - hash-rate                       # adds: hash_rate
        - n-unique-addresses              # adds: n_unique_addresses
        - transaction-fees               # adds: transaction_fees

    select:                               # columns to keep (null = keep all)
      - close                             # CRITICAL for RL: must match bot's features list
      - rsi_14                            # obs_shape = len(select) × lookback
      - macd
      ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SHAPE CONSISTENCY (RL):

  Training config  data.features.select  ←→  Bot config features list
  Training config  environment.lookback  ←→  Bot config lookback

  obs_shape = len(select) × lookback   (must be identical in both)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MERGING STRATEGY:

  All external sources are left-joined into the technical DataFrame
  via merge_asof(direction="backward") — i.e. forward-fill semantics.
  This handles multi-frequency data (daily fear & greed joined to hourly candles).
"""
import logging
import pandas as pd
from data.processing.technical import compute_features
from data.processing.external import (
    load_fear_greed,
    load_funding_rate,
    load_open_interest,
    load_blockchain_metric,
)

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Builds a complete, model-ready feature DataFrame from config.

    Usage:
        # From a training YAML's data section:
        fb = FeatureBuilder.from_config(config["data"])
        df = fb.build()

        # Directly:
        fb = FeatureBuilder(
            symbol="BTC/USDT",
            timeframe="1h",
            external={"fear_greed": True, "funding_rate": True},
            select=["close", "rsi_14", "fear_greed_value"],
        )
        df = fb.build()
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        external: dict | None = None,
        select: list[str] | None = None,
        ema_periods: list[int] | None = None,
    ):
        """
        Args:
            symbol:       Trading pair (e.g. "BTC/USDT")
            timeframe:    Candle granularity (e.g. "1h", "4h", "1d")
            external:     External data config dict (see module docstring)
            select:       List of column names to keep. None = keep all.
                          For RL training this MUST match the bot's features list.
            ema_periods:  Additional EMA periods to compute (auto-detected from
                          select if not provided)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.external = external or {}
        self.select = select
        self.ema_periods = ema_periods

    @classmethod
    def from_config(cls, data_config: dict) -> "FeatureBuilder":
        """
        Factory from the `data` section of a training YAML config.

        Handles both RL (timeframe: str) and ML (timeframes: list) configs.
        For multi-timeframe ML configs, uses timeframes[0] as the primary.

        Args:
            data_config: the `data` section of a training YAML
        """
        # Support both RL (timeframe) and ML (timeframes) configs
        timeframe = data_config.get("timeframe") or data_config["timeframes"][0]
        features_cfg = data_config.get("features", {})
        select = features_cfg.get("select")  # None if not configured

        # Auto-detect EMA periods from select list to avoid computing unused ones
        ema_periods = None
        if select:
            detected = [int(f.split("_")[1]) for f in select if f.startswith("ema_")]
            ema_periods = detected or None

        return cls(
            symbol=data_config["symbol"],
            timeframe=timeframe,
            external=features_cfg.get("external", {}),
            select=select,
            ema_periods=ema_periods,
        )

    def build(self) -> pd.DataFrame:
        """
        Builds the complete feature DataFrame.

        Steps:
          1. Load technical indicators via compute_features()
          2. Load and merge each configured external source (merge_asof backward)
          3. Filter to select columns if configured
          4. Drop any remaining NaN rows

        Returns:
            UTC-indexed DataFrame with requested features.
            Rows with any NaN are dropped.

        Raises:
            ValueError: if any column in `select` is not available after loading.
        """
        # 1. Technical indicators (OHLCV + EMA/RSI/MACD/BB/ATR)
        df = compute_features(
            symbol=self.symbol,
            timeframe=self.timeframe,
            ema_periods=self.ema_periods,
        )

        # 2. External sources (merged via merge_asof backward fill)
        if self.external:
            df = self._merge_external(df)

        # 3. Column selection
        if self.select:
            missing = [c for c in self.select if c not in df.columns]
            if missing:
                available = sorted(df.columns.tolist())
                raise ValueError(
                    f"[FeatureBuilder] Selected features not available: {missing}\n"
                    f"Available columns ({len(available)}): {available}"
                )
            df = df[self.select].copy()
            logger.info(f"  Column selection: {len(self.select)} features → {self.select}")

        # 4. Final NaN drop (edges from indicator warmup or sparse external data)
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        if dropped:
            logger.debug(f"  Dropped {dropped} NaN rows after merge")

        logger.info(
            f"FeatureBuilder: {len(df)} rows × {len(df.columns)} features "
            f"[{self.symbol} {self.timeframe}]"
        )
        return df

    def _merge_external(self, df: pd.DataFrame) -> pd.DataFrame:
        """Loads and merges all configured external data sources into df."""
        ext = self.external

        if ext.get("fear_greed"):
            fg = load_fear_greed()
            df = _merge_asof(df, fg, label="fear_greed")

        if ext.get("funding_rate"):
            fr_symbol = ext.get("funding_rate_symbol", "BTC/USDT:USDT")
            fr = load_funding_rate(symbol=fr_symbol)
            df = _merge_asof(df, fr, label="funding_rate")

        for oi_cfg in ext.get("open_interest", []):
            oi_symbol = oi_cfg.get("symbol", "BTC/USDT:USDT")
            oi_tf = oi_cfg.get("timeframe", "1h")
            oi = load_open_interest(symbol=oi_symbol, timeframe=oi_tf)
            df = _merge_asof(df, oi, label=f"open_interest_{oi_tf}")

        for metric in ext.get("blockchain", []):
            bm = load_blockchain_metric(metric=metric)
            df = _merge_asof(df, bm, label=metric)

        return df


def _merge_asof(base: pd.DataFrame, ext: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Left-joins ext into base using merge_asof (backward = forward-fill semantics).
    Normalizes both indexes to UTC before merging.
    """
    if ext.empty:
        logger.warning(f"  External source '{label}' is empty — skipping merge")
        return base

    if base.index.tz is None:
        base = base.tz_localize("UTC")
    if ext.index.tz is None:
        ext = ext.tz_localize("UTC")

    result = pd.merge_asof(
        base.sort_index(),
        ext.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    logger.info(f"  Merged '{label}': +{len(ext.columns)} column(s) → {list(ext.columns)}")
    return result
