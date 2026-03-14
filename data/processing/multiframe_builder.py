# data/processing/multiframe_builder.py
"""
MultiFrameFeatureBuilder: extends FeatureBuilder to merge multiple timeframes
into a single model-ready DataFrame aligned at the base (primary) timeframe.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

How it works:
  1. Builds technical features for the BASE timeframe (e.g. 1H).
  2. For each AUXILIARY timeframe (e.g. 4H):
       a. Builds technical features for that timeframe.
       b. Renames all columns with a suffix (e.g. rsi_14 → rsi_14_4h).
       c. Merges them into the 1H DataFrame via merge_asof(backward) —
          i.e. forward-fill: each 1H row inherits the last available 4H value.
  3. Merges external sources (fear_greed, funding_rate) into the base tf only.
  4. Applies column selection and drops NaN rows.

Why multi-timeframe helps:
  Market regimes are most visible at longer timeframes:
    - 4H RSI > 70 confirms overbought state that 1H oscillations might miss.
    - Daily trend (4H EMA slope) provides context for 1H entry signals.
  Combining them lets the agent see both the immediate signal (1H) and
  the structural context (4H) in a single observation vector.

YAML schema:
  model_type: td3_multiframe
  symbol: BTC/USDT
  timeframe: 1h          # BASE timeframe for environment steps
  aux_timeframes: [4h]   # auxiliary timeframes to merge in

  features:
    lookback: 60
    select:
      # 1H features (primary signal)
      - close
      - rsi_14
      ...
      # 4H features (trend context) — NOTE: suffix matches aux_timeframes entry
      - rsi_14_4h
      - ema_20_4h
      - adx_14_4h
    external:
      fear_greed: true
      funding_rate: true

Feature naming convention:
  Base (1H) features retain their original names: close, rsi_14, ema_20, ...
  Auxiliary features are suffixed: rsi_14_4h, ema_20_4h, adx_14_4h, ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import logging
import pandas as pd
from data.processing.feature_builder import FeatureBuilder, _merge_asof
from data.processing.technical import compute_features

logger = logging.getLogger(__name__)


class MultiFrameFeatureBuilder:
    """
    Merges features from multiple timeframes into a single DataFrame at
    the base timeframe's resolution.

    This class follows the same interface contract as FeatureBuilder:
      builder = MultiFrameFeatureBuilder.from_config(config)
      df = builder.build()

    The resulting DataFrame is ready to feed directly into
    BtcTradingEnvProfessionalContinuous (or any _BaseTradingEnv subclass).
    """

    def __init__(
        self,
        symbol: str,
        base_timeframe: str,
        aux_timeframes: list[str],
        external: dict | None = None,
        select: list[str] | None = None,
    ):
        """
        Args:
            symbol:           Trading pair (e.g. "BTC/USDT")
            base_timeframe:   Primary resolution (e.g. "1h") — environment steps
                              are aligned to this timeframe.
            aux_timeframes:   Additional resolutions to merge in (e.g. ["4h"]).
                              All their features will be available with a suffix.
            external:         External data config (same as FeatureBuilder).
                              Applied only to the base timeframe.
            select:           Explicit column list to keep. None = keep all.
                              For RL this MUST match features.select in the YAML.
        """
        self.symbol = symbol
        self.base_timeframe = base_timeframe
        self.aux_timeframes = aux_timeframes
        self.external = external or {}
        self.select = select

    @classmethod
    def from_config(cls, config: dict) -> "MultiFrameFeatureBuilder":
        """
        Factory from the top-level unified YAML config dict.

        Reads: symbol, timeframe (base), aux_timeframes, features.select,
        features.external.
        """
        features_cfg = config.get("features", {})
        return cls(
            symbol=config["symbol"],
            base_timeframe=config["timeframe"],
            aux_timeframes=config.get("aux_timeframes", []),
            external=features_cfg.get("external", {}),
            select=features_cfg.get("select"),
        )

    def build(self) -> pd.DataFrame:
        """
        Builds the multi-timeframe feature DataFrame.

        Returns:
            UTC-indexed DataFrame with base + auxiliary features merged.
            Only columns in `select` are kept (if configured).
            All NaN rows are dropped.
        """
        # ── 1. Base timeframe features ────────────────────────────────────────
        # Detect EMA periods from base (non-suffixed) feature names
        base_ema_periods = None
        if self.select:
            base_names = [
                f for f in self.select
                if not any(f.endswith(f"_{tf}") for tf in self.aux_timeframes)
            ]
            detected = [
                int(f.split("_")[1]) for f in base_names
                if f.startswith("ema_") and f.split("_")[1].isdigit()
            ]
            base_ema_periods = detected if detected else None

        base_fb = FeatureBuilder(
            symbol=self.symbol,
            timeframe=self.base_timeframe,
            external=self.external,
            select=None,  # don't filter yet — merge first
            ema_periods=base_ema_periods,
        )
        df_base = base_fb.build()
        logger.info(
            f"MultiFrameFeatureBuilder: base {self.base_timeframe} — "
            f"{len(df_base)} rows × {len(df_base.columns)} cols"
        )

        # ── 2. Auxiliary timeframes ────────────────────────────────────────────
        for aux_tf in self.aux_timeframes:
            df_aux = self._build_aux(aux_tf)
            # Rename: rsi_14 → rsi_14_4h
            suffix = f"_{aux_tf}"
            df_aux = df_aux.add_suffix(suffix)

            # Merge asof (forward-fill into base)
            df_base = _merge_asof(df_base, df_aux, label=aux_tf)
            logger.info(
                f"  After merging {aux_tf}: "
                f"{len(df_base)} rows × {len(df_base.columns)} cols"
            )

        # ── 3. Column selection ────────────────────────────────────────────────
        if self.select:
            missing = [c for c in self.select if c not in df_base.columns]
            if missing:
                available = sorted(df_base.columns.tolist())
                raise ValueError(
                    f"[MultiFrameFeatureBuilder] Selected features not available: "
                    f"{missing}\nAvailable ({len(available)}): {available}"
                )
            df_base = df_base[self.select].copy()

        # ── 4. Final NaN drop ──────────────────────────────────────────────────
        before = len(df_base)
        df_base = df_base.dropna()
        dropped = before - len(df_base)
        if dropped:
            logger.info(f"  Dropped {dropped} NaN rows after multi-frame merge")

        logger.info(
            f"MultiFrameFeatureBuilder: final DataFrame "
            f"{len(df_base)} rows × {len(df_base.columns)} features "
            f"[{self.symbol} {self.base_timeframe}+{self.aux_timeframes}]"
        )
        return df_base

    def _build_aux(self, aux_tf: str) -> pd.DataFrame:
        """
        Builds features for an auxiliary timeframe.

        Determines which columns to compute based on the select list:
        strips the suffix to find the originating feature names.
        """
        # Detect which features are needed for this auxiliary tf
        aux_select = None
        aux_ema_periods = None
        if self.select:
            suffix = f"_{aux_tf}"
            # e.g. select contains "rsi_14_4h" → need "rsi_14" from 4H builder
            aux_with_suffix = [f for f in self.select if f.endswith(suffix)]
            if aux_with_suffix:
                aux_select = [f[: -len(suffix)] for f in aux_with_suffix]
                detected = [
                    int(f.split("_")[1]) for f in aux_select
                    if f.startswith("ema_") and f.split("_")[1].isdigit()
                ]
                aux_ema_periods = detected if detected else None

        aux_fb = FeatureBuilder(
            symbol=self.symbol,
            timeframe=aux_tf,
            external={},  # external data merged at base tf only
            select=aux_select,
            ema_periods=aux_ema_periods,
        )
        df_aux = aux_fb.build()
        logger.info(
            f"  Built {aux_tf}: {len(df_aux)} rows × {len(df_aux.columns)} cols"
        )
        return df_aux
