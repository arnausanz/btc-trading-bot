# data/processing/dataset.py
import logging
import pandas as pd
from data.processing.feature_builder import FeatureBuilder
from data.processing.technical import compute_features
from core.config import TRAIN_UNTIL

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Builds supervised datasets for training ML models.
    Supports multiple timeframes via merge_asof and optional external features.

    WALK-FORWARD: train_until prevents the model from seeing test period data
    (lookahead bias). Features are computed on all historical data first
    (for indicator warm-up), then filtered to the training period.

    Config structure (data section of training YAML):

      data:
        symbol: BTC/USDT
        timeframes: [1h, 4h]     # first = primary, rest = secondary (technical only)
        forward_window: 12
        threshold_pct: 0.01
        train_until: "2024-12-31" # optional, overrides global

        features:                 # optional
          external:               # same as FeatureBuilder external config
            fear_greed: false
            funding_rate: false
            open_interest: []
            blockchain: []
          select:                 # optional: restrict which columns go into X
            - close               # null/omitted = all columns
            - rsi_14
            - fear_greed_value
    """

    def __init__(
        self,
        symbol: str,
        timeframes: list[str],
        forward_window: int,
        threshold_pct: float,
        train_until: str = TRAIN_UNTIL,
        features_cfg: dict | None = None,
    ):
        self.symbol = symbol
        self.timeframes = timeframes
        self.forward_window = forward_window
        self.threshold_pct = threshold_pct
        self.train_until = train_until
        self.features_cfg = features_cfg or {}

    @classmethod
    def from_config(cls, config: dict) -> "DatasetBuilder":
        return cls(
            symbol=config["data"]["symbol"],
            timeframes=config["data"]["timeframes"],
            forward_window=config["data"]["forward_window"],
            threshold_pct=config["data"]["threshold_pct"],
            train_until=config["data"].get("train_until", TRAIN_UNTIL),
            features_cfg=config["data"].get("features", {}),
        )

    def build(self) -> tuple[pd.DataFrame, pd.Series]:
        primary_tf = self.timeframes[0]

        # Primary timeframe via FeatureBuilder (handles external features)
        # select=None here: let ML models see all columns; optional select applied later
        fb = FeatureBuilder(
            symbol=self.symbol,
            timeframe=primary_tf,
            external=self.features_cfg.get("external", {}),
            select=None,
        )
        primary_df = fb.build()

        # Filter to training period AFTER computing features
        # (indicator warmup needs all prior data for correctness)
        if self.train_until:
            cutoff = pd.Timestamp(self.train_until)
            if cutoff.tzinfo is None:
                cutoff = cutoff.tz_localize("UTC")
            primary_df = primary_df[primary_df.index <= cutoff]
            logger.info(f"  Filtered to train_until={self.train_until}: {len(primary_df)} rows")

        # Labels: will price rise > threshold_pct in the next forward_window candles?
        future_return = primary_df["close"].shift(-self.forward_window) / primary_df["close"] - 1
        y = (future_return > self.threshold_pct).astype(int)
        X = primary_df.copy()

        # Secondary timeframes (technical indicators only, no external)
        for tf in self.timeframes[1:]:
            tf_df = compute_features(symbol=self.symbol, timeframe=tf)
            if self.train_until:
                cutoff = pd.Timestamp(self.train_until)
                if cutoff.tzinfo is None:
                    cutoff = cutoff.tz_localize("UTC")
                tf_df = tf_df[tf_df.index <= cutoff]
            tf_df = tf_df.add_suffix(f"_{tf}")
            X = pd.merge_asof(
                X.sort_index(), tf_df.sort_index(),
                left_index=True, right_index=True, direction="backward",
            )
            logger.info(f"  Added {tf} features: {len(tf_df.columns)} columns")

        # Remove the last forward_window rows (no valid labels there)
        X = X.iloc[:-self.forward_window]
        y = y.iloc[:-self.forward_window]

        # Optional column selection (e.g. feature subset for specific experiments)
        select = self.features_cfg.get("select")
        if select:
            missing = [c for c in select if c not in X.columns]
            if missing:
                raise ValueError(
                    f"[DatasetBuilder] Selected features not available: {missing}\n"
                    f"Available: {sorted(X.columns.tolist())}"
                )
            X = X[select]
            logger.info(f"  Column selection: {len(select)} features")

        # Drop rows with any NaN
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]

        logger.info(
            f"Dataset built: {len(X)} rows, {len(X.columns)} features, "
            f"timeframes: {self.timeframes}, positive target: {y.mean():.1%}"
        )
        return X, y
