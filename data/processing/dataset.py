# data/processing/dataset.py
import logging
import numpy as np
import pandas as pd
from data.processing.technical import compute_features

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Construeix datasets supervisats per entrenar models ML.
    Suporta múltiples timeframes — les features de cada timeframe
    s'afegeixen com a columnes addicionals alineades pel timestamp.
    """

    def __init__(
        self,
        symbol: str,
        timeframes: list[str],
        forward_window: int,
        threshold_pct: float,
    ):
        self.symbol = symbol
        self.timeframes = timeframes
        self.forward_window = forward_window
        self.threshold_pct = threshold_pct

    @classmethod
    def from_config(cls, config: dict) -> "DatasetBuilder":
        """Construeix el DatasetBuilder des d'un diccionari de config."""
        return cls(
            symbol=config["data"]["symbol"],
            timeframes=config["data"]["timeframes"],
            forward_window=config["data"]["forward_window"],
            threshold_pct=config["data"]["threshold_pct"],
        )

    def build(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Retorna (X, y) on:
        X = features tècniques de tots els timeframes
        y = 1 si el preu puja >threshold en forward_window candles, 0 si baixa
        El timeframe principal (primer de la llista) defineix la granularitat del dataset.
        """
        primary_tf = self.timeframes[0]
        primary_df = compute_features(symbol=self.symbol, timeframe=primary_tf)

        # Target basat sempre en el timeframe principal
        future_return = primary_df["close"].shift(-self.forward_window) / primary_df["close"] - 1
        y = (future_return > self.threshold_pct).astype(int)

        # Features del timeframe principal
        feature_cols = [c for c in primary_df.columns]
        X = primary_df[feature_cols].copy()

        # Afegeix features de timeframes secundaris via resample i merge
        for tf in self.timeframes[1:]:
            tf_df = compute_features(symbol=self.symbol, timeframe=tf)
            tf_df = tf_df.add_suffix(f"_{tf}")
            # Merge asof: per cada candle principal, agafa la feature secundària més recent
            X = pd.merge_asof(
                X.sort_index(),
                tf_df.sort_index(),
                left_index=True,
                right_index=True,
                direction="backward",
            )
            logger.info(f"  Afegides features de {tf}: {len(tf_df.columns)} columnes")

        # Elimina les últimes forward_window files (no tenen target)
        X = X.iloc[:-self.forward_window]
        y = y.iloc[:-self.forward_window]

        # Elimina NaNs
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]

        logger.info(
            f"Dataset construït: {len(X)} files, {len(X.columns)} features, "
            f"timeframes: {self.timeframes}, "
            f"target positiu: {y.mean():.1%}"
        )
        return X, y