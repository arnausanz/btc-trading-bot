# data/processing/dataset.py
import logging
import pandas as pd
from data.processing.technical import compute_features

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Construeix datasets supervisats per entrenar models ML.
    Suporta múltiples timeframes via merge_asof.
    """

    def __init__(self, symbol: str, timeframes: list[str], forward_window: int, threshold_pct: float):
        self.symbol = symbol
        self.timeframes = timeframes
        self.forward_window = forward_window
        self.threshold_pct = threshold_pct

    @classmethod
    def from_config(cls, config: dict) -> "DatasetBuilder":
        return cls(
            symbol=config["data"]["symbol"],
            timeframes=config["data"]["timeframes"],
            forward_window=config["data"]["forward_window"],
            threshold_pct=config["data"]["threshold_pct"],
        )

    def build(self) -> tuple[pd.DataFrame, pd.Series]:
        primary_tf = self.timeframes[0]
        primary_df = compute_features(symbol=self.symbol, timeframe=primary_tf)

        future_return = primary_df["close"].shift(-self.forward_window) / primary_df["close"] - 1
        y = (future_return > self.threshold_pct).astype(int)
        X = primary_df.copy()

        for tf in self.timeframes[1:]:
            tf_df = compute_features(symbol=self.symbol, timeframe=tf).add_suffix(f"_{tf}")
            X = pd.merge_asof(X.sort_index(), tf_df.sort_index(), left_index=True, right_index=True, direction="backward")
            logger.info(f"  Afegides features de {tf}: {len(tf_df.columns)} columnes")

        X = X.iloc[:-self.forward_window]
        y = y.iloc[:-self.forward_window]

        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]

        logger.info(
            f"Dataset construït: {len(X)} files, {len(X.columns)} features, "
            f"timeframes: {self.timeframes}, target positiu: {y.mean():.1%}"
        )
        return X, y