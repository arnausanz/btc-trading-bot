# data/processing/dataset.py
import logging
import pandas as pd
from data.processing.technical import compute_features
from core.config import TRAIN_UNTIL

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Construeix datasets supervisats per entrenar models ML.
    Suporta múltiples timeframes via merge_asof.

    WALK-FORWARD: el paràmetre train_until limita les dades d'entrenament
    per evitar que el model vegi dades del període de test (lookahead bias).
    """

    def __init__(
        self,
        symbol: str,
        timeframes: list[str],
        forward_window: int,
        threshold_pct: float,
        train_until: str = TRAIN_UNTIL,
    ):
        self.symbol = symbol
        self.timeframes = timeframes
        self.forward_window = forward_window
        self.threshold_pct = threshold_pct
        self.train_until = train_until

    @classmethod
    def from_config(cls, config: dict) -> "DatasetBuilder":
        return cls(
            symbol=config["data"]["symbol"],
            timeframes=config["data"]["timeframes"],
            forward_window=config["data"]["forward_window"],
            threshold_pct=config["data"]["threshold_pct"],
            # Permet sobreescriure train_until per config; sinó usa el global
            train_until=config["data"].get("train_until", TRAIN_UNTIL),
        )

    def build(self) -> tuple[pd.DataFrame, pd.Series]:
        primary_tf = self.timeframes[0]
        primary_df = compute_features(symbol=self.symbol, timeframe=primary_tf)

        # ─── Filtra al període de train ABANS de calcular labels ──────────────
        # Es fa DESPRÉS de compute_features perquè els indicadors (EMA, RSI, etc.)
        # necessiten totes les dades anteriors per al càlcul correcte (warm-up).
        if self.train_until:
            cutoff = pd.Timestamp(self.train_until)
            if cutoff.tzinfo is None:
                cutoff = cutoff.tz_localize("UTC")
            primary_df = primary_df[primary_df.index <= cutoff]
            logger.info(f"  Filtrant per train_until={self.train_until}: {len(primary_df)} files de train")

        future_return = primary_df["close"].shift(-self.forward_window) / primary_df["close"] - 1
        y = (future_return > self.threshold_pct).astype(int)
        X = primary_df.copy()

        for tf in self.timeframes[1:]:
            tf_df = compute_features(symbol=self.symbol, timeframe=tf)
            # Filtra el timeframe secundari al mateix període
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
