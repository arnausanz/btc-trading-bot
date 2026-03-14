# data/processing/technical.py
import logging
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from core.db.session import SessionLocal
from core.db.models import CandleDB

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Computes technical indicators on a DataFrame of candles.
    All indicators return the DataFrame with new columns added.
    """

    @staticmethod
    def ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.DataFrame:
        """Exponential Moving Average."""
        df[f"ema_{period}"] = df[column].ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.DataFrame:
        """Relative Strength Index (0-100)."""
        delta = df[column].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: str = "close"
    ) -> pd.DataFrame:
        """MACD, Signal line and Histogram."""
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    @staticmethod
    def bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        column: str = "close"
    ) -> pd.DataFrame:
        """Bollinger Bands: upper, middle (SMA), lower."""
        df[f"bb_middle_{period}"] = df[column].rolling(window=period).mean()
        std = df[column].rolling(window=period).std()
        df[f"bb_upper_{period}"] = df[f"bb_middle_{period}"] + std_dev * std
        df[f"bb_lower_{period}"] = df[f"bb_middle_{period}"] - std_dev * std
        return df

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average True Range — measures volatility."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f"atr_{period}"] = true_range.ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Average Directional Index — measures trend strength (not direction).
        > 25: trend present.  < 20: ranging market.
        Uses Wilder-style EWM smoothing (same as ATR).
        """
        high = df["high"]
        low  = df["low"]

        # Directional movement
        up_move   = high.diff()
        down_move = -low.diff()

        plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # True Range (same formula as atr())
        high_low   = high - low
        high_close = (high - df["close"].shift()).abs()
        low_close  = (low  - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        tr_smooth       = true_range.ewm(span=period, adjust=False).mean()
        plus_dm_smooth  = pd.Series(plus_dm,  index=df.index).ewm(span=period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean()

        plus_di  = 100 * plus_dm_smooth  / tr_smooth.replace(0, np.nan)
        minus_di = 100 * minus_dm_smooth / tr_smooth.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df[f"adx_{period}"] = dx.ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def volume_normalized(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Volume normalised by its rolling mean.
        Values > 1 = above-average volume (breakout confirmation).
        Values < 1 = below-average volume (weak move / compression).
        """
        rolling_mean = df["volume"].rolling(window=period).mean()
        df[f"volume_norm_{period}"] = df["volume"] / rolling_mean.replace(0, np.nan)
        return df


def load_candles(symbol: str, timeframe: str, session: Session) -> pd.DataFrame:
    """Loads candles from DB and returns them as DataFrame sorted by timestamp."""
    rows = (
        session.query(CandleDB)
        .filter_by(symbol=symbol, timeframe=timeframe)
        .order_by(CandleDB.timestamp)
        .all()
    )
    data = [{
        "timestamp": r.timestamp,
        "open": r.open,
        "high": r.high,
        "low": r.low,
        "close": r.close,
        "volume": r.volume,
    } for r in rows]

    if not data:
        raise ValueError(
            f"No candles found in DB for symbol='{symbol}', timeframe='{timeframe}'. "
            f"Run: python scripts/download_data.py"
        )
    return pd.DataFrame(data).set_index("timestamp")


def compute_features(symbol: str, timeframe: str, ema_periods: list[int] | None = None) -> pd.DataFrame:
    """
    Loads candles and computes all technical indicators.

    Always computes the full professional feature set:
        - EMA (9, 20, 50, 200 + any extra via ema_periods)
        - RSI-14, MACD, Bollinger Bands (20), ATR-14, ATR-5
        - ADX-14 (regime: trend strength)
        - vol_ratio = ATR-5 / ATR-14  (regime: volatility expansion)
        - volume_norm_20              (breakout confirmation)

    FeatureBuilder.select filters which columns are actually used by the model.
    """
    default_ema_periods = [9, 20, 50, 200]
    all_ema_periods = list(set(default_ema_periods + (ema_periods or [])))

    session = SessionLocal()
    try:
        df = load_candles(symbol=symbol, timeframe=timeframe, session=session)
        logger.info(f"Loaded {len(df)} candles from {symbol} {timeframe}")

        ti = TechnicalIndicators()

        # ── Core indicators (always computed) ───────────────────────────────
        for period in all_ema_periods:
            df = ti.ema(df, period)
        df = ti.rsi(df, 14)
        df = ti.macd(df)
        df = ti.bollinger_bands(df, 20)
        df = ti.atr(df, 14)

        # ── Professional indicators ──────────────────────────────────────────
        df = ti.atr(df, 5)                   # short ATR for vol_ratio
        df["vol_ratio"] = (
            df["atr_5"] / df["atr_14"].replace(0, np.nan)
        )                                    # > 1.2 expanding, < 0.8 compressing
        df = ti.adx(df, 14)                  # trend strength (0-100)
        df = ti.volume_normalized(df, 20)    # relative volume

        df = df.dropna()
        logger.info(f"Features computed: {len(df)} rows, {len(df.columns)} columns")
        return df
    finally:
        session.close()