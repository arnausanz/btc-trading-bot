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
    Calcula indicadors tècnics sobre un DataFrame de candles.
    Tots els indicadors retornen el DataFrame amb columnes noves afegides.
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
        """MACD, Signal line i Histogram."""
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
        """Average True Range — mesura la volatilitat."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f"atr_{period}"] = true_range.ewm(span=period, adjust=False).mean()
        return df


def load_candles(symbol: str, timeframe: str, session: Session) -> pd.DataFrame:
    """Carrega candles de la DB i les retorna com a DataFrame ordenat per timestamp."""
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

    return pd.DataFrame(data).set_index("timestamp")


def compute_features(symbol: str, timeframe: str, ema_periods: list[int] | None = None) -> pd.DataFrame:
    """
    Carrega candles i calcula tots els indicadors tècnics.
    ema_periods permet especificar períodes addicionals d'EMA.
    """
    default_ema_periods = [9, 20, 50, 200]
    all_ema_periods = list(set(default_ema_periods + (ema_periods or [])))

    session = SessionLocal()
    try:
        df = load_candles(symbol=symbol, timeframe=timeframe, session=session)
        logger.info(f"Carregades {len(df)} candles de {symbol} {timeframe}")

        ti = TechnicalIndicators()
        for period in all_ema_periods:
            df = ti.ema(df, period)
        df = ti.rsi(df, 14)
        df = ti.macd(df)
        df = ti.bollinger_bands(df, 20)
        df = ti.atr(df, 14)

        df = df.dropna()
        logger.info(f"Features calculades: {len(df)} files, {len(df.columns)} columnes")
        return df
    finally:
        session.close()