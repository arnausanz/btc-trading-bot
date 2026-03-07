# tests/smoke/test_technical.py
"""Test technical indicator calculations with synthetic data."""
import pandas as pd
import numpy as np
import pytest
from data.processing.technical import TechnicalIndicators


def make_df(n: int = 100) -> pd.DataFrame:
    """Create a synthetic DataFrame with OHLCV data."""
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    return pd.DataFrame({"close": close, "high": high, "low": low})


def test_ema():
    df = TechnicalIndicators.ema(make_df(), period=20)
    assert "ema_20" in df.columns
    assert df["ema_20"].notna().any()


def test_ema_different_periods():
    df = make_df()
    df = TechnicalIndicators.ema(df, period=5)
    df = TechnicalIndicators.ema(df, period=20)
    assert "ema_5" in df.columns
    assert "ema_20" in df.columns


def test_rsi_range():
    """RSI values should be between 0 and 100."""
    df = TechnicalIndicators.rsi(make_df(), period=14)
    assert "rsi_14" in df.columns
    valid = df["rsi_14"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_different_periods():
    df = make_df()
    df = TechnicalIndicators.rsi(df, period=7)
    df = TechnicalIndicators.rsi(df, period=14)
    assert "rsi_7" in df.columns
    assert "rsi_14" in df.columns


def test_macd():
    df = TechnicalIndicators.macd(make_df())
    assert all(col in df.columns for col in ["macd", "macd_signal", "macd_hist"])


def test_bollinger_bands():
    df = TechnicalIndicators.bollinger_bands(make_df(), period=20)
    assert all(col in df.columns for col in ["bb_upper_20", "bb_middle_20", "bb_lower_20"])
    valid = df.dropna()
    assert (valid["bb_upper_20"] >= valid["bb_middle_20"]).all()
    assert (valid["bb_middle_20"] >= valid["bb_lower_20"]).all()


def test_bollinger_bands_different_std_dev():
    df = make_df()
    df1 = TechnicalIndicators.bollinger_bands(df.copy(), period=20, std_dev=1.0)
    df2 = TechnicalIndicators.bollinger_bands(df.copy(), period=20, std_dev=3.0)

    # Wider std_dev should produce wider bands
    valid = df1.dropna()
    width1 = (valid["bb_upper_20"] - valid["bb_lower_20"]).mean()

    valid = df2.dropna()
    width2 = (valid["bb_upper_20"] - valid["bb_lower_20"]).mean()

    assert width2 > width1


def test_atr():
    """ATR should be non-negative."""
    df = TechnicalIndicators.atr(make_df(), period=14)
    assert "atr_14" in df.columns
    assert (df["atr_14"].dropna() >= 0).all()


def test_atr_different_periods():
    df = make_df()
    df = TechnicalIndicators.atr(df, period=7)
    df = TechnicalIndicators.atr(df, period=14)
    assert "atr_7" in df.columns
    assert "atr_14" in df.columns


def test_chaining_indicators():
    """Test that multiple indicators can be chained together."""
    df = make_df()
    df = TechnicalIndicators.ema(df, period=20)
    df = TechnicalIndicators.rsi(df, period=14)
    df = TechnicalIndicators.macd(df)

    assert "ema_20" in df.columns
    assert "rsi_14" in df.columns
    assert "macd" in df.columns
