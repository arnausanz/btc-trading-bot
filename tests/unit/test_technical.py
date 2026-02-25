# tests/unit/test_technical.py
import pandas as pd
import numpy as np
import pytest
from data.processing.technical import TechnicalIndicators


def make_df(n: int = 100) -> pd.DataFrame:
    """DataFrame de test amb dades sintètiques."""
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    return pd.DataFrame({"close": close, "high": high, "low": low})


def test_ema():
    df = TechnicalIndicators.ema(make_df(), period=20)
    assert "ema_20" in df.columns
    assert df["ema_20"].notna().any()


def test_rsi_range():
    df = TechnicalIndicators.rsi(make_df(), period=14)
    assert "rsi_14" in df.columns
    valid = df["rsi_14"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_macd():
    df = TechnicalIndicators.macd(make_df())
    assert all(col in df.columns for col in ["macd", "macd_signal", "macd_hist"])


def test_bollinger_bands():
    df = TechnicalIndicators.bollinger_bands(make_df(), period=20)
    assert all(col in df.columns for col in ["bb_upper_20", "bb_middle_20", "bb_lower_20"])
    valid = df.dropna()
    assert (valid["bb_upper_20"] >= valid["bb_middle_20"]).all()
    assert (valid["bb_middle_20"] >= valid["bb_lower_20"]).all()


def test_atr():
    df = TechnicalIndicators.atr(make_df(), period=14)
    assert "atr_14" in df.columns
    assert (df["atr_14"].dropna() >= 0).all()