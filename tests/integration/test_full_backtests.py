# tests/integration/test_full_backtests.py
"""
Integration tests for full backtest runs.
Requires a real PostgreSQL database with OHLCV data.

Mark these tests with @pytest.mark.integration.

Run with:
  pytest tests/integration/ -m integration -v

To skip these tests, run smoke and unit tests instead:
  pytest tests/smoke/ tests/unit/ -v
"""
import pytest
from core.backtesting.metrics import BacktestMetrics


@pytest.mark.integration
def test_holdbot_completes_full_backtest():
    """
    HoldBot should complete a full backtest and return valid metrics.
    Requires database with historical OHLCV data.
    """
    pytest.skip("Integration test: requires real PostgreSQL + OHLCV data")


@pytest.mark.integration
def test_dcabot_completes_full_backtest():
    """
    DCABot should complete a full backtest and return valid metrics.
    Requires database with historical OHLCV data.
    """
    pytest.skip("Integration test: requires real PostgreSQL + OHLCV data")


@pytest.mark.integration
def test_trendbot_completes_full_backtest():
    """
    TrendBot should complete a full backtest and return valid metrics.
    Requires database with historical OHLCV data.
    """
    pytest.skip("Integration test: requires real PostgreSQL + OHLCV data")


@pytest.mark.integration
def test_backtest_produces_valid_metrics():
    """
    Backtest should produce valid metrics with no NaN values.
    Requires database with historical OHLCV data.
    """
    pytest.skip("Integration test: requires real PostgreSQL + OHLCV data")


@pytest.mark.integration
def test_backtest_respects_date_filtering():
    """
    Backtest should correctly filter ticks by TRAIN_UNTIL and TEST_FROM.
    Requires database with historical OHLCV data.
    """
    pytest.skip("Integration test: requires real PostgreSQL + OHLCV data")
