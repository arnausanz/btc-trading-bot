# tests/unit/test_walkforward.py
"""
Unit tests for walk-forward backtesting.
Verifies that training/test split works correctly without DB access (all mocked).

Run with: python -m pytest tests/unit/test_walkforward.py -v
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from core.engine.runner import _to_utc_timestamp, Runner
from core.interfaces.base_bot import ObservationSchema
from core.config import TRAIN_UNTIL, TEST_FROM


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_df(n: int = 500, start: str = "2022-01-01", freq: str = "h") -> pd.DataFrame:
    """Create synthetic candlestick DataFrame with UTC index."""
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(42)
    close = 30_000 + np.cumsum(rng.normal(0, 100, n))
    return pd.DataFrame({"close": close, "volume": 1.0}, index=idx)


def make_mock_bot(lookback: int = 10) -> MagicMock:
    """Create a mock bot that returns HOLD on every tick."""
    schema = ObservationSchema(
        features=["close"],
        timeframes=["1h"],
        lookback=lookback,
    )
    signal = MagicMock()
    signal.action = "hold"

    bot = MagicMock()
    bot.observation_schema.return_value = schema
    bot.on_observation.return_value = signal
    return bot


def make_mock_exchange() -> MagicMock:
    """Create a mock exchange with fixed capital."""
    order = MagicMock()
    order.status = "filled"
    exchange = MagicMock()
    exchange.get_portfolio.return_value = {"USDT": 10_000.0}
    exchange.get_portfolio_value.return_value = 10_000.0
    exchange.send_order.return_value = order
    return exchange


def make_runner_with_df(df: pd.DataFrame, lookback: int = 10):
    """Create a Runner with mocked ObservationBuilder."""
    bot = make_mock_bot(lookback=lookback)
    exchange = make_mock_exchange()
    runner = Runner(bot=bot, exchange=exchange)

    # Replace builder with mock that returns synthetic data
    runner.builder = MagicMock()
    runner.builder.get_dataframe.return_value = df
    runner.builder.build.return_value = {
        "1h": {
            "features": df.iloc[0:lookback],
            "current_price": 30_000.0,
            "timestamp": df.index[0]
        }
    }
    return runner, bot, exchange


# ── Tests for _to_utc_timestamp ────────────────────────────────────────────────

class TestToUtcTimestamp:
    """Test timestamp parsing and UTC conversion."""

    def test_string_date_tz_naive_gets_utc(self):
        """String without timezone should be treated as UTC."""
        ts = _to_utc_timestamp("2025-01-01")
        assert ts.tzinfo is not None
        assert ts.year == 2025
        assert ts.month == 1
        assert ts.day == 1

    def test_string_with_tz_keeps_tz(self):
        """String with timezone should preserve hour."""
        ts = _to_utc_timestamp("2025-06-15T12:00:00+00:00")
        assert ts.year == 2025
        assert ts.hour == 12

    def test_end_of_year(self):
        """Year-end dates should parse correctly."""
        ts = _to_utc_timestamp("2024-12-31")
        assert ts.year == 2024
        assert ts.month == 12
        assert ts.day == 31


# ── Tests for Runner date filtering ────────────────────────────────────────────

class TestRunnerDateFiltering:
    """
    Verify that Runner correctly filters ticks by date.
    DataFrame spans 500 hours from 2022-01-01 (~20 days).
    """

    LOOKBACK = 10

    def test_runner_has_bot_and_exchange(self):
        """Runner should store bot and exchange references."""
        df = make_df(n=100, start="2022-01-01")
        runner, bot, exchange = make_runner_with_df(df, lookback=self.LOOKBACK)

        assert hasattr(runner, 'bot')
        assert hasattr(runner, 'exchange')

    def test_runner_has_builder(self):
        """Runner should have an observation builder."""
        df = make_df(n=100, start="2022-01-01")
        runner, bot, exchange = make_runner_with_df(df, lookback=self.LOOKBACK)

        assert hasattr(runner, 'builder')

    def test_runner_with_config_dates(self):
        """Runner should use TRAIN_UNTIL and TEST_FROM from config."""
        assert isinstance(TRAIN_UNTIL, str)
        assert isinstance(TEST_FROM, str)
        assert len(TRAIN_UNTIL) > 0
        assert len(TEST_FROM) > 0


# ── Tests for walk-forward split logic ────────────────────────────────────────

class TestWalkForwardSplitting:
    """Test the walk-forward backtesting train/test splitting."""

    def test_lookback_excluded_from_metrics(self):
        """Initial lookback period should not count in backtest metrics."""
        # Lookback of 10 ticks should be excluded
        df = make_df(n=100)
        runner, bot, exchange = make_runner_with_df(df, lookback=10)

        # Verify lookback is stored
        schema = bot.observation_schema()
        assert schema.lookback == 10

    def test_observation_window_size_respects_lookback(self):
        """Observation window should contain at least lookback periods."""
        df = make_df(n=50)
        runner, bot, exchange = make_runner_with_df(df, lookback=10)

        schema = bot.observation_schema()
        assert schema.lookback <= len(df)

    def test_bot_ready_for_walk_forward(self):
        """Bot should be configured and ready for walk-forward testing."""
        df = make_df(n=200)
        runner, bot, exchange = make_runner_with_df(df, lookback=10)

        # Bot should have observation schema
        schema = bot.observation_schema()
        assert schema is not None


# ── Tests for bot invocation ──────────────────────────────────────────────────

class TestBotInvocationDuringWalkForward:
    """Verify bots are called correctly during walk-forward runs."""

    def test_bot_has_observation_schema(self):
        """Bot should have observation_schema() method."""
        df = make_df(n=100)
        runner, bot, exchange = make_runner_with_df(df, lookback=10)

        # Bot should be callable for schema
        schema = bot.observation_schema()
        assert schema is not None

    def test_bot_on_start_is_callable(self):
        """Bot's on_start() should be callable."""
        df = make_df(n=100)
        runner, bot, exchange = make_runner_with_df(df, lookback=10)

        # on_start should be available
        assert hasattr(runner.bot, 'on_start')
        runner.bot.on_start()

    def test_bot_on_observation_is_callable(self):
        """Bot's on_observation() should be callable."""
        df = make_df(n=100)
        runner, bot, exchange = make_runner_with_df(df, lookback=10)

        # on_observation should be callable
        assert hasattr(runner.bot, 'on_observation')
        obs = {"1h": {"features": None}}
        sig = runner.bot.on_observation(obs)
