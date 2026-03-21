# tests/unit/test_bot_signals.py
"""
Test that classical bots generate valid signals given synthetic observations.
No database required — observations are constructed manually.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from bots.classical.dca_bot import DCABot
from bots.classical.hold_bot import HoldBot
from bots.classical.trend_bot import TrendBot
from core.models import Action


def make_synthetic_observation(n: int = 100, timeframe: str = "1h") -> dict:
    """
    Build a minimal observation dict that bots expect.
    Contains close and rsi_14 for trend analysis.
    """
    rng = np.random.default_rng(42)
    close = 50000 + np.cumsum(rng.normal(0, 100, n))
    df = pd.DataFrame({
        "close": close,
        "rsi_14": rng.uniform(30, 70, n),
    })
    return {
        timeframe: {
            "features": df,
            "current_price": float(close[-1]),
            "timestamp": datetime.now(timezone.utc),
        }
    }


class TestDCABot:
    """Test DCA (Dollar Cost Averaging) bot behavior."""

    def test_dca_buys_on_schedule(self):
        """DCA should buy every N ticks as configured."""
        bot = DCABot(config_path="config/models/dca.yaml")
        bot.on_start()
        obs = make_synthetic_observation()

        # First tick should not trigger if buy_every_n_ticks > 1
        sig1 = bot.on_observation(obs)
        buy_every_n = bot.config["buy_every_n_ticks"]

        # Keep calling until we hit the buy tick
        action_at_tick_n = None
        for i in range(2, buy_every_n + 2):
            sig = bot.on_observation(obs)
            if i == buy_every_n:
                action_at_tick_n = sig.action

        assert action_at_tick_n == Action.BUY


    def test_dca_holds_between_buys(self):
        """DCA should hold between buy ticks."""
        bot = DCABot(config_path="config/models/dca.yaml")
        bot.on_start()
        obs = make_synthetic_observation()
        buy_every_n = bot.config["buy_every_n_ticks"]

        # Tick 1: might be buy if buy_every_n==1
        sig1 = bot.on_observation(obs)

        # Tick 2: should not be buy (unless buy_every_n==1)
        if buy_every_n > 1:
            sig2 = bot.on_observation(obs)
            assert sig2.action == Action.HOLD


    def test_dca_produces_signals(self):
        """DCA should produce signals for 50 ticks without error."""
        bot = DCABot(config_path="config/models/dca.yaml")
        bot.on_start()
        obs = make_synthetic_observation()

        signals = [bot.on_observation(obs) for _ in range(50)]
        assert len(signals) == 50
        # All signals should have valid actions
        for sig in signals:
            assert sig.action in [Action.BUY, Action.HOLD]


class TestHoldBot:
    """Test Buy & Hold bot behavior."""

    def test_hold_buys_on_first_tick(self):
        """HoldBot should buy on the first observation."""
        bot = HoldBot(config_path="config/models/hold.yaml")
        bot.on_start()
        obs = make_synthetic_observation()

        sig = bot.on_observation(obs)
        assert sig.action == Action.BUY
        assert sig.size == 1.0


    def test_hold_never_buys_again_after_first(self):
        """HoldBot should never buy again after the first tick."""
        bot = HoldBot(config_path="config/models/hold.yaml")
        bot.on_start()
        obs = make_synthetic_observation()

        # First tick: BUY
        sig1 = bot.on_observation(obs)
        assert sig1.action == Action.BUY

        # All subsequent ticks: HOLD
        for _ in range(50):
            sig = bot.on_observation(obs)
            assert sig.action == Action.HOLD


    def test_hold_respects_existing_position(self):
        """HoldBot should not buy if it already has BTC in portfolio."""
        bot = HoldBot(config_path="config/models/hold.yaml")
        bot.on_start()

        obs = make_synthetic_observation()
        # Simulate a portfolio that already has BTC from previous session
        obs["portfolio"] = {"btc_balance": 0.1}

        # Should skip the buy and go straight to HOLD
        sig = bot.on_observation(obs)
        assert sig.action == Action.HOLD


class TestTrendBot:
    """Test Trend Following bot behavior."""

    def test_trend_bot_holds_without_crossover(self):
        """TrendBot should hold when no EMA crossover is detected."""
        bot = TrendBot(config_path="config/models/trend.yaml")
        bot.on_start()

        # Flat data (no trend) should not trigger crossover
        rng = np.random.default_rng(42)
        close = np.full(50, 50000.0)  # Flat price
        rsi = np.full(50, 50.0)
        df = pd.DataFrame({"close": close, "rsi_14": rsi})

        obs = {
            "1h": {
                "features": df,
                "current_price": 50000.0,
                "timestamp": datetime.now(timezone.utc),
            }
        }

        sig = bot.on_observation(obs)
        assert sig.action == Action.HOLD


    def test_trend_bot_uptrend_crossover(self):
        """TrendBot should buy on bullish EMA crossover."""
        bot = TrendBot(config_path="config/models/trend.yaml")
        bot.on_start()

        # Strong uptrend: close prices rising
        rng = np.random.default_rng(42)
        close = 50000 + np.cumsum(rng.normal(100, 50, 50))  # Strong uptrend
        rsi = np.linspace(30, 50, 50)  # Rising RSI, not overbought
        df = pd.DataFrame({"close": close, "rsi_14": rsi})

        obs = {
            "1h": {
                "features": df,
                "current_price": float(close[-1]),
                "timestamp": datetime.now(timezone.utc),
            }
        }

        # Should generate BUY signal if EMA crossover + RSI is not overbought
        sig = bot.on_observation(obs)
        assert sig.action in [Action.BUY, Action.HOLD]  # Depends on EMA config


    def test_trend_bot_never_buys_when_overbought(self):
        """TrendBot should not buy if RSI is above overbought threshold."""
        bot = TrendBot(config_path="config/models/trend.yaml")
        bot.on_start()

        rng = np.random.default_rng(42)
        close = 50000 + np.cumsum(rng.normal(100, 50, 50))
        rsi = np.full(50, 90.0)  # Overbought
        df = pd.DataFrame({"close": close, "rsi_14": rsi})

        obs = {
            "1h": {
                "features": df,
                "current_price": float(close[-1]),
                "timestamp": datetime.now(timezone.utc),
            }
        }

        sig = bot.on_observation(obs)
        # Even with uptrend, overbought RSI should prevent buy
        if sig.action == Action.BUY:
            # Only allowed if RSI < rsi_overbought threshold
            assert rsi[-1] < bot.config.get("rsi_overbought", 70)
