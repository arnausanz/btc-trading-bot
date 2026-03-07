# tests/smoke/test_backtest_metrics.py
"""
Comprehensive tests for BacktestMetrics.
No database required — all tests use synthetic history data.
"""
import numpy as np
import pandas as pd
import pytest

from core.backtesting.metrics import BacktestMetrics, _PERIODS_PER_YEAR


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_history(n: int = 100, initial: float = 10_000.0, seed: int = 42) -> list[dict]:
    """Synthetic backtest history (no timestamps). Fixed seed for determinism."""
    rng = np.random.default_rng(seed)
    values = initial + np.cumsum(rng.normal(0, 10, n))
    return [{"portfolio_value": float(v), "order_status": "hold"} for v in values]


def make_history_with_timestamps(
    n: int = 100,
    initial: float = 10_000.0,
    timeframe: str = "1h",
    seed: int = 42,
) -> list[dict]:
    """Backtest history with real timestamps for testing _duration_days."""
    rng = np.random.default_rng(seed)
    values = initial + np.cumsum(rng.normal(0, 10, n))
    freq_map = {"1h": "h", "4h": "4h", "1d": "D", "1w": "W"}
    freq = freq_map.get(timeframe, "h")
    timestamps = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    return [
        {
            "portfolio_value": float(v),
            "order_status": "hold",
            "signal": "hold",
            "timestamp": ts,
        }
        for v, ts in zip(values, timestamps)
    ]


# ── Basic Tests: total_return, max_drawdown ────────────────────────────────────

def test_total_return_positive():
    history = [
        {"portfolio_value": 10_000.0, "order_status": "hold"},
        {"portfolio_value": 11_000.0, "order_status": "hold"},
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.total_return() == pytest.approx(10.0)


def test_total_return_negative():
    history = [
        {"portfolio_value": 10_000.0, "order_status": "hold"},
        {"portfolio_value": 9_000.0, "order_status": "hold"},
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.total_return() == pytest.approx(-10.0)


def test_total_return_zero():
    history = [{"portfolio_value": 10_000.0, "order_status": "hold"}] * 10
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.total_return() == pytest.approx(0.0)


def test_max_drawdown_is_negative():
    history = [
        {"portfolio_value": 10_000.0, "order_status": "hold"},
        {"portfolio_value": 12_000.0, "order_status": "hold"},
        {"portfolio_value": 9_000.0, "order_status": "hold"},
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.max_drawdown() < 0


def test_max_drawdown_exact():
    """Rise to 12,000 then fall to 9,000 → drawdown = -25%."""
    history = [
        {"portfolio_value": 10_000.0, "order_status": "hold"},
        {"portfolio_value": 12_000.0, "order_status": "hold"},
        {"portfolio_value": 9_000.0, "order_status": "hold"},
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.max_drawdown() == pytest.approx(-25.0)


def test_max_drawdown_no_drawdown():
    """Always increasing portfolio → drawdown = 0%."""
    values = [10_000.0 * (1.01 ** i) for i in range(50)]
    history = [{"portfolio_value": v, "order_status": "hold"} for v in values]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.max_drawdown() == pytest.approx(0.0)


def test_max_drawdown_continuous_decline():
    """Monotonically declining portfolio."""
    values = [10_000.0 * (0.99 ** i) for i in range(50)]
    history = [{"portfolio_value": v, "order_status": "hold"} for v in values]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    dd = m.max_drawdown()
    assert dd < 0
    assert dd < -30  # Should be around -39%


# ── Sharpe Ratio Tests ────────────────────────────────────────────────────────

def test_sharpe_ratio_flat_portfolio():
    """No changes → std = 0 → Sharpe = 0."""
    history = [{"portfolio_value": 10_000.0, "order_status": "hold"}] * 50
    m = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    assert m.sharpe_ratio() == 0.0


def test_sharpe_positive_for_growing_portfolio():
    """Always increasing → positive Sharpe."""
    values = [10_000.0 * (1.001 ** i) for i in range(200)]
    history = [{"portfolio_value": v, "order_status": "hold"} for v in values]
    m = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    assert m.sharpe_ratio() > 0


def test_sharpe_negative_for_declining_portfolio():
    """Always decreasing → negative Sharpe."""
    values = [10_000.0 * (0.999 ** i) for i in range(200)]
    history = [{"portfolio_value": v, "order_status": "hold"} for v in values]
    m = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    assert m.sharpe_ratio() < 0


def test_sharpe_annualization_1h_vs_1d():
    """
    On same return sequence, Sharpe(1h) / Sharpe(1d) = sqrt(24).
    Verifies correct annualization factor for each timeframe.
    """
    history = make_history(n=500, seed=42)
    m_1h = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    m_1d = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1d")

    s_1h = m_1h.sharpe_ratio()
    s_1d = m_1d.sharpe_ratio()

    assert s_1d != 0, "Sharpe(1d) should be non-zero with random data"
    ratio = s_1h / s_1d
    assert ratio == pytest.approx(np.sqrt(24), rel=0.01)


def test_sharpe_annualization_1h_vs_4h():
    """Sharpe(1h) / Sharpe(4h) = sqrt(4) = 2."""
    history = make_history(n=500, seed=42)
    m_1h = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    m_4h = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="4h")

    s_1h = m_1h.sharpe_ratio()
    s_4h = m_4h.sharpe_ratio()

    assert s_4h != 0
    ratio = s_1h / s_4h
    assert ratio == pytest.approx(np.sqrt(4), rel=0.01)


def test_sharpe_periods_per_year_values():
    """_PERIODS_PER_YEAR dictionary should have correct values."""
    assert _PERIODS_PER_YEAR["1h"] == 365 * 24        # 8,760
    assert _PERIODS_PER_YEAR["4h"] == 365 * 6         # 2,190
    assert _PERIODS_PER_YEAR["1d"] == 365
    assert _PERIODS_PER_YEAR["1w"] == 52
    assert _PERIODS_PER_YEAR["1m"] == 365 * 24 * 60


def test_sharpe_unknown_timeframe_defaults_to_1h():
    """Unknown timeframe should use 1h factor as default."""
    history = make_history(n=100, seed=42)
    m_unknown = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="7h")
    m_1h = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    assert m_unknown.sharpe_ratio() == pytest.approx(m_1h.sharpe_ratio())


# ── Calmar Ratio Tests ────────────────────────────────────────────────────────

def test_calmar_zero_drawdown():
    """No drawdown → Calmar = 0 (by convention: undefined)."""
    values = [10_000.0 * (1.001 ** i) for i in range(50)]
    history = [{"portfolio_value": v, "order_status": "hold"} for v in values]
    m = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    assert m.calmar_ratio() == 0.0


def test_calmar_with_drawdown():
    """Portfolio with significant drawdown should have negative return vs drawdown."""
    history = [
        {"portfolio_value": 10_000.0, "order_status": "hold"},
        {"portfolio_value": 12_000.0, "order_status": "hold"},
        {"portfolio_value": 8_000.0, "order_status": "hold"},
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1d")
    calmar = m.calmar_ratio()
    # With negative return (-20%) and 33% drawdown, Calmar should be negative
    assert m.total_return() < 0 or m.max_drawdown() < 0


# ── Duration Tests ────────────────────────────────────────────────────────────

def test_duration_days_with_timestamps_1h():
    """360 hourly ticks = 15 days exactly."""
    history = make_history_with_timestamps(n=361, timeframe="1h")
    m = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    assert m._duration_days() == pytest.approx(360 / 24, abs=0.1)


def test_duration_days_with_timestamps_1d():
    """365 daily ticks = ~364 days."""
    history = make_history_with_timestamps(n=365, timeframe="1d")
    m = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1d")
    assert m._duration_days() == pytest.approx(364, abs=1)


def test_duration_days_fallback_without_timestamps():
    """Without timestamps, fallback should estimate days consistently."""
    history = make_history(n=8760)  # 1 year of hourly data
    m = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    # 8760 ticks / 24 periods_per_day = 365 days
    assert m._duration_days() == pytest.approx(365.0, rel=0.01)


def test_duration_days_1h_vs_1d_same_n_ticks():
    """
    Same number of ticks, different timeframes:
    100 ticks 1h → ~4.17 days
    100 ticks 1d → ~100 days
    """
    history = make_history(n=100)
    m_1h = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    m_1d = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1d")
    assert m_1d._duration_days() > m_1h._duration_days()
    assert m_1d._duration_days() / m_1h._duration_days() == pytest.approx(24, rel=0.01)


# ── Win Rate Tests ────────────────────────────────────────────────────────────

def test_win_rate_no_signal_column():
    """History without 'signal' column → win_rate = 0."""
    m = BacktestMetrics(history=make_history(), initial_capital=10_000.0)
    assert m.win_rate() == 0.0


def test_win_rate_no_trades_only_hold():
    """All HOLD signals → 0 completed trades → win_rate = 0."""
    history = [
        {"portfolio_value": 10_000.0, "signal": "hold", "order_status": "hold"},
        {"portfolio_value": 10_500.0, "signal": "hold", "order_status": "hold"},
        {"portfolio_value": 11_000.0, "signal": "hold", "order_status": "hold"},
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.win_rate() == 0.0


def test_win_rate_open_position_not_counted():
    """BUY without SELL → open position, not a completed trade."""
    history = [
        {"portfolio_value": 10_000.0, "signal": "buy",  "order_status": "filled"},
        {"portfolio_value": 11_000.0, "signal": "hold", "order_status": "filled"},
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.win_rate() == 0.0


def test_win_rate_all_winning():
    """2 completed profitable trades → win_rate = 100%."""
    history = [
        {"portfolio_value": 10_000.0, "signal": "buy",  "order_status": "filled"},
        {"portfolio_value": 10_500.0, "signal": "hold", "order_status": "hold"},
        {"portfolio_value": 11_000.0, "signal": "sell", "order_status": "filled"},
        {"portfolio_value": 11_000.0, "signal": "buy",  "order_status": "filled"},
        {"portfolio_value": 12_000.0, "signal": "sell", "order_status": "filled"},
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.win_rate() == pytest.approx(100.0)


def test_win_rate_all_losing():
    """2 completed losing trades → win_rate = 0%."""
    history = [
        {"portfolio_value": 11_000.0, "signal": "buy",  "order_status": "filled"},
        {"portfolio_value": 10_000.0, "signal": "sell", "order_status": "filled"},
        {"portfolio_value": 10_000.0, "signal": "buy",  "order_status": "filled"},
        {"portfolio_value": 9_000.0,  "signal": "sell", "order_status": "filled"},
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.win_rate() == pytest.approx(0.0)


def test_win_rate_two_wins_one_loss():
    """2 winning, 1 losing → win_rate = 66.67%."""
    history = [
        {"portfolio_value": 10_000.0, "signal": "buy",  "order_status": "filled"},
        {"portfolio_value": 11_000.0, "signal": "sell", "order_status": "filled"},
        {"portfolio_value": 11_000.0, "signal": "buy",  "order_status": "filled"},
        {"portfolio_value": 10_000.0, "signal": "sell", "order_status": "filled"},
        {"portfolio_value": 10_000.0, "signal": "buy",  "order_status": "filled"},
        {"portfolio_value": 10_500.0, "signal": "sell", "order_status": "filled"},
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.win_rate() == pytest.approx(100 * 2 / 3, rel=0.01)


def test_win_rate_breakeven_not_winning():
    """Portfolio equal to entry price → not counted as winner."""
    history = [
        {"portfolio_value": 10_000.0, "signal": "buy",  "order_status": "filled"},
        {"portfolio_value": 10_000.0, "signal": "sell", "order_status": "filled"},
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.win_rate() == pytest.approx(0.0)


# ── Summary Tests ──────────────────────────────────────────────────────────────

def test_summary_contains_all_keys():
    m = BacktestMetrics(history=make_history(), initial_capital=10_000.0)
    summary = m.summary()
    expected_keys = [
        "total_return_pct", "sharpe_ratio", "max_drawdown_pct",
        "calmar_ratio", "win_rate_pct", "initial_capital",
        "final_capital", "total_ticks",
    ]
    for key in expected_keys:
        assert key in summary, f"Missing key in summary: {key}"


def test_summary_total_ticks():
    history = make_history(n=150)
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.summary()["total_ticks"] == 150


def test_summary_initial_capital_preserved():
    history = make_history(n=50)
    m = BacktestMetrics(history=history, initial_capital=12_345.0)
    assert m.summary()["initial_capital"] == 12_345.0


def test_summary_final_capital():
    history = [
        {"portfolio_value": 10_000.0, "order_status": "hold"},
        {"portfolio_value": 12_345.0, "order_status": "hold"},
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.summary()["final_capital"] == 12_345.0
