# tests/unit/test_backtesting.py
import numpy as np
import pandas as pd
import pytest
from core.backtesting.metrics import BacktestMetrics
from core.backtesting.engine import BacktestEngine
from bots.classical.hold_bot import HoldBot


def make_history(n: int = 100, initial: float = 10_000.0) -> list[dict]:
    """Genera historial sintètic per testejar les mètriques."""
    values = initial + np.cumsum(np.random.randn(n) * 10)
    return [{"portfolio_value": v, "order_status": "filled"} for v in values]


def test_total_return():
    history = [
        {"portfolio_value": 10_000.0, "order_status": "hold"},
        {"portfolio_value": 11_000.0, "order_status": "hold"},
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.total_return() == pytest.approx(10.0)


def test_max_drawdown_negative():
    history = [
        {"portfolio_value": 10_000.0, "order_status": "hold"},
        {"portfolio_value": 12_000.0, "order_status": "hold"},
        {"portfolio_value": 9_000.0, "order_status": "hold"},
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.max_drawdown() < 0


def test_sharpe_ratio_flat():
    history = [{"portfolio_value": 10_000.0, "order_status": "hold"}] * 50
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.sharpe_ratio() == 0.0


def test_summary_keys():
    m = BacktestMetrics(history=make_history(), initial_capital=10_000.0)
    summary = m.summary()
    expected_keys = [
        "total_return_pct", "sharpe_ratio", "max_drawdown_pct",
        "calmar_ratio", "win_rate_pct", "initial_capital",
        "final_capital", "total_ticks"
    ]
    for key in expected_keys:
        assert key in summary


def test_backtest_hold_bot():
    bot = HoldBot(config_path="config/bots/hold.yaml")
    engine = BacktestEngine(bot=bot)
    metrics = engine.run(symbol="BTC/USDT", timeframe="1h")
    summary = metrics.summary()
    assert summary["total_return_pct"] == 0.0
    assert summary["total_ticks"] > 0