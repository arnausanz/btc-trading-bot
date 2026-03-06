# tests/integration/test_backtest_integration.py
# Aquests tests necessiten connexió real a la DB (PostgreSQL amb dades).
# Executa'ls amb: poetry run pytest tests/integration/ -v
import pytest
from core.backtesting.engine import BacktestEngine


@pytest.mark.integration
def test_backtest_hold_bot():
    """HoldBot: retorn total ha de ser 0% (mai fa res)."""
    from bots.classical.hold_bot import HoldBot
    bot = HoldBot(config_path="config/bots/hold.yaml")
    engine = BacktestEngine(bot=bot)
    metrics = engine.run(symbol="BTC/USDT", timeframe="1h")
    summary = metrics.summary()
    assert summary["total_return_pct"] == 0.0
    assert summary["total_ticks"] > 0


@pytest.mark.integration
def test_grid_bot_backtest():
    """GridBot: ha de produir un historial vàlid."""
    from bots.classical.grid_bot import GridBot
    bot = GridBot(config_path="config/bots/grid.yaml")
    engine = BacktestEngine(bot=bot)
    metrics = engine.run(symbol="BTC/USDT", timeframe="1h")
    summary = metrics.summary()
    assert summary["total_ticks"] > 0
    assert "sharpe_ratio" in summary
