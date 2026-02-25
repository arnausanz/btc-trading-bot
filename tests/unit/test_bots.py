# tests/unit/test_bots.py
from core.backtesting.engine import BacktestEngine
from bots.classical.dca_bot import DCABot
from bots.classical.trend_bot import TrendBot
from core.models import Action


def test_dca_bot_backtest():
    bot = DCABot(config_path="config/bots/dca.yaml")
    engine = BacktestEngine(bot=bot)
    metrics = engine.run(symbol="BTC/USDT", timeframe="1h")
    summary = metrics.summary()
    assert summary["total_ticks"] > 0
    # DCA sempre compra, mai ha de tenir retorn exactament 0
    assert summary["total_return_pct"] != 0.0


def test_trend_bot_backtest():
    bot = TrendBot(config_path="config/bots/trend.yaml")
    engine = BacktestEngine(bot=bot)
    metrics = engine.run(symbol="BTC/USDT", timeframe="1h")
    summary = metrics.summary()
    assert summary["total_ticks"] > 0
    assert "sharpe_ratio" in summary