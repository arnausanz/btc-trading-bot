# tests/unit/test_comparator.py
from bots.classical.hold_bot import HoldBot
from bots.classical.dca_bot import DCABot
from core.backtesting.comparator import BotComparator


def test_comparator_returns_sorted_results():
    bots = [
        HoldBot(config_path="config/bots/hold.yaml"),
        DCABot(config_path="config/bots/dca.yaml"),
    ]
    comparator = BotComparator(
        bots=bots,
        symbol="BTC/USDT",
        timeframe="1h",
    )
    results = comparator.run()

    assert len(results) == 2
    assert results[0]["sharpe_ratio"] >= results[1]["sharpe_ratio"]
    assert all("bot_id" in r for r in results)