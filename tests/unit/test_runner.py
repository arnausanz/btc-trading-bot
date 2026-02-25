# tests/unit/test_runner.py
from core.engine.runner import Runner
from bots.classical.hold_bot import HoldBot
from exchanges.paper import PaperExchange


def test_runner_hold_bot():
    bot = HoldBot(config_path="config/bots/hold.yaml")
    exchange = PaperExchange(config_path="config/exchanges/paper.yaml")
    runner = Runner(bot=bot, exchange=exchange)

    history = runner.run(symbol="BTC/USDT", timeframe="1h")

    assert len(history) > 0
    initial_value = 10_000.0
    final_value = history[-1]["portfolio_value"]
    assert abs(final_value - initial_value) < 0.01