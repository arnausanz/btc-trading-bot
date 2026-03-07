# scripts/run_comparison.py
import logging
import sys

sys.path.append(".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from bots.classical.hold_bot import HoldBot
from bots.classical.dca_bot import DCABot
from bots.classical.trend_bot import TrendBot
from bots.classical.grid_bot import GridBot
from bots.ml.ml_bot import MLBot
from core.backtesting.comparator import BotComparator

if __name__ == "__main__":
    bots = [
        HoldBot(config_path="config/bots/hold.yaml"),
        DCABot(config_path="config/bots/dca.yaml"),
        TrendBot(config_path="config/bots/trend.yaml"),
        GridBot(config_path="config/bots/grid.yaml"),
        MLBot(config_path="config/bots/ml_bot.yaml"),
    ]
    # train_until i test_from llegits automàticament de config/settings.yaml
    comparator = BotComparator(bots=bots, symbol="BTC/USDT", timeframe="1h")
    comparator.run()