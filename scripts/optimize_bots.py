# scripts/optimize_bots.py
import logging
import sys

sys.path.append(".")

from bots.classical.dca_bot import DCABot
from bots.classical.trend_bot import TrendBot
from bots.classical.grid_bot import GridBot
from core.backtesting.optimizer import BotOptimizer
from core.config import TRAIN_UNTIL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info(f"Walk-forward: optimitzant ÚNICAMENT sobre dades fins a {TRAIN_UNTIL}")

    # Optimitza DCABot
    logger.info("Optimitzant DCABot...")
    dca_optimizer = BotOptimizer(
        bot_class=DCABot,
        base_config_path="config/bots/dca.yaml",
        param_space={
            "buy_every_n_ticks": {"type": "int", "low": 6, "high": 168},
            "buy_size": {"type": "float", "low": 0.05, "high": 0.5},
        },
        n_trials=30,
        train_until=TRAIN_UNTIL,
    )
    dca_study = dca_optimizer.run()
    logger.info(f"DCA millors params: {dca_study.best_params}")

    # Optimitza TrendBot
    logger.info("Optimitzant TrendBot...")
    trend_optimizer = BotOptimizer(
        bot_class=TrendBot,
        base_config_path="config/bots/trend.yaml",
        param_space={
            "ema_fast": {"type": "int", "low": 5, "high": 50},
            "ema_slow": {"type": "int", "low": 20, "high": 200},
            "trade_size": {"type": "float", "low": 0.1, "high": 1.0},
            "rsi_overbought": {"type": "int", "low": 60, "high": 85},
            "rsi_oversold": {"type": "int", "low": 15, "high": 40},
        },
        n_trials=30,
        train_until=TRAIN_UNTIL,
    )
    trend_study = trend_optimizer.run()
    logger.info(f"Trend millors params: {trend_study.best_params}")

    # Optimitza GridBot
    logger.info("Optimitzant GridBot...")
    grid_optimizer = BotOptimizer(
        bot_class=GridBot,
        base_config_path="config/bots/grid.yaml",
        param_space={
            "level_size": {"type": "float", "low": 0.02, "high": 0.3},
            "rsi_filter_high": {"type": "int", "low": 60, "high": 85},
            "rsi_filter_low": {"type": "int", "low": 15, "high": 40},
        },
        n_trials=30,
        train_until=TRAIN_UNTIL,
    )
    grid_study = grid_optimizer.run()
    logger.info(f"Grid millors params: {grid_study.best_params}")
    logger.info("=== RESUM FINAL OPTIMITZACIÓ ===")
    logger.info(f"DCA   → Sharpe: {dca_study.best_value:.4f} | Params: {dca_study.best_params}")
    logger.info(f"Trend → Sharpe: {trend_study.best_value:.4f} | Params: {trend_study.best_params}")
    logger.info(f"Grid  → Sharpe: {grid_study.best_value:.4f} | Params: {grid_study.best_params}")