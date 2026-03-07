# scripts/optimize_bots.py
"""
Optimizes classical bots with Optuna (walk-forward, data up to TRAIN_UNTIL).

Usage:
  python scripts/optimize_bots.py                    # optimize all
  python scripts/optimize_bots.py --bots dca trend   # only DCA and Trend
  python scripts/optimize_bots.py --bots grid        # only Grid
  python scripts/optimize_bots.py --trials 50        # 50 trials instead of 30
"""
import argparse
import logging
import sys

sys.path.append(".")

from bots.classical.dca_bot   import DCABot
from bots.classical.trend_bot import TrendBot
from bots.classical.grid_bot  import GridBot
from core.backtesting.optimizer import BotOptimizer
from core.config import TRAIN_UNTIL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Definition of all optimizable bots ──────────────────────────────────
ALL_BOTS = {
    "dca": {
        "class": DCABot,
        "config": "config/bots/dca.yaml",
        "space": {
            "buy_every_n_ticks": {"type": "int",   "low": 6,    "high": 168},
            "buy_size":          {"type": "float",  "low": 0.05, "high": 0.5},
        },
    },
    "trend": {
        "class": TrendBot,
        "config": "config/bots/trend.yaml",
        "space": {
            "ema_fast":       {"type": "int",   "low": 5,   "high": 50},
            "ema_slow":       {"type": "int",   "low": 20,  "high": 200},
            "trade_size":     {"type": "float", "low": 0.1, "high": 1.0},
            "rsi_overbought": {"type": "int",   "low": 60,  "high": 85},
            "rsi_oversold":   {"type": "int",   "low": 15,  "high": 40},
        },
    },
    "grid": {
        "class": GridBot,
        "config": "config/bots/grid.yaml",
        "space": {
            "level_size":      {"type": "float", "low": 0.02, "high": 0.3},
            "rsi_filter_high": {"type": "int",   "low": 60,   "high": 85},
            "rsi_filter_low":  {"type": "int",   "low": 15,   "high": 40},
        },
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize classical bots with Optuna")
    parser.add_argument(
        "--bots", nargs="+", default=list(ALL_BOTS.keys()),
        choices=list(ALL_BOTS.keys()),
        metavar="BOT",
        help=f"Bots to optimize (default: all). Options: {list(ALL_BOTS.keys())}",
    )
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials (default: 30)")
    args = parser.parse_args()

    logger.info(f"Walk-forward: optimizing ONLY on data up to {TRAIN_UNTIL}")
    logger.info(f"Selected bots: {args.bots} | Trials: {args.trials}")

    studies = {}
    for bot_key in args.bots:
        cfg     = ALL_BOTS[bot_key]
        logger.info(f"Optimizing {bot_key.upper()}Bot...")
        optimizer = BotOptimizer(
            bot_class=cfg["class"],
            base_config_path=cfg["config"],
            param_space=cfg["space"],
            n_trials=args.trials,
            train_until=TRAIN_UNTIL,
        )
        study = optimizer.run()
        studies[bot_key] = study

        # Save the best config to config/bots/{bot_name}_optimized.yaml
        out_path = f"config/bots/{bot_key}_optimized.yaml"
        optimizer.save_best_config(study, out_path)

    logger.info("=== FINAL OPTIMIZATION SUMMARY ===")
    for bot_key, study in studies.items():
        logger.info(f"{bot_key.upper():<6} → Sharpe: {study.best_value:.4f} | Params: {study.best_params}")
