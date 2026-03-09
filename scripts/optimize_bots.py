# scripts/optimize_bots.py
"""
Optimizes classical bots with Optuna (walk-forward, data up to TRAIN_UNTIL).

El search space es llegeix del YAML unificat (config/models/{bot}.yaml),
secció optimization.search_space. No cal modificar codi Python per canviar
el search space — tot és configurable via YAML.

Usage:
  python scripts/optimize_bots.py                    # optimize all
  python scripts/optimize_bots.py --bots dca trend   # only DCA and Trend
  python scripts/optimize_bots.py --bots grid        # only Grid
  python scripts/optimize_bots.py --trials 50        # override n_trials per tots
"""
import argparse
import logging
import sys
import yaml

sys.path.append(".")

from bots.classical.dca_bot            import DCABot
from bots.classical.trend_bot          import TrendBot
from bots.classical.grid_bot           import GridBot
from bots.classical.mean_reversion_bot import MeanReversionBot
from bots.classical.momentum_bot       import MomentumBot
from core.backtesting.optimizer import BotOptimizer
from core.config import TRAIN_UNTIL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Registry de bots optimitzables ──────────────────────────────────────────
# El search space s'obté del YAML (secció optimization.search_space)
# en lloc d'estar hardcoded aquí. Afegir un nou bot = 1 línia.
ALL_BOTS = {
    "dca":            {"class": DCABot,           "config": "config/models/dca.yaml"},
    "trend":          {"class": TrendBot,          "config": "config/models/trend.yaml"},
    "grid":           {"class": GridBot,           "config": "config/models/grid.yaml"},
    "mean_reversion": {"class": MeanReversionBot,  "config": "config/models/mean_reversion.yaml"},
    "momentum":       {"class": MomentumBot,       "config": "config/models/momentum.yaml"},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize classical bots with Optuna")
    parser.add_argument(
        "--bots", nargs="+", default=list(ALL_BOTS.keys()),
        choices=list(ALL_BOTS.keys()),
        metavar="BOT",
        help=f"Bots to optimize (default: all). Options: {list(ALL_BOTS.keys())}",
    )
    parser.add_argument(
        "--trials", type=int, default=None,
        help="Override n_trials (default: usa el valor del YAML optimization.n_trials)"
    )
    args = parser.parse_args()

    logger.info(f"Walk-forward: optimizing ONLY on data up to {TRAIN_UNTIL}")
    logger.info(f"Selected bots: {args.bots}")

    studies = {}
    for bot_key in args.bots:
        cfg = ALL_BOTS[bot_key]

        # Llegim search_space i n_trials directament del YAML unificat
        with open(cfg["config"]) as f:
            bot_yaml = yaml.safe_load(f)
        opt_cfg = bot_yaml.get("optimization", {})
        param_space = opt_cfg.get("search_space", {})
        n_trials = args.trials if args.trials is not None else opt_cfg.get("n_trials", 30)

        logger.info(f"Optimizing {bot_key.upper()}Bot | {n_trials} trials...")
        optimizer = BotOptimizer(
            bot_class=cfg["class"],
            base_config_path=cfg["config"],
            param_space=param_space,
            n_trials=n_trials,
            train_until=TRAIN_UNTIL,
        )
        study = optimizer.run()
        studies[bot_key] = study

        # Guarda el millor config a config/models/{bot}_optimized.yaml
        out_path = f"config/models/{bot_key}_optimized.yaml"
        optimizer.save_best_config(study, out_path)

    logger.info("=== FINAL OPTIMIZATION SUMMARY ===")
    for bot_key, study in studies.items():
        logger.info(f"{bot_key.upper():<6} → Sharpe: {study.best_value:.4f} | Params: {study.best_params}")
