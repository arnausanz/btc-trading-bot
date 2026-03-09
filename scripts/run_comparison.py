# scripts/run_comparison.py
"""
Compares all bots (classical, ML, RL) on train and test periods.

Llegeix configs unificades de config/models/*.yaml.
El camp 'category' del YAML determina el tipus de bot (classic/ML/RL).

Usage:
  python scripts/run_comparison.py                          # all available bots
  python scripts/run_comparison.py --bots hold trend dca    # selection
  python scripts/run_comparison.py --bots rf xgb ppo
  python scripts/run_comparison.py --no-rl                  # all except RL (if not trained)

Available bots:
  Classic: hold, dca, trend, grid, mean_reversion, momentum
  ML     : rf, xgb, lgbm, catboost, gru, patchtst
  RL     : ppo, sac (require running train_rl.py)
"""
import argparse
import logging
import os
import sys
import yaml

sys.path.append(".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Registry de tots els bots: clau → YAML base ─────────────────────────────
BOT_REGISTRY = {
    # Clàssics
    "hold":           "config/models/hold.yaml",
    "dca":            "config/models/dca.yaml",
    "trend":          "config/models/trend.yaml",
    "grid":           "config/models/grid.yaml",
    "mean_reversion": "config/models/mean_reversion.yaml",
    "momentum":       "config/models/momentum.yaml",
    # ML supervisat
    "rf":       "config/models/random_forest.yaml",
    "xgb":      "config/models/xgboost.yaml",
    "lgbm":     "config/models/lightgbm.yaml",
    "catboost": "config/models/catboost.yaml",
    "gru":      "config/models/gru.yaml",
    "patchtst": "config/models/patchtst.yaml",
    # RL
    "ppo":      "config/models/ppo.yaml",
    "sac":      "config/models/sac.yaml",
}

# Mapeig clau → classe de bot clàssic (per a instanciació)
_CLASSICAL_CLASSES = {
    "hold": None,   # s'omple lazy a _instantiate_bot
    "dca":  None,
    "trend": None,
    "grid":  None,
}


def get_best_config_path(bot_key: str) -> str:
    """
    Retorna el millor config disponible per a un bot.

    Prefereix {bot_key}_optimized.yaml si existeix (generat per Optuna),
    sinó usa el YAML base del registry.
    """
    default = BOT_REGISTRY[bot_key]
    # Construïm el nom _optimized a partir del YAML base
    optimized = default.replace(".yaml", "_optimized.yaml")
    if os.path.exists(optimized):
        logger.info(f"Using optimized config: {optimized}")
        return optimized
    logger.info(f"Using default config: {default}")
    return default


def _instantiate_bot(key: str):
    """
    Instancia el bot corresponent llegint el camp 'category' del YAML.
    Retorna None si el model RL no existeix (no entrenat).
    """
    config_path = get_best_config_path(key)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    category = cfg.get("category", "classic")

    if category == "RL":
        # Comprova que el model entrenat existeix
        model_path = cfg["training"]["model_path"]
        if not (os.path.exists(model_path) or os.path.exists(model_path + ".zip")):
            logger.warning(
                f"RL model not found: {model_path} — skipping '{key}'. "
                f"Run: python scripts/train_rl.py --agents {cfg['model_type']}"
            )
            return None
        from bots.rl.rl_bot import RLBot
        return RLBot(config_path=config_path)

    if category == "ML":
        from bots.ml.ml_bot import MLBot
        return MLBot(config_path=config_path)

    # Classic
    from bots.classical.hold_bot           import HoldBot
    from bots.classical.dca_bot            import DCABot
    from bots.classical.trend_bot          import TrendBot
    from bots.classical.grid_bot           import GridBot
    from bots.classical.mean_reversion_bot import MeanReversionBot
    from bots.classical.momentum_bot       import MomentumBot
    _CLASSICAL = {
        "hold": HoldBot, "dca": DCABot,
        "trend": TrendBot, "grid": GridBot,
        "mean_reversion": MeanReversionBot,
        "momentum": MomentumBot,
    }
    model_type = cfg.get("model_type", key)
    if model_type not in _CLASSICAL:
        raise ValueError(f"Unknown classic bot model_type: '{model_type}'")
    return _CLASSICAL[model_type](config_path=config_path)


# Bots per defecte: tots excepte RL
BOT_CONFIGS = BOT_REGISTRY  # alias per compatibilitat
DEFAULT_BOTS = [k for k in BOT_REGISTRY if k not in ("ppo", "sac")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare bots in walk-forward backtesting")
    parser.add_argument(
        "--bots", nargs="+", default=None,
        choices=list(BOT_REGISTRY.keys()),
        metavar="BOT",
        help=f"Bots to compare (default: all except RL). Options: {list(BOT_REGISTRY.keys())}",
    )
    parser.add_argument("--no-rl", action="store_true", help="Exclude RL agents (useful if not trained)")
    parser.add_argument("--all",   action="store_true", help="Include RL if models exist")
    args = parser.parse_args()

    # Determine list of bots
    if args.bots:
        selected_keys = args.bots
    elif args.all:
        selected_keys = list(BOT_REGISTRY.keys())
    elif args.no_rl:
        selected_keys = [k for k in BOT_REGISTRY if k not in ("ppo", "sac")]
    else:
        selected_keys = DEFAULT_BOTS  # all except RL

    logger.info(f"Selected bots: {selected_keys}")

    from core.backtesting.comparator import BotComparator

    bots = []
    for key in selected_keys:
        bot = _instantiate_bot(key)
        if bot is not None:
            bots.append(bot)

    if not bots:
        logger.error("No bots available. Check that models exist.")
        sys.exit(1)

    logger.info(f"Loaded bots ({len(bots)}): {[b.bot_id for b in bots]}")
    comparator = BotComparator(bots=bots, symbol="BTC/USDT", timeframe="1h")
    comparator.run()
