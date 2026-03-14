# scripts/run_comparison.py
"""
Compares all bots (classical, ML, RL) on train and test periods.

Llegeix configs unificades de config/models/*.yaml.
El camp 'category' del YAML determina el tipus de bot (classic/ML/RL).

Usage:
  python scripts/run_comparison.py                                           # all available bots
  python scripts/run_comparison.py --bots hold trend dca                     # selection
  python scripts/run_comparison.py --bots rf xgb ppo
  python scripts/run_comparison.py --bots ppo_professional sac_professional  # professional RL
  python scripts/run_comparison.py --bots td3_professional td3_multiframe    # C3 TD3
  python scripts/run_comparison.py --no-rl                                   # all except RL

Available bots:
  Classic     : hold, dca, trend, grid, mean_reversion, momentum
  ML          : rf, xgb, lgbm, catboost, gru, patchtst
  RL baseline : ppo, sac (require: python scripts/train_rl.py)
  RL on-chain : ppo_onchain, sac_onchain (require external DB data)
  RL pro      : ppo_professional, sac_professional
  RL C3 TD3   : td3_professional, td3_multiframe
                (require: python scripts/train_rl.py --agents td3_professional td3_multiframe)
"""
import argparse
import logging
import os
import sys
import yaml

sys.path.append(".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# Registry: key -> base YAML path
BOT_REGISTRY = {
    # Classic
    "hold":           "config/models/hold.yaml",
    "dca":            "config/models/dca.yaml",
    "trend":          "config/models/trend.yaml",
    "grid":           "config/models/grid.yaml",
    "mean_reversion": "config/models/mean_reversion.yaml",
    "momentum":       "config/models/momentum.yaml",
    # ML supervised
    "rf":       "config/models/random_forest.yaml",
    "xgb":      "config/models/xgboost.yaml",
    "lgbm":     "config/models/lightgbm.yaml",
    "catboost": "config/models/catboost.yaml",
    "gru":      "config/models/gru.yaml",
    "patchtst": "config/models/patchtst.yaml",
    # RL baseline
    "ppo": "config/models/ppo.yaml",
    "sac": "config/models/sac.yaml",
    # RL on-chain
    "ppo_onchain": "config/models/ppo_onchain.yaml",
    "sac_onchain": "config/models/sac_onchain.yaml",
    # RL professional (12H swing + regime + on-chain + position state)
    # Training: python scripts/train_rl.py --agents ppo_professional sac_professional
    "ppo_professional": "config/models/ppo_professional.yaml",
    "sac_professional": "config/models/sac_professional.yaml",
    # C3: TD3 advanced (TD3 + sentiment / multi-timeframe)
    # Training: python scripts/train_rl.py --agents td3_professional td3_multiframe
    "td3_professional": "config/models/td3_professional.yaml",
    "td3_multiframe":   "config/models/td3_multiframe.yaml",
}


def get_best_config_path(bot_key: str) -> str:
    """
    Retorna la ruta al YAML base del bot.

    Els best_params d'Optuna es guarden directament dins el YAML base
    (secció ``best_params``).  Cada bot els aplica via ``apply_best_params``
    — no cal cap fitxer *_optimized.yaml separat.
    """
    path = BOT_REGISTRY[bot_key]
    logger.info(f"Config for {bot_key}: {path}")
    return path


def _instantiate_bot(key: str):
    """
    Instantiates the bot by reading the 'category' field from the YAML.
    Returns None if the RL model has not been trained yet.
    """
    config_path = get_best_config_path(key)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    category = cfg.get("category", "classic")

    if category == "RL":
        model_path = cfg["training"]["model_path"]
        if not (os.path.exists(model_path) or os.path.exists(model_path + ".zip")):
            agent_key = cfg["model_type"]
            logger.warning(
                f"RL model not found: {model_path} -- skipping '{key}'. "
                f"Run: python scripts/train_rl.py --agents {agent_key}"
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


# RL keys excluded from default "train all" run
_RL_KEYS = {
    "ppo", "sac", "ppo_onchain", "sac_onchain",
    "ppo_professional", "sac_professional",
    "td3_professional", "td3_multiframe",
}
BOT_CONFIGS = BOT_REGISTRY  # alias for compatibility
DEFAULT_BOTS = [k for k in BOT_REGISTRY if k not in _RL_KEYS]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare bots in walk-forward backtesting")
    parser.add_argument(
        "--bots", nargs="+", default=None,
        choices=list(BOT_REGISTRY.keys()),
        metavar="BOT",
        help=f"Bots to compare (default: all except RL). Options: {list(BOT_REGISTRY.keys())}",
    )
    parser.add_argument("--no-rl", action="store_true", help="Exclude RL agents")
    parser.add_argument("--all",   action="store_true", help="Include RL if models exist")
    args = parser.parse_args()

    if args.bots:
        selected_keys = args.bots
    elif args.all:
        selected_keys = list(BOT_REGISTRY.keys())
    elif args.no_rl:
        selected_keys = [k for k in BOT_REGISTRY if k not in _RL_KEYS]
    else:
        selected_keys = DEFAULT_BOTS

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
