# scripts/run_comparison.py
"""
Compares all bots (classical, ML, RL) on train and test periods.

Usage:
  python scripts/run_comparison.py                          # all available bots
  python scripts/run_comparison.py --bots hold trend dca    # selection
  python scripts/run_comparison.py --bots ml_rf ml_gru rl_ppo
  python scripts/run_comparison.py --no-rl                  # all except RL (if not trained)

Available bots:
  Classical: hold, dca, trend, grid
  ML       : ml_rf, ml_xgb, ml_lgbm, ml_catboost, ml_gru, ml_patchtst
  RL       : rl_ppo, rl_sac (require running train_rl.py)
"""
import argparse
import logging
import os
import sys

sys.path.append(".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def get_best_bot_config_path(bot_name: str) -> str:
    """
    Check if an optimized config exists for the given bot.
    If yes, use it; otherwise fall back to default config.

    Args:
        bot_name: The bot name (e.g., 'dca', 'trend', 'hold', 'grid')

    Returns:
        Path to the optimized config if it exists, otherwise default config path
    """
    optimized = f"config/bots/{bot_name}_optimized.yaml"
    default = f"config/bots/{bot_name}.yaml"

    if os.path.exists(optimized):
        logger.info(f"Using optimized bot config: {optimized}")
        return optimized

    logger.info(f"Using default bot config: {default}")
    return default


# ── Registry of all bots ────────────────────────────────────────────────
BOT_CONFIGS = {
    # Classical - use helper function for config path resolution
    "hold":        {"type": "classical", "config": get_best_bot_config_path("hold")},
    "dca":         {"type": "classical", "config": get_best_bot_config_path("dca")},
    "trend":       {"type": "classical", "config": get_best_bot_config_path("trend")},
    "grid":        {"type": "classical", "config": get_best_bot_config_path("grid")},
    # Supervised ML
    "ml_rf":       {"type": "ml",        "config": "config/bots/ml_bot.yaml"},
    "ml_xgb":      {"type": "ml",        "config": "config/bots/ml_bot_xgb.yaml"},
    "ml_lgbm":     {"type": "ml",        "config": "config/bots/ml_bot_lgbm.yaml"},
    "ml_catboost": {"type": "ml",        "config": "config/bots/ml_bot_catboost.yaml"},
    "ml_gru":      {"type": "ml",        "config": "config/bots/ml_bot_gru.yaml"},
    "ml_patchtst": {"type": "ml",        "config": "config/bots/ml_bot_patchtst.yaml"},
    # RL
    "rl_ppo":      {"type": "rl",        "config": "config/bots/rl_bot_ppo.yaml", "model": "models/ppo_btc_v1.zip"},
    "rl_sac":      {"type": "rl",        "config": "config/bots/rl_bot_sac.yaml", "model": "models/sac_btc_v1.zip"},
}

DEFAULT_BOTS = [k for k in BOT_CONFIGS if BOT_CONFIGS[k]["type"] != "rl"]  # all except RL by default


def _instantiate_bot(key: str):
    """Instantiate the corresponding bot. Returns None if model does not exist (RL not trained)."""
    cfg = BOT_CONFIGS[key]

    # Check if RL model exists
    if cfg["type"] == "rl":
        model_path = cfg.get("model", "")
        # stable-baselines3 can save as .zip or as folder
        if not (os.path.exists(model_path) or os.path.exists(model_path + ".zip")):
            logger.warning(f"RL model not found: {model_path} — skipping {key}. Run train_rl.py first.")
            return None
        from bots.rl.rl_bot import RLBot
        return RLBot(config_path=cfg["config"])

    if cfg["type"] == "ml":
        from bots.ml.ml_bot import MLBot
        return MLBot(config_path=cfg["config"])

    # Classical
    from bots.classical.hold_bot  import HoldBot
    from bots.classical.dca_bot   import DCABot
    from bots.classical.trend_bot import TrendBot
    from bots.classical.grid_bot  import GridBot
    _CLASSICAL = {"hold": HoldBot, "dca": DCABot, "trend": TrendBot, "grid": GridBot}
    return _CLASSICAL[key](config_path=cfg["config"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare bots in walk-forward backtesting")
    parser.add_argument(
        "--bots", nargs="+", default=None,
        choices=list(BOT_CONFIGS.keys()),
        metavar="BOT",
        help=f"Bots to compare (default: all except RL). Options: {list(BOT_CONFIGS.keys())}",
    )
    parser.add_argument("--no-rl",  action="store_true", help="Exclude RL agents (useful if not trained)")
    parser.add_argument("--all",    action="store_true", help="Include RL if models exist")
    args = parser.parse_args()

    # Determine list of bots
    if args.bots:
        selected_keys = args.bots
    elif args.all:
        selected_keys = list(BOT_CONFIGS.keys())
    elif args.no_rl:
        selected_keys = [k for k in BOT_CONFIGS if BOT_CONFIGS[k]["type"] != "rl"]
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
