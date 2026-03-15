# scripts/run_comparison.py
"""
Compares all bots (classical, ML, RL) on train and test periods.

The bot registry is auto-discovered from config/models/*.yaml — no edits
needed when adding new models or bots.  Just create the YAML and the Python
class; they will appear automatically in the CLI choices.

Usage:
  python scripts/run_comparison.py                             # all non-RL bots
  python scripts/run_comparison.py --bots hold trend xgboost  # selection
  python scripts/run_comparison.py --bots ppo sac             # RL agents
  python scripts/run_comparison.py --no-rl                    # explicit no-RL
  python scripts/run_comparison.py --all                      # everything (skips untrainied RL)
"""
import argparse
import importlib
import logging
import os
import sys

import yaml

sys.path.append(".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Auto-discover ALL bot configs ─────────────────────────────────────────────
from core.config_utils import discover_configs

BOT_REGISTRY: dict[str, str] = discover_configs()          # {stem: yaml_path}
BOT_CONFIGS = BOT_REGISTRY                                  # alias for compatibility

# Derive RL keys from the YAML category field
_RL_KEYS: set[str] = set()
for _stem, _path in BOT_REGISTRY.items():
    try:
        with open(_path) as _f:
            _cfg = yaml.safe_load(_f)
        if _cfg and _cfg.get("category") == "RL":
            _RL_KEYS.add(_stem)
    except Exception:
        pass

DEFAULT_BOTS = [k for k in BOT_REGISTRY if k not in _RL_KEYS]


def _instantiate_bot(key: str):
    """
    Instantiates the bot for *key* by reading its YAML category.

    - RL: RLBot (skipped if model file not found)
    - ML: MLBot
    - classic: dynamic import via module + class_name fields in the YAML
    """
    config_path = BOT_REGISTRY[key]
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    category = cfg.get("category", "classic")

    if category == "RL":
        model_path = cfg["training"]["model_path"]
        if not (os.path.exists(model_path) or os.path.exists(model_path + ".zip")):
            agent_key = cfg["model_type"]
            logger.warning(
                f"RL model not found: {model_path} — skipping '{key}'. "
                f"Run: python scripts/train_rl.py --agents {agent_key}"
            )
            return None
        from bots.rl.rl_bot import RLBot
        return RLBot(config_path=config_path)

    if category == "ML":
        from bots.ml.ml_bot import MLBot
        return MLBot(config_path=config_path)

    # classic — dynamic class loading from module + class_name in YAML
    mod_path = cfg.get("module")
    cls_name = cfg.get("class_name")
    if not (mod_path and cls_name):
        raise ValueError(
            f"Classic bot '{key}' is missing 'module' or 'class_name' in {config_path}. "
            f"Add them to enable auto-discovery."
        )
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    return cls(config_path=config_path)


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
