# scripts/run_comparison.py
"""
Compares all bots (classical, ML, RL) on train and test periods.

The bot registry is auto-discovered from config/models/*.yaml — no edits
needed when adding new models or bots.  Just create the YAML and the Python
class; they will appear automatically in the CLI choices.

By default, ALL comparable bots are included: classical, ML, and RL (if the
trained model file exists). Meta-strategies like 'gate' are always excluded
because they are filters/components, not standalone tradeable strategies.

Usage:
  python scripts/run_comparison.py                             # all comparable bots (classical + ML + RL if trained)
  python scripts/run_comparison.py --bots hold trend xgboost  # explicit selection
  python scripts/run_comparison.py --no-rl                    # exclude RL agents
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
_NON_COMPARABLE_KEYS: set[str] = set()   # gate, ensemble, meta-bots, etc.
for _stem, _path in BOT_REGISTRY.items():
    try:
        with open(_path) as _f:
            _cfg = yaml.safe_load(_f)
        _cat = _cfg.get("category") if _cfg else None
        if _cat == "RL":
            _RL_KEYS.add(_stem)
        elif _cat not in ("classic", "ML", None):
            # gate, ensemble, and any future meta-category are not standalone bots
            _NON_COMPARABLE_KEYS.add(_stem)
        elif _cfg and _cfg.get("comparable", True) is False:
            # Explicitly marked as non-comparable (e.g. missing historical external data)
            _NON_COMPARABLE_KEYS.add(_stem)
    except Exception:
        pass

# Default: everything comparable — classical, ML, and RL (RL skipped at instantiation
# time if the model file doesn't exist yet, so this is always safe to run).
DEFAULT_BOTS = [k for k in BOT_REGISTRY if k not in _NON_COMPARABLE_KEYS]


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

    if category not in ("classic", "ML", "RL", None):
        logger.warning(
            f"Bot '{key}' has category='{category}' which is not comparable in walk-forward "
            f"backtesting — skipping. Use the demo runner for meta-strategies like gate."
        )
        return None

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
        help=f"Bots to compare (default: all comparable). Options: {list(BOT_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--no-rl", action="store_true",
        help="Exclude RL agents (faster run, no trained model required)",
    )
    args = parser.parse_args()

    if args.bots:
        selected_keys = args.bots
    elif args.no_rl:
        selected_keys = [k for k in DEFAULT_BOTS if k not in _RL_KEYS]
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
