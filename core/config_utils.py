# core/config_utils.py
"""
Utility functions for working with unified model configs (config/models/*.yaml).

best_params workflow
--------------------
After Optuna optimization, the optimizer writes the best trial's parameters into
the base YAML as a ``best_params`` section (instead of creating a separate
``*_optimized.yaml`` file).  At training / inference time, ``apply_best_params``
reads that section and applies the overrides to the correct config sub-trees
using the same routing logic as each optimizer's ``_build_config``.

Routing rules
~~~~~~~~~~~~~
* RL / ML configs (have ``training.model``):
    - ``lookback``, ``reward_type``, ``reward_scaling``, ``stop_atr_multiplier``
      → ``training.environment[name]`` (RL only) and ``features.lookback``
    - all other params → ``training.model[name]``
* Classical bot configs (flat dict, no ``training.model``):
    - all params applied directly to the top-level config dict

discover_configs
----------------
Reads all ``config/models/*.yaml`` files and returns a dict of
``{filename_stem: yaml_path}`` for configs matching a given category
(``"ML"``, ``"RL"``, ``"classic"``, or ``None`` for all).

This allows ``train_models.py``, ``run_comparison.py``, and ``ml_bot.py``
to auto-populate their registries without any hardcoded lists.  Adding a
new model only requires creating the YAML + the Python class — no script edits.
"""
import copy
import glob
import logging
import os

import yaml

logger = logging.getLogger(__name__)

# RL environment-level params (same list as in RLOptimizer._build_config)
_RL_ENV_PARAMS = frozenset({
    "lookback", "reward_type", "reward_scaling", "stop_atr_multiplier",
})


def apply_best_params(config: dict) -> dict:
    """
    Apply ``config['best_params']`` overrides to the appropriate config sections.

    Returns a *new* deep-copied config dict with the overrides applied.
    The ``best_params`` section itself is preserved in the returned dict so it
    can be re-applied after subsequent saves.

    If ``best_params`` is absent or empty the original dict is returned unchanged
    (no copy).
    """
    if not config.get("best_params"):
        return config

    config = copy.deepcopy(config)
    params: dict = config["best_params"]

    # ── RL / ML structured config ─────────────────────────────────────────────
    training = config.get("training")
    if isinstance(training, dict) and "model" in training:
        for name, value in params.items():
            if name in _RL_ENV_PARAMS:
                # Route to environment sub-tree (RL only)
                env = training.get("environment")
                if isinstance(env, dict):
                    env[name] = value
                # Also synchronise features.lookback
                if name == "lookback":
                    features = config.get("features")
                    if isinstance(features, dict):
                        features["lookback"] = value
            else:
                training["model"][name] = value

    # ── Classical bot: flat config ────────────────────────────────────────────
    else:
        for name, value in params.items():
            config[name] = value

    logger.debug("Applied best_params: %s", params)
    return config


def discover_configs(
    category: str | None = None,
    base_dir: str = "config/models",
) -> dict[str, str]:
    """
    Returns ``{filename_stem: yaml_path}`` for every config matching *category*.

    Args:
        category: ``"ML"``, ``"RL"``, ``"classic"``, or ``None`` (all).
        base_dir: Directory to scan.  Defaults to ``"config/models"``.

    Returns:
        Ordered dict sorted by filename stem.

    Example::

        from core.config_utils import discover_configs

        ml_configs = discover_configs("ML")
        # → {"catboost": "config/models/catboost.yaml",
        #    "gru": "config/models/gru.yaml", ...}

        all_configs = discover_configs()
        # → all 20 YAML files, keyed by stem
    """
    result: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(base_dir, "*.yaml"))):
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            if cfg and (category is None or cfg.get("category") == category):
                stem = os.path.splitext(os.path.basename(path))[0]
                result[stem] = path
        except Exception as exc:
            logger.warning("discover_configs: could not read %s — %s", path, exc)
    return result
