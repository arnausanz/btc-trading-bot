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
"""
import copy
import logging

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
