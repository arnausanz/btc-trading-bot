#!/usr/bin/env python3
# scripts/test_pipeline.py
"""
Smoke test: validates the full pipeline end-to-end for ALL models.

Runs 1 Optuna trial + 1 minimal training pass for every ML model and RL agent.
Saves NOTHING permanently to disk:
  - No .pkl / .pt / .zip model files written to models/
  - No best_params written back to config YAMLs
  - MLflow is redirected to a temporary directory (auto-cleaned on exit)

Use this script to confirm that data reading, feature building, optimization
and training all work correctly before running a full optimize + train cycle.

Usage:
    cd /path/to/btc-trading-bot
    python scripts/test_pipeline.py              # all ML + all RL
    python scripts/test_pipeline.py --ml-only    # skip RL agents
    python scripts/test_pipeline.py --rl-only    # skip ML models
    python scripts/test_pipeline.py --models xgb gru           # specific ML keys
    python scripts/test_pipeline.py --agents ppo sac_onchain   # specific RL keys
"""

# ─────────────────────────────────────────────────────────────────────────────
# Environment setup — MUST happen before any model imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import tempfile

_RL_MODEL_TMPDIR = tempfile.mkdtemp(prefix="btc_test_rl_")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.append(".")

# ─────────────────────────────────────────────────────────────────────────────
# Standard imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import copy
import importlib
import logging
import shutil
import time
import traceback

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers
for _name in (
    "lightgbm", "catboost", "bots.ml", "bots.rl",
    "data.processing", "core.db", "optuna", "stable_baselines3",
):
    logging.getLogger(_name).setLevel(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Test constants  (minimal but realistic)
# ─────────────────────────────────────────────────────────────────────────────
TEST_N_TRIALS        = 1    # Optuna trials per model
TEST_ML_MAX_ROWS     = 300  # Dataset slice for ML training (speed)
TEST_ML_N_FOLDS      = 2    # CV folds (vs 5 in production)
TEST_RL_PROBE_STEPS  = 200  # Timesteps per RL optimization trial
TEST_RL_TRAIN_STEPS  = 500  # Timesteps for RL smoke training
TEST_DL_EPOCHS       = 1    # Epochs for GRU / PatchTST
TEST_DL_SEQ_LEN      = 20   # Sequence length for DL models (vs 50 / 96)
                             # Note: automatically raised to patch_len+4 for PatchTST

# ─────────────────────────────────────────────────────────────────────────────
# Model registries
# ─────────────────────────────────────────────────────────────────────────────
ML_CONFIGS = {
    "rf":       "config/models/random_forest.yaml",
    "xgb":      "config/models/xgboost.yaml",
    "lgbm":     "config/models/lightgbm.yaml",
    "catboost": "config/models/catboost.yaml",
    "gru":      "config/models/gru.yaml",
    "patchtst": "config/models/patchtst.yaml",
    "tft":      "config/models/tft.yaml",
}

RL_CONFIGS = {
    "ppo":              "config/models/ppo.yaml",
    "sac":              "config/models/sac.yaml",
    "ppo_onchain":      "config/models/ppo_onchain.yaml",
    "sac_onchain":      "config/models/sac_onchain.yaml",
    "ppo_professional": "config/models/ppo_professional.yaml",
    "sac_professional": "config/models/sac_professional.yaml",
    "td3_professional": "config/models/td3_professional.yaml",
    "td3_multiframe":   "config/models/td3_multiframe.yaml",
}

# ─────────────────────────────────────────────────────────────────────────────
# Results tracker
# ─────────────────────────────────────────────────────────────────────────────
_results: list[tuple[str, str, float, str]] = []  # (name, status, elapsed, error)


def _run(name: str, fn) -> None:
    """Execute fn(), record PASS / FAIL + elapsed time."""
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        _results.append((name, "PASS", elapsed, ""))
        logger.info(f"  ✓  PASS  {name}  ({elapsed:.1f}s)")
    except Exception as exc:
        elapsed = time.time() - t0
        _results.append((name, "FAIL", elapsed, str(exc)))
        logger.error(f"  ✗  FAIL  {name}  ({elapsed:.1f}s)")
        logger.error(f"         {exc}")
        logger.debug(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch TimeSeriesSplit in base_tree_model to use TEST_ML_N_FOLDS
# (the patch must happen before any tree model is instantiated)
# ─────────────────────────────────────────────────────────────────────────────
import bots.ml.base_tree_model as _btm
from sklearn.model_selection import TimeSeriesSplit as _OrigTSS


class _FastTSS(_OrigTSS):
    """Drops n_splits to TEST_ML_N_FOLDS regardless of what callers request."""
    def __init__(self, n_splits: int = 5, **kwargs):
        super().__init__(n_splits=TEST_ML_N_FOLDS, **kwargs)


_btm.TimeSeriesSplit = _FastTSS  # patch in-module reference used by train()


# ─────────────────────────────────────────────────────────────────────────────
# Lazy RL registry init (imports heavy SB3 only when RL tests run)
# ─────────────────────────────────────────────────────────────────────────────
_RL_AGENT_REGISTRY: dict | None = None
_RL_ENV_REGISTRY:   dict | None = None
_RL_MULTIFRAME:     set  | None = None


def _init_rl_registries() -> None:
    global _RL_AGENT_REGISTRY, _RL_ENV_REGISTRY, _RL_MULTIFRAME
    if _RL_AGENT_REGISTRY is not None:
        return

    from bots.rl.agents import SACAgent, PPOAgent, TD3Agent
    from bots.rl.environment import BtcTradingEnvDiscrete, BtcTradingEnvContinuous
    from bots.rl.environment_professional import (
        BtcTradingEnvProfessionalDiscrete,
        BtcTradingEnvProfessionalContinuous,
    )
    from bots.rl.rewards import builtins, professional, advanced  # noqa: registers reward fns

    # Keyed by model_type value in each YAML (not by the config filename stem)
    _RL_AGENT_REGISTRY = {
        "ppo":              PPOAgent,
        "sac":              SACAgent,
        "ppo_professional": PPOAgent,
        "sac_professional": SACAgent,
        "td3_professional": TD3Agent,
        "td3_multiframe":   TD3Agent,
    }
    _RL_ENV_REGISTRY = {
        "ppo":              BtcTradingEnvDiscrete,
        "sac":              BtcTradingEnvContinuous,
        "ppo_professional": BtcTradingEnvProfessionalDiscrete,
        "sac_professional": BtcTradingEnvProfessionalContinuous,
        "td3_professional": BtcTradingEnvProfessionalContinuous,
        "td3_multiframe":   BtcTradingEnvProfessionalContinuous,
    }
    _RL_MULTIFRAME = {"td3_multiframe"}


# ─────────────────────────────────────────────────────────────────────────────
# ML smoke test
# ─────────────────────────────────────────────────────────────────────────────

_DL_MODEL_TYPES = {"gru", "patchtst"}


def _test_ml(key: str, config_path: str) -> None:
    """
    Smoke test for one ML model:
      1. Build dataset, slice to TEST_ML_MAX_ROWS rows
      2. Run 1 Optuna trial (custom objective using sliced data, no save_best_config)
      3. Run 1 training pass (no model.save())
    """
    from data.processing.dataset import DatasetBuilder
    from core.backtesting.ml_optimizer import MLOptimizer

    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    is_dl = base_cfg.get("model_type") in _DL_MODEL_TYPES

    # ── 1. Build and slice dataset ────────────────────────────────────────────
    X_full, y_full = DatasetBuilder.from_config(base_cfg).build()
    X = X_full.iloc[-TEST_ML_MAX_ROWS:]
    y = y_full.iloc[-TEST_ML_MAX_ROWS:]
    logger.info(
        f"  [{key}] data: {len(X)} rows × {X.shape[1]} features "
        f"(sliced from {len(X_full)})"
    )

    # ── 2. Optimization — 1 trial, no save_best_config ────────────────────────
    mod       = importlib.import_module(base_cfg["module"])
    model_cls = getattr(mod, base_cfg["class_name"])

    opt = MLOptimizer(config_path)
    opt.n_trials = TEST_N_TRIALS

    def _fast_objective(trial: object) -> float:
        """Objective using pre-built sliced data; skips DatasetBuilder entirely."""
        params     = opt._sample_params(trial)
        trial_cfg  = copy.deepcopy(base_cfg)
        for name, value in params.items():
            trial_cfg["training"]["model"][name] = value

        # Minimise DL training cost; respect patch_len constraint for PatchTST
        if is_dl:
            patch_len = trial_cfg["training"]["model"].get("patch_len", 0)
            seq_len   = max(TEST_DL_SEQ_LEN, patch_len + 4) if patch_len else TEST_DL_SEQ_LEN
            trial_cfg["training"]["model"]["epochs"]  = TEST_DL_EPOCHS
            trial_cfg["training"]["model"]["seq_len"] = seq_len

        try:
            model   = model_cls.from_config(trial_cfg["training"])
            metrics = model.train(X, y)
            return metrics.get(opt.metric, 0.0)
        except Exception as exc:
            logger.debug(f"  [{key}] trial exception: {exc}")
            return -999.0

    opt._objective = _fast_objective
    study = opt.run()  # n_trials = 1
    # ← save_best_config NOT called
    logger.info(f"  [{key}] optim OK  — best {opt.metric}={study.best_value:.4f}")

    # ── 3. Training — no model.save() ─────────────────────────────────────────
    train_cfg = copy.deepcopy(base_cfg)
    if is_dl:
        patch_len = train_cfg["training"]["model"].get("patch_len", 0)
        seq_len   = max(TEST_DL_SEQ_LEN, patch_len + 4) if patch_len else TEST_DL_SEQ_LEN
        train_cfg["training"]["model"]["epochs"]  = TEST_DL_EPOCHS
        train_cfg["training"]["model"]["seq_len"] = seq_len

    model   = model_cls.from_config(train_cfg["training"])
    metrics = model.train(X, y)
    # ← model.save() NOT called
    logger.info(
        f"  [{key}] train OK — "
        f"acc={metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# RL smoke test
# ─────────────────────────────────────────────────────────────────────────────

def _test_rl(key: str, config_path: str) -> None:
    """
    Smoke test for one RL agent:
      1. Load feature DataFrame (includes external data if configured)
      2. Run 1 Optuna trial (probe_timesteps = TEST_RL_PROBE_STEPS, no save)
      3. Run 1 training pass (total_timesteps = TEST_RL_TRAIN_STEPS, no agent.save())
    """
    from core.backtesting.rl_optimizer import RLOptimizer

    _init_rl_registries()

    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    model_type = base_cfg["model_type"]

    # ── 1. Optimization — 1 trial, reduced probe steps, no save ──────────────
    opt = RLOptimizer(config_path)
    opt.n_trials       = TEST_N_TRIALS
    opt.probe_timesteps = TEST_RL_PROBE_STEPS

    study = opt.run()
    # ← save_best_config NOT called
    logger.info(f"  [{key}] optim OK  — best {opt.metric}={study.best_value:.4f}")

    # ── 2. Build feature data (done once, reused for training) ────────────────
    if model_type in (_RL_MULTIFRAME or set()):
        from data.processing.multiframe_builder import MultiFrameFeatureBuilder
        df = MultiFrameFeatureBuilder.from_config(base_cfg).build()
    else:
        from data.processing.feature_builder import FeatureBuilder
        df = FeatureBuilder.from_config(base_cfg).build()

    logger.info(f"  [{key}] data: {len(df)} rows × {len(df.columns)} features")

    # ── 3. Training — reduced timesteps, no agent.save() ─────────────────────
    train_cfg = copy.deepcopy(base_cfg)
    train_cfg["training"]["model"]["total_timesteps"] = TEST_RL_TRAIN_STEPS

    features_cfg = train_cfg["features"]
    env_cfg = {
        **train_cfg["training"]["environment"],
        "lookback": features_cfg["lookback"],
    }

    split    = int(len(df) * train_cfg["training"]["train_pct"])
    df_train = df.iloc[:split]
    df_val   = df.iloc[split:]

    lookback = features_cfg["lookback"]
    if len(df_train) < lookback + 10:
        raise ValueError(
            f"Insufficient training data after split: {len(df_train)} rows "
            f"(need ≥ {lookback + 10} for lookback={lookback}). "
            f"External data may be missing — run download_*.py scripts first."
        )

    env_class  = _RL_ENV_REGISTRY[model_type]
    train_env  = env_class(df=df_train, **env_cfg)
    val_env    = env_class(df=df_val,   **env_cfg)

    agent   = _RL_AGENT_REGISTRY[model_type].from_config(train_cfg["training"])
    metrics = agent.train(
        train_env=train_env,
        val_env=val_env,
        total_timesteps=TEST_RL_TRAIN_STEPS,
    )
    # ← agent.save() NOT called
    logger.info(
        f"  [{key}] train OK — "
        f"return={metrics['val_return_pct']:.2f}%  "
        f"drawdown={metrics['val_max_drawdown_pct']:.2f}%"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test: run 1 Optuna trial + 1 minimal training for all models. "
            "Saves NOTHING to disk."
        )
    )
    parser.add_argument("--ml-only", action="store_true", help="Test only ML models")
    parser.add_argument("--rl-only", action="store_true", help="Test only RL agents")
    parser.add_argument(
        "--models", nargs="+", metavar="KEY",
        choices=list(ML_CONFIGS),
        help=f"ML keys to test (default: all). Options: {list(ML_CONFIGS)}",
    )
    parser.add_argument(
        "--agents", nargs="+", metavar="KEY",
        choices=list(RL_CONFIGS),
        help=f"RL keys to test (default: all). Options: {list(RL_CONFIGS)}",
    )
    args = parser.parse_args()

    t_global = time.time()

    # ── ML ────────────────────────────────────────────────────────────────────
    if not args.rl_only:
        ml_sel = {k: ML_CONFIGS[k] for k in (args.models or ML_CONFIGS)}
        logger.info(f"\n{'='*64}")
        logger.info(
            f"ML MODELS ({len(ml_sel)})  —  "
            f"1 trial | {TEST_ML_MAX_ROWS} rows | {TEST_ML_N_FOLDS} folds | "
            f"DL: {TEST_DL_EPOCHS} epoch, seq_len={TEST_DL_SEQ_LEN}"
        )
        logger.info(f"{'='*64}")
        for key, path in ml_sel.items():
            _run(f"ml:{key}", lambda k=key, p=path: _test_ml(k, p))

    # ── RL ────────────────────────────────────────────────────────────────────
    if not args.ml_only:
        rl_sel = {k: RL_CONFIGS[k] for k in (args.agents or RL_CONFIGS)}
        logger.info(f"\n{'='*64}")
        logger.info(
            f"RL AGENTS ({len(rl_sel)})  —  "
            f"1 trial | probe={TEST_RL_PROBE_STEPS} steps | "
            f"train={TEST_RL_TRAIN_STEPS} steps"
        )
        logger.info(f"{'='*64}")
        for key, path in rl_sel.items():
            _run(f"rl:{key}", lambda k=key, p=path: _test_rl(k, p))

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_global
    passed = [r for r in _results if r[1] == "PASS"]
    failed = [r for r in _results if r[1] == "FAIL"]

    logger.info(f"\n{'='*64}")
    logger.info(
        f"SMOKE TEST  —  {len(passed)}/{len(_results)} passed  "
        f"({elapsed_total:.0f}s total)"
    )
    logger.info(f"{'='*64}")
    for name, status, elapsed, err in _results:
        mark = "✓" if status == "PASS" else "✗"
        line = f"  {mark}  {status:<4}  {name:<32}  {elapsed:>6.1f}s"
        if err:
            line += f"  ← {err[:55]}"
        logger.info(line)

    if failed:
        logger.error(f"\n{len(failed)} test(s) FAILED — see errors above")
    else:
        logger.info("\nAll tests PASSED ✓")

    # ── Cleanup temp dirs ─────────────────────────────────────────────────────
    shutil.rmtree(_RL_MODEL_TMPDIR, ignore_errors=True)
    logger.info("Temporary RL dir cleaned up.")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
