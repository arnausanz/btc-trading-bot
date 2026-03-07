# scripts/optimize_models.py
"""
Optimizes hyperparameters for ML models and RL agents using Optuna.

Each trial trains a model with a different hyperparameter combination and evaluates
it with walk-forward cross-validation (ML) or short probe runs (RL).
Outputs the best config as a ready-to-train YAML in config/training/.

Usage:
  python scripts/optimize_models.py                          # all ML + all RL
  python scripts/optimize_models.py --models rf xgb lgbm    # ML only, specific
  python scripts/optimize_models.py --agents sac ppo        # RL only
  python scripts/optimize_models.py --models rf --trials 20 # override n_trials
  python scripts/optimize_models.py --no-rl                 # skip RL agents
  python scripts/optimize_models.py --no-ml                 # skip ML models
"""
import argparse
import logging
import sys
import os

sys.path.append(".")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy training logs during optimization
logging.getLogger("bots.ml").setLevel(logging.WARNING)
logging.getLogger("bots.rl").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.WARNING)
logging.getLogger("catboost").setLevel(logging.WARNING)

# Registry: short key → optimization config path
ALL_ML_CONFIGS = {
    "rf":       "config/optimization/random_forest.yaml",
    "xgb":      "config/optimization/xgboost.yaml",
    "lgbm":     "config/optimization/lightgbm.yaml",
    "catboost": "config/optimization/catboost.yaml",
    "gru":      "config/optimization/gru.yaml",
    "patchtst": "config/optimization/patchtst.yaml",
}

ALL_RL_CONFIGS = {
    "sac": "config/optimization/sac.yaml",
    "ppo": "config/optimization/ppo.yaml",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize ML model and RL agent hyperparameters with Optuna"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=list(ALL_ML_CONFIGS.keys()),
        metavar="MODEL",
        help=f"ML models to optimize (default: all). Options: {list(ALL_ML_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--agents", nargs="+", default=None,
        choices=list(ALL_RL_CONFIGS.keys()),
        metavar="AGENT",
        help=f"RL agents to optimize (default: all). Options: {list(ALL_RL_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--trials", type=int, default=None,
        help="Override n_trials for all selected optimizers (default: use value from config YAML)",
    )
    parser.add_argument(
        "--no-ml", action="store_true",
        help="Skip ML model optimization entirely",
    )
    parser.add_argument(
        "--no-rl", action="store_true",
        help="Skip RL agent optimization entirely",
    )
    args = parser.parse_args()

    from core.backtesting.ml_optimizer import MLOptimizer
    from core.backtesting.rl_optimizer import RLOptimizer

    # Resolve which ML models to optimize
    if args.no_ml:
        ml_selection = {}
    elif args.models:
        ml_selection = {k: ALL_ML_CONFIGS[k] for k in args.models}
    else:
        ml_selection = ALL_ML_CONFIGS

    # Resolve which RL agents to optimize
    if args.no_rl:
        rl_selection = {}
    elif args.agents:
        rl_selection = {k: ALL_RL_CONFIGS[k] for k in args.agents}
    else:
        rl_selection = ALL_RL_CONFIGS

    logger.info(f"ML models selected: {list(ml_selection.keys()) or 'none'}")
    logger.info(f"RL agents selected:  {list(rl_selection.keys()) or 'none'}")
    if args.trials:
        logger.info(f"n_trials override:   {args.trials}")

    results = []

    # ── Optimize ML models ────────────────────────────────────────────────
    for key, config_path in ml_selection.items():
        logger.info(f"=== Optimizing ML: {key.upper()} ({config_path}) ===")
        optimizer = MLOptimizer(config_path)
        if args.trials:
            optimizer.n_trials = args.trials
        study = optimizer.run()

        model_type = optimizer.model_type
        out_path = f"config/training/{model_type}_optimized.yaml"
        optimizer.save_best_config(study, out_path)

        results.append({
            "type": "ML",
            "model": model_type,
            "metric": optimizer.metric,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "output_config": out_path,
        })

    # ── Optimize RL agents ────────────────────────────────────────────────
    for key, config_path in rl_selection.items():
        logger.info(f"=== Optimizing RL: {key.upper()} ({config_path}) ===")
        optimizer = RLOptimizer(config_path)
        if args.trials:
            optimizer.n_trials = args.trials
        study = optimizer.run()

        model_type = optimizer.model_type
        out_path = f"config/training/{model_type}_optimized.yaml"
        optimizer.save_best_config(study, out_path)

        results.append({
            "type": "RL",
            "model": model_type,
            "metric": optimizer.metric,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "output_config": out_path,
        })

    # ── Final summary ─────────────────────────────────────────────────────
    if not results:
        logger.info("No optimizers ran — check --models / --agents / --no-ml / --no-rl flags.")
    else:
        logger.info("=== OPTIMIZATION SUMMARY ===")
        logger.info(f"{'Type':<5} {'Model':<15} {'Metric':<20} {'Best value':>12}")
        logger.info("-" * 60)
        for r in results:
            logger.info(
                f"{r['type']:<5} {r['model']:<15} "
                f"{r['metric']:<20} {r['best_value']:>12.4f}"
            )
            logger.info(f"       Config saved to: {r['output_config']}")

        logger.info("\nTo train with the best parameters:")
        logger.info("  python scripts/train_models.py   # uses *_optimized.yaml if present")
        logger.info("  python scripts/train_rl.py       # uses *_optimized.yaml if present")
