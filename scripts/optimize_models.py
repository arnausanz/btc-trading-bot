# scripts/optimize_models.py
"""
Optimizes hyperparameters for ML models and RL agents using Optuna.

Each trial trains a model with a different hyperparameter combination and evaluates
it with walk-forward cross-validation (ML) or short probe runs (RL).
Outputs the best config as a ready-to-train YAML in config/training/.

Usage:
  python scripts/optimize_models.py                               # all ML + all RL
  python scripts/optimize_models.py --models rf xgb lgbm         # ML only, specific
  python scripts/optimize_models.py --agents sac ppo             # RL baseline only
  python scripts/optimize_models.py --agents ppo_onchain sac_onchain  # RL on-chain
  python scripts/optimize_models.py --models rf --trials 20      # override n_trials
  python scripts/optimize_models.py --no-rl                      # skip RL agents
  python scripts/optimize_models.py --no-ml                      # skip ML models
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

# Registry: clau curta → YAML unificat (config/models/*.yaml)
# Tots els YAMLs contenen la secció optimization amb el search space.
ALL_ML_CONFIGS = {
    "rf":       "config/models/random_forest.yaml",
    "xgb":      "config/models/xgboost.yaml",
    "lgbm":     "config/models/lightgbm.yaml",
    "catboost": "config/models/catboost.yaml",
    "gru":      "config/models/gru.yaml",
    "patchtst": "config/models/patchtst.yaml",
}

ALL_RL_CONFIGS = {
    "sac":         "config/models/sac.yaml",
    "ppo":         "config/models/ppo.yaml",
    # On-chain: tècnics + Fear&Greed + funding rate + OI + hash-rate
    "ppo_onchain": "config/models/ppo_onchain.yaml",
    "sac_onchain": "config/models/sac_onchain.yaml",
    # Professional: 12h swing + règim + on-chain + position state
    "ppo_professional": "config/models/ppo_professional.yaml",
    "sac_professional": "config/models/sac_professional.yaml",
    # C3 Advanced: TD3 variants
    "td3_professional": "config/models/td3_professional.yaml",
    "td3_multiframe":   "config/models/td3_multiframe.yaml",
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

    # Resolve which ML models to optimize.
    # Si s'especifica --agents sense --models, saltar ML automàticament.
    if args.no_ml or (args.agents and not args.models):
        ml_selection = {}
    elif args.models:
        ml_selection = {k: ALL_ML_CONFIGS[k] for k in args.models}
    else:
        ml_selection = ALL_ML_CONFIGS

    # Resolve which RL agents to optimize.
    # Si s'especifica --models sense --agents, saltar RL automàticament.
    if args.no_rl or (args.models and not args.agents):
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

        # best_params es guarden directament al YAML base (in-place)
        optimizer.save_best_config(study)

        results.append({
            "type": "ML",
            "model": optimizer.model_type,
            "metric": optimizer.metric,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "output_config": config_path,
        })

    # ── Optimize RL agents ────────────────────────────────────────────────
    for key, config_path in rl_selection.items():
        logger.info(f"=== Optimizing RL: {key.upper()} ({config_path}) ===")
        optimizer = RLOptimizer(config_path)
        if args.trials:
            optimizer.n_trials = args.trials
        study = optimizer.run()

        # best_params es guarden directament al YAML base (in-place)
        optimizer.save_best_config(study)

        results.append({
            "type": "RL",
            "model": key,
            "metric": optimizer.metric,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "output_config": config_path,
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
            logger.info(f"       best_params saved to: {r['output_config']}")

        logger.info("\nbest_params saved in-place to the base YAML (config/models/*.yaml)")
        logger.info("To train with the best parameters:")
        logger.info("  python scripts/train_models.py   # applies best_params automatically")
        logger.info("  python scripts/train_rl.py       # applies best_params automatically")