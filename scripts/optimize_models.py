# scripts/optimize_models.py
import logging
import sys
import yaml
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

# Silencia soroll d'entrenament durant optimització
logging.getLogger("bots.ml").setLevel(logging.WARNING)
logging.getLogger("bots.rl").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.WARNING)
logging.getLogger("catboost").setLevel(logging.WARNING)

if __name__ == "__main__":
    from core.backtesting.ml_optimizer import MLOptimizer
    from core.backtesting.rl_optimizer import RLOptimizer

    # ── ML Models ─────────────────────────────────────────────────────────
    ML_CONFIGS = [
        "config/optimization/random_forest.yaml",
        "config/optimization/xgboost.yaml",
        "config/optimization/lightgbm.yaml",
        "config/optimization/catboost.yaml",
        "config/optimization/gru.yaml",
        "config/optimization/patchtst.yaml",
    ]

    # ── RL Agents ──────────────────────────────────────────────────────────
    RL_CONFIGS = [
        "config/optimization/sac.yaml",
        "config/optimization/ppo.yaml",
    ]

    results = []

    # Optimitza models ML
    for config_path in ML_CONFIGS:
        optimizer = MLOptimizer(config_path)
        study = optimizer.run()

        # Guarda la millor config llesta per entrenar
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

    # Optimitza agents RL
    for config_path in RL_CONFIGS:
        optimizer = RLOptimizer(config_path)
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

    # ── Resum final ────────────────────────────────────────────────────────
    logger.info("=== RESUM OPTIMITZACIÓ ===")
    logger.info(f"{'Tipus':<5} {'Model':<15} {'Mètrica':<20} {'Millor valor':>12}")
    logger.info("-" * 60)
    for r in results:
        logger.info(
            f"{r['type']:<5} {r['model']:<15} "
            f"{r['metric']:<20} {r['best_value']:>12.4f}"
        )
        logger.info(f"       Config guardada: {r['output_config']}")

    logger.info("\nPer entrenar amb els millors paràmetres:")
    logger.info("  poetry run python scripts/train_models.py  # afegeix *_optimized.yaml a CONFIGS")
    logger.info("  poetry run python scripts/train_rl.py      # afegeix *_optimized.yaml a CONFIGS")