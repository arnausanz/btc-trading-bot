# scripts/train_models.py
"""
Trains supervised ML models.

Usage:
  python scripts/train_models.py                       # train all
  python scripts/train_models.py --models rf xgb       # only RF and XGB
  python scripts/train_models.py --models gru patchtst # only DL models
"""
import os
import argparse
import logging
import sys
import yaml
import mlflow
from tqdm import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.append(".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Key → config file map (recommended training order: fast first)
ALL_CONFIGS = {
    "rf":       "config/training/rf_experiment_1.yaml",
    "xgb":      "config/training/xgb_experiment_1.yaml",
    "lgbm":     "config/training/lgbm_experiment_1.yaml",
    "catboost": "config/training/catboost_experiment_1.yaml",
    "gru":      "config/training/gru_experiment_1.yaml",
    "patchtst": "config/training/patchtst_experiment_1.yaml",
}


def get_best_config_path(model_name: str) -> str:
    """
    Get the best available config path for a model.

    Checks for {model_name}_optimized.yaml first, then falls back to
    {model_name}_experiment_1.yaml. Logs which config was selected.

    Args:
        model_name: Model key (e.g., 'rf', 'xgb', 'gru')

    Returns:
        Path to the config file to use
    """
    optimized = f"config/training/{model_name}_optimized.yaml"
    default = f"config/training/{model_name}_experiment_1.yaml"

    if os.path.exists(optimized):
        logger.info(f"Found optimized config for {model_name}: {optimized}")
        return optimized

    logger.info(f"Using default config for {model_name}: {default}")
    return default

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train supervised ML models")
    parser.add_argument(
        "--models", nargs="+", default=list(ALL_CONFIGS.keys()),
        choices=list(ALL_CONFIGS.keys()),
        metavar="MODEL",
        help=f"Models to train (default: all). Options: {list(ALL_CONFIGS.keys())}",
    )
    args = parser.parse_args()

    from data.processing.dataset import DatasetBuilder
    from core.interfaces.base_ml_model import BaseMLModel
    from bots.ml.random_forest   import RandomForestModel
    from bots.ml.xgboost_model   import XGBoostModel
    from bots.ml.lightgbm_model  import LightGBMModel
    from bots.ml.catboost_model  import CatBoostModel
    from bots.ml.gru_model       import GRUModel
    from bots.ml.patchtst_model  import PatchTSTModel

    _MODEL_REGISTRY: dict[str, type[BaseMLModel]] = {
        "random_forest": RandomForestModel,
        "xgboost":       XGBoostModel,
        "lightgbm":      LightGBMModel,
        "catboost":      CatBoostModel,
        "gru":           GRUModel,
        "patchtst":      PatchTSTModel,
    }

    selected_models = args.models
    selected_configs = [get_best_config_path(m) for m in selected_models]
    logger.info(f"Selected models: {selected_models}")

    for _noisy in ("data.processing.technical", "data.observation.builder", "core.db", "mlflow", "bots.ml"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    results = []

    with tqdm(selected_configs, desc="Training models", unit="model", dynamic_ncols=True) as model_bar:
        for config_path in model_bar:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            name       = config["experiment_name"]
            model_type = config["model_type"]
            model_bar.set_description(f"Training {name}")
            mlflow.end_run()

            tqdm.write(f"\n── {name} ──")
            builder = DatasetBuilder.from_config(config)
            X, y    = builder.build()
            tqdm.write(f"  Dataset: {X.shape[0]} rows x {X.shape[1]} features")

            if model_type not in _MODEL_REGISTRY:
                raise ValueError(f"Unknown model: {model_type}")

            model   = _MODEL_REGISTRY[model_type].from_config(config)
            metrics = model.train(X, y)
            model.save(config["output"]["model_path"])
            results.append({"name": name, "accuracy": metrics["accuracy_mean"],
                "precision": metrics["precision_mean"], "recall": metrics["recall_mean"]})

    logger.info("=== FINAL COMPARISON ===")
    logger.info(f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
    logger.info("-" * 65)
    for r in results:
        logger.info(f"{r['name']:<30} {r['accuracy']:>10.3f} {r['precision']:>10.3f} {r['recall']:>10.3f}")