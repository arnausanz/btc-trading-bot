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

# Clau → YAML unificat (config/models/*.yaml)
# Ordre recomanat: ràpids primer
ALL_CONFIGS = {
    "rf":       "config/models/random_forest.yaml",
    "xgb":      "config/models/xgboost.yaml",
    "lgbm":     "config/models/lightgbm.yaml",
    "catboost": "config/models/catboost.yaml",
    "gru":      "config/models/gru.yaml",
    "patchtst": "config/models/patchtst.yaml",
}

# Mapeig clau → model_type per al registry
_KEY_TO_MODEL_TYPE = {
    "rf": "random_forest",
    "xgb": "xgboost",
    "lgbm": "lightgbm",
    "catboost": "catboost",
    "gru": "gru",
    "patchtst": "patchtst",
}


def get_best_config_path(model_key: str) -> str:
    """
    Retorna el millor config disponible per a un model.

    Cerca primer {model_type}_optimized.yaml (generat per Optuna),
    si no existeix usa el YAML base (config/models/{model_key}.yaml).

    Args:
        model_key: Clau del model (e.g., 'rf', 'xgb', 'gru')

    Returns:
        Path al fitxer de config a usar
    """
    model_type = _KEY_TO_MODEL_TYPE.get(model_key, model_key)
    optimized = f"config/models/{model_type}_optimized.yaml"
    default = ALL_CONFIGS[model_key]

    if os.path.exists(optimized):
        logger.info(f"Found optimized config for {model_key}: {optimized}")
        return optimized

    logger.info(f"Using default config for {model_key}: {default}")
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

            # Llegim des de la nova estructura unificada
            name       = config["training"]["experiment_name"]
            model_type = config["model_type"]
            model_bar.set_description(f"Training {name}")
            mlflow.end_run()

            tqdm.write(f"\n── {name} ──")
            # DatasetBuilder.from_config llegeix des del top-level (symbol, timeframes, ...)
            builder = DatasetBuilder.from_config(config)
            X, y    = builder.build()
            tqdm.write(f"  Dataset: {X.shape[0]} rows x {X.shape[1]} features")

            if model_type not in _MODEL_REGISTRY:
                raise ValueError(f"Unknown model: {model_type}")

            # El model llegeix config["model"] → passem config["training"]
            model   = _MODEL_REGISTRY[model_type].from_config(config["training"])
            metrics = model.train(X, y)
            model.save(config["training"]["model_path"])
            results.append({"name": name, "accuracy": metrics["accuracy_mean"],
                "precision": metrics["precision_mean"], "recall": metrics["recall_mean"]})

    logger.info("=== FINAL COMPARISON ===")
    logger.info(f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
    logger.info("-" * 65)
    for r in results:
        logger.info(f"{r['name']:<30} {r['accuracy']:>10.3f} {r['precision']:>10.3f} {r['recall']:>10.3f}")