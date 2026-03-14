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


def get_best_config_path(model_key: str) -> str:
    """
    Retorna la ruta al YAML base del model.

    Els best_params d'Optuna es guarden directament dins el YAML base
    (secció ``best_params``).  ``apply_best_params`` els aplica en temps
    d'entrenament — no cal cap fitxer *_optimized.yaml separat.
    """
    path = ALL_CONFIGS[model_key]
    logger.info(f"Config for {model_key}: {path}")
    return path

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
            from core.config_utils import apply_best_params
            with open(config_path) as f:
                config = apply_best_params(yaml.safe_load(f))

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