# scripts/train_models.py
"""
Trains supervised ML models.

The list of available models is auto-discovered from config/models/*.yaml
(category: ML) — no edits needed when adding new models.

Usage:
  python scripts/train_models.py                           # train all ML models
  python scripts/train_models.py --models xgboost gru      # only these two
  python scripts/train_models.py --models patchtst catboost
"""
import argparse
import importlib
import logging
import os
import sys

import mlflow
import yaml
from tqdm import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.append(".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Auto-discover all ML model configs ────────────────────────────────────────
from core.config_utils import apply_best_params, discover_configs

ALL_CONFIGS: dict[str, str] = discover_configs("ML")
# e.g. {"catboost": "config/models/catboost.yaml",
#        "gru":      "config/models/gru.yaml", ...}


def _load_model_class(config: dict):
    """Dynamically imports and returns the model class declared in the YAML."""
    mod_path = config.get("module")
    cls_name = config.get("class_name")
    if not (mod_path and cls_name):
        raise ValueError(
            f"YAML for model_type '{config.get('model_type')}' is missing "
            f"'module' or 'class_name' fields."
        )
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


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

    selected_configs = [ALL_CONFIGS[m] for m in args.models]
    logger.info(f"Selected models: {args.models}")

    for _noisy in ("data.processing.technical", "data.observation.builder", "core.db", "mlflow", "bots.ml"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    results = []

    with tqdm(selected_configs, desc="Training models", unit="model", dynamic_ncols=True) as model_bar:
        for config_path in model_bar:
            with open(config_path) as f:
                config = apply_best_params(yaml.safe_load(f))

            name       = config["training"]["experiment_name"]
            model_type = config["model_type"]
            model_bar.set_description(f"Training {name}")
            mlflow.end_run()

            tqdm.write(f"\n── {name} ──")
            builder = DatasetBuilder.from_config(config)
            X, y = builder.build()
            tqdm.write(f"  Dataset: {X.shape[0]} rows x {X.shape[1]} features")

            model_cls = _load_model_class(config)
            model = model_cls.from_config(config["training"])
            metrics = model.train(X, y)
            model.save(config["training"]["model_path"])
            results.append({
                "name": name,
                "accuracy": metrics["accuracy_mean"],
                "precision": metrics["precision_mean"],
                "recall": metrics["recall_mean"],
            })

    logger.info("=== FINAL COMPARISON ===")
    logger.info(f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
    logger.info("-" * 65)
    for r in results:
        logger.info(
            f"{r['name']:<30} {r['accuracy']:>10.3f} {r['precision']:>10.3f} {r['recall']:>10.3f}"
        )
