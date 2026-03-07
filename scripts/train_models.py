# scripts/train_models.py
import os
import logging
import sys
import yaml
import mlflow
from tqdm import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.append(".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    from data.processing.dataset import DatasetBuilder
    from core.interfaces.base_ml_model import BaseMLModel
    from bots.ml.random_forest import RandomForestModel
    from bots.ml.xgboost_model import XGBoostModel
    from bots.ml.lightgbm_model import LightGBMModel
    from bots.ml.catboost_model import CatBoostModel
    from bots.ml.gru_model import GRUModel
    from bots.ml.patchtst_model import PatchTSTModel

    # Afegir un model nou = una línia aquí
    _MODEL_REGISTRY: dict[str, type[BaseMLModel]] = {
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel,
        "lightgbm": LightGBMModel,
        "catboost": CatBoostModel,
        "gru": GRUModel,
        "patchtst": PatchTSTModel,
    }

    CONFIGS = [
        "config/training/rf_experiment_1.yaml",
        "config/training/xgb_experiment_1.yaml",
        "config/training/lgbm_experiment_1.yaml",
        "config/training/catboost_experiment_1.yaml",
        "config/training/gru_experiment_1.yaml",
        "config/training/patchtst_experiment_1.yaml",
    ]

    # Suprimim loggers sorollosos — tqdm mostra el progrés
    for _noisy in ("data.processing.technical", "data.observation.builder",
                   "core.db", "mlflow", "bots.ml"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    results = []

    with tqdm(CONFIGS, desc="Entrenant models", unit="model", dynamic_ncols=True) as model_bar:
        for config_path in model_bar:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            name = config["experiment_name"]
            model_type = config["model_type"]
            model_bar.set_description(f"Entrenant {name}")

            mlflow.end_run()

            tqdm.write(f"\n── {name} ──")
            builder = DatasetBuilder.from_config(config)
            X, y = builder.build()
            tqdm.write(f"  Dataset: {X.shape[0]} files x {X.shape[1]} features")

            if model_type not in _MODEL_REGISTRY:
                raise ValueError(f"Model desconegut: {model_type}")

            model = _MODEL_REGISTRY[model_type].from_config(config)
            metrics = model.train(X, y)
            model.save(config["output"]["model_path"])

            results.append({
                "name": name,
                "accuracy": metrics["accuracy_mean"],
                "precision": metrics["precision_mean"],
                "recall": metrics["recall_mean"],
            })

    logger.info("=== COMPARATIVA FINAL ===")
    logger.info(f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
    logger.info("-" * 65)
    for r in results:
        logger.info(
            f"{r['name']:<30} {r['accuracy']:>10.3f} "
            f"{r['precision']:>10.3f} {r['recall']:>10.3f}"
        )