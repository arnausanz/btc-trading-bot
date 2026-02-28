# scripts/train_models.py
import logging
import sys
import yaml

sys.path.append(".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)

from data.processing.dataset import DatasetBuilder
from bots.ml.random_forest import RandomForestModel
from bots.ml.xgboost_model import XGBoostModel

CONFIGS = [
    "config/training/rf_experiment_1.yaml",
    "config/training/xgb_experiment_1.yaml",
]

def build_model(config: dict):
    model_type = config["model_type"]
    if model_type == "random_forest":
        return RandomForestModel(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
        )
    elif model_type == "xgboost":
        return XGBoostModel(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
            learning_rate=config["model"]["learning_rate"],
            scale_pos_weight=config["model"]["scale_pos_weight"],
        )
    else:
        raise ValueError(f"Model desconegut: {model_type}")


if __name__ == "__main__":
    results = []

    for config_path in CONFIGS:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        logger.info(f"=== Experiment: {config['experiment_name']} ===")

        builder = DatasetBuilder.from_config(config)
        X, y = builder.build()

        model = build_model(config)
        metrics = model.train(X, y)
        model.save(config["output"]["model_path"])

        results.append({
            "name": config["experiment_name"],
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