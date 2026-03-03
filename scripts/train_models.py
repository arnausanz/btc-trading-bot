# scripts/train_models.py
import os
import logging
import sys
import yaml
import mlflow

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
    from bots.ml.random_forest import RandomForestModel
    from bots.ml.xgboost_model import XGBoostModel
    from bots.ml.lightgbm_model import LightGBMModel
    from bots.ml.catboost_model import CatBoostModel
    from bots.ml.gru_model import GRUModel

    _MODEL_BUILDERS = {
        "random_forest": lambda cfg: RandomForestModel(
            n_estimators=cfg["model"]["n_estimators"],
            max_depth=cfg["model"]["max_depth"],
        ),
        "xgboost": lambda cfg: XGBoostModel(
            n_estimators=cfg["model"]["n_estimators"],
            max_depth=cfg["model"]["max_depth"],
            learning_rate=cfg["model"]["learning_rate"],
            scale_pos_weight=cfg["model"]["scale_pos_weight"],
        ),
        "lightgbm": lambda cfg: LightGBMModel(
            n_estimators=cfg["model"]["n_estimators"],
            max_depth=cfg["model"]["max_depth"],
            learning_rate=cfg["model"]["learning_rate"],
            num_leaves=cfg["model"].get("num_leaves", 31),
            scale_pos_weight=cfg["model"]["scale_pos_weight"],
        ),
        "catboost": lambda cfg: CatBoostModel(
            iterations=cfg["model"]["iterations"],
            depth=cfg["model"]["depth"],
            learning_rate=cfg["model"]["learning_rate"],
            scale_pos_weight=cfg["model"]["scale_pos_weight"],
        ),
        "gru": lambda cfg: GRUModel(
            seq_len=cfg["model"]["seq_len"],
            hidden_size=cfg["model"]["hidden_size"],
            num_layers=cfg["model"]["num_layers"],
            dropout=cfg["model"]["dropout"],
            epochs=cfg["model"]["epochs"],
            batch_size=cfg["model"]["batch_size"],
            learning_rate=cfg["model"]["learning_rate"],
            patience=cfg["model"].get("patience", 4),
        ),
    }

    CONFIGS = [
        "config/training/rf_experiment_1.yaml",
        "config/training/xgb_experiment_1.yaml",
        "config/training/lgbm_experiment_1.yaml",
        "config/training/catboost_experiment_1.yaml",
        "config/training/gru_experiment_1.yaml",
    ]

    results = []

    for config_path in CONFIGS:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        name = config["experiment_name"]
        model_type = config["model_type"]

        logger.info(f"=== {name} ===")
        mlflow.end_run()

        builder = DatasetBuilder.from_config(config)
        X, y = builder.build()
        logger.info(f"  Dataset: {X.shape[0]} files x {X.shape[1]} features")

        if model_type not in _MODEL_BUILDERS:
            raise ValueError(f"Model desconegut: {model_type}")

        model = _MODEL_BUILDERS[model_type](config)
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