# core/backtesting/ml_optimizer.py
import copy
import logging
import mlflow
import optuna
import yaml
from core.interfaces.base_ml_model import BaseMLModel
from data.processing.dataset import DatasetBuilder

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


class MLOptimizer:
    """
    Optimizes hyperparameters of ML models (supervised) with Optuna.
    Each trial = a complete training with Walk-Forward Cross-Validation.
    Objective metric: accuracy_mean | precision_mean | recall_mean (configurable).

    Adding a new model to registry = automatically optimizable.
    """

    def __init__(self, optimization_config_path: str):
        with open(optimization_config_path) as f:
            self.opt_cfg = yaml.safe_load(f)

        with open(self.opt_cfg["base_config"]) as f:
            self.base_train_cfg = yaml.safe_load(f)

        self.model_type = self.opt_cfg["model_type"]
        self.n_trials = self.opt_cfg["n_trials"]
        self.metric = self.opt_cfg.get("metric", "accuracy_mean")
        self.direction = self.opt_cfg.get("direction", "maximize")
        self.search_space = self.opt_cfg["search_space"]

    def _sample_params(self, trial: optuna.Trial) -> dict:
        """Samples hyperparameters from the search space defined in YAML."""
        params = {}
        for name, spec in self.search_space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"],
                    log=spec.get("log", False)
                )
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
        return params

    def _build_config(self, params: dict) -> dict:
        """Applies trial parameters to the base config."""
        config = copy.deepcopy(self.base_train_cfg)
        for name, value in params.items():
            config["model"][name] = value
        return config

    def _objective(self, trial: optuna.Trial) -> float:
        from bots.ml.random_forest import RandomForestModel
        from bots.ml.xgboost_model import XGBoostModel
        from bots.ml.lightgbm_model import LightGBMModel
        from bots.ml.catboost_model import CatBoostModel
        from bots.ml.gru_model import GRUModel
        from bots.ml.patchtst_model import PatchTSTModel

        _REGISTRY: dict[str, type[BaseMLModel]] = {
            "random_forest": RandomForestModel,
            "xgboost": XGBoostModel,
            "lightgbm": LightGBMModel,
            "catboost": CatBoostModel,
            "gru": GRUModel,
            "patchtst": PatchTSTModel,
        }

        params = self._sample_params(trial)
        config = self._build_config(params)

        try:
            mlflow.end_run()
            builder = DatasetBuilder.from_config(config)
            X, y = builder.build()

            model = _REGISTRY[self.model_type].from_config(config)
            metrics = model.train(X, y)

            score = metrics.get(self.metric, 0.0)
            logger.info(
                f"  Trial {trial.number:03d} | "
                f"{self.metric}={score:.4f} | "
                f"params={params}"
            )
            return score

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.warning(f"  Trial {trial.number} failed: {e}")
            return -999.0
        finally:
            mlflow.end_run()

    def run(self) -> optuna.Study:
        logger.info(
            f"=== Optimizing {self.model_type} | "
            f"{self.n_trials} trials | "
            f"objective: {self.metric} ==="
        )
        study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        study.optimize(self._objective, n_trials=self.n_trials)

        logger.info(f"  Best {self.metric}: {study.best_value:.4f}")
        logger.info(f"  Best parameters: {study.best_params}")
        return study

    def best_config(self, study: optuna.Study) -> dict:
        """Returns the complete config with the best parameters."""
        return self._build_config(study.best_params)

    def save_best_config(self, study: optuna.Study, output_path: str) -> None:
        """Saves the best config as YAML ready for training."""
        best = self.best_config(study)
        best["experiment_name"] = f"{self.model_type}_optimized"
        best["output"]["model_path"] = best["output"]["model_path"].replace(
            ".pkl", "_optimized.pkl"
        ).replace(".pt", "_optimized.pt")
        with open(output_path, "w") as f:
            yaml.dump(best, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"  Optimized config saved to {output_path}")