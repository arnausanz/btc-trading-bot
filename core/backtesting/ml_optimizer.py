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

    Llegeix des del YAML unificat (config/models/*.yaml) — un sol fitxer conté
    tota la informació: dades, training, optimization search space i bot config.

    Adding a new model to registry = automatically optimizable.
    """

    def __init__(self, config_path: str):
        """
        Args:
            config_path: ruta al YAML unificat (config/models/{model}.yaml)
        """
        self.config_path = config_path
        with open(config_path) as f:
            self.unified_cfg = yaml.safe_load(f)

        opt = self.unified_cfg["optimization"]
        self.model_type = self.unified_cfg["model_type"]
        self.n_trials = opt["n_trials"]
        self.metric = opt.get("metric", "accuracy_mean")
        self.direction = opt.get("direction", "maximize")
        self.search_space = opt["search_space"]

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
        """Aplica paràmetres del trial al config unificat."""
        config = copy.deepcopy(self.unified_cfg)
        for name, value in params.items():
            config["training"]["model"][name] = value
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
            # DatasetBuilder.from_config llegeix des del top-level del config unificat
            builder = DatasetBuilder.from_config(config)
            X, y = builder.build()

            # El model llegeix config["model"] → passem config["training"]
            model = _REGISTRY[self.model_type].from_config(config["training"])
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

    def save_best_config(self, study: optuna.Study, config_path: str | None = None) -> None:
        """
        Guarda best_params dins el YAML base (in-place).

        En lloc de crear un fitxer *_optimized.yaml separat, escriu la secció
        ``best_params`` directament al YAML base.  Al moment d'entrenar,
        ``apply_best_params`` aplica aquests valors automàticament.

        Args:
            config_path: ruta del YAML on escriure.  Per defecte: el YAML base
                         passat al constructor (self.config_path).
        """
        target = config_path or self.config_path
        with open(target) as f:
            config = yaml.safe_load(f)

        config["best_params"] = study.best_params

        with open(target, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(
            f"  best_params saved to {target}: {study.best_params}"
        )