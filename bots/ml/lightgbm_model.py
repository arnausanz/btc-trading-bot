# bots/ml/lightgbm_model.py
from lightgbm import LGBMClassifier
from bots.ml.base_tree_model import BaseTreeModel


class LightGBMModel(BaseTreeModel):
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        scale_pos_weight: float = 2.0,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.scale_pos_weight = scale_pos_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            scale_pos_weight=scale_pos_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

    @classmethod
    def from_config(cls, config: dict) -> "LightGBMModel":
        return cls(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
            learning_rate=config["model"]["learning_rate"],
            num_leaves=config["model"].get("num_leaves", 31),
            scale_pos_weight=config["model"]["scale_pos_weight"],
            min_child_samples=config["model"].get("min_child_samples", 20),
            subsample=config["model"].get("subsample", 0.8),
            colsample_bytree=config["model"].get("colsample_bytree", 0.8),
            reg_alpha=config["model"].get("reg_alpha", 0.1),
            reg_lambda=config["model"].get("reg_lambda", 1.0),
        )

    def _get_mlflow_experiment(self) -> str:
        return "lightgbm"

    def _get_mlflow_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "scale_pos_weight": self.scale_pos_weight,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
        }

    def _get_model_label(self) -> str:
        return "LGB"
