# bots/ml/xgboost_model.py
from xgboost import XGBClassifier
from bots.ml.base_tree_model import BaseTreeModel


class XGBoostModel(BaseTreeModel):
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        scale_pos_weight: float = 2.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        )

    @classmethod
    def from_config(cls, config: dict) -> "XGBoostModel":
        return cls(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
            learning_rate=config["model"]["learning_rate"],
            scale_pos_weight=config["model"]["scale_pos_weight"],
            subsample=config["model"].get("subsample", 0.8),
            colsample_bytree=config["model"].get("colsample_bytree", 0.8),
            min_child_weight=config["model"].get("min_child_weight", 5),
            reg_alpha=config["model"].get("reg_alpha", 0.1),
            reg_lambda=config["model"].get("reg_lambda", 1.0),
        )

    def _get_mlflow_experiment(self) -> str:
        return "xgboost"

    def _get_mlflow_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "scale_pos_weight": self.scale_pos_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
        }

    def _get_model_label(self) -> str:
        return "XGB"
