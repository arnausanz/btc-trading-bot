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
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
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
        )

    def _get_mlflow_experiment(self) -> str:
        return "xgboost"

    def _get_mlflow_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "scale_pos_weight": self.scale_pos_weight,
        }

    def _get_model_label(self) -> str:
        return "XGB"
