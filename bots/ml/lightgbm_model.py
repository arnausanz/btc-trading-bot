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
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.scale_pos_weight = scale_pos_weight
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            scale_pos_weight=scale_pos_weight,
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
        }

    def _get_model_label(self) -> str:
        return "LGB"
