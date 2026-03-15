# bots/ml/random_forest.py
from sklearn.ensemble import RandomForestClassifier
from bots.ml.base_tree_model import BaseTreeModel


class RandomForestModel(BaseTreeModel):
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

    @classmethod
    def from_config(cls, config: dict) -> "RandomForestModel":
        return cls(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
        )

    def _get_mlflow_experiment(self) -> str:
        return "random_forest"

    def _get_mlflow_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
        }

    def _get_model_label(self) -> str:
        return "RF"
