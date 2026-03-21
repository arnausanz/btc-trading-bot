# bots/ml/random_forest.py
from sklearn.ensemble import RandomForestClassifier
from bots.ml.base_tree_model import BaseTreeModel


class RandomForestModel(BaseTreeModel):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

    @classmethod
    def from_config(cls, config: dict) -> "RandomForestModel":
        return cls(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
            min_samples_leaf=config["model"].get("min_samples_leaf", 5),
            max_features=config["model"].get("max_features", "sqrt"),
        )

    def _get_model_label(self) -> str:
        return "RF"
