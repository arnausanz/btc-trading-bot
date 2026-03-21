# bots/ml/catboost_model.py
from catboost import CatBoostClassifier
from bots.ml.base_tree_model import BaseTreeModel


class CatBoostModel(BaseTreeModel):
    def __init__(
        self,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.03,
        scale_pos_weight: float = 2.0,
        l2_leaf_reg: float = 3.0,
        min_data_in_leaf: int = 10,
    ):
        super().__init__()
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.l2_leaf_reg = l2_leaf_reg
        self.min_data_in_leaf = min_data_in_leaf
        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            l2_leaf_reg=l2_leaf_reg,
            min_data_in_leaf=min_data_in_leaf,
            random_seed=42,
            verbose=0,
            eval_metric="Accuracy",
        )

    @classmethod
    def from_config(cls, config: dict) -> "CatBoostModel":
        return cls(
            iterations=config["model"]["iterations"],
            depth=config["model"]["depth"],
            learning_rate=config["model"]["learning_rate"],
            scale_pos_weight=config["model"]["scale_pos_weight"],
            l2_leaf_reg=config["model"].get("l2_leaf_reg", 3.0),
            min_data_in_leaf=config["model"].get("min_data_in_leaf", 10),
        )

    def _get_mlflow_experiment(self) -> str:
        return "catboost"

    def _get_mlflow_params(self) -> dict:
        return {
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "scale_pos_weight": self.scale_pos_weight,
            "l2_leaf_reg": self.l2_leaf_reg,
            "min_data_in_leaf": self.min_data_in_leaf,
        }

    def _get_model_label(self) -> str:
        return "CB"
