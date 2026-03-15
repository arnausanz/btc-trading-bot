# bots/ml/base_tree_model.py
"""
Shared training loop for tree-based classifiers (XGB, LGB, CB, RF).

All four models share the same:
  - 5-fold TimeSeriesSplit cross-validation loop
  - StandardScaler normalisation
  - MLflow experiment logging
  - save() / load() contract (pickle with model + scaler + feature_names)
  - predict() with probability threshold

Subclasses only need to implement:
  - __init__(): build self.model with model-specific hyperparameters
  - from_config(): YAML constructor
  - _get_mlflow_experiment(): experiment name string
  - _get_mlflow_params(): dict of hyperparameters to log
  - _get_model_label(): short display label ("XGB", "LGB", "CB", "RF")
"""
import logging
import pickle
from abc import abstractmethod

import numpy as np
import pandas as pd
import mlflow
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from core.config import MLFLOW_TRACKING_URI
from core.interfaces.base_ml_model import BaseMLModel

logger = logging.getLogger(__name__)


class BaseTreeModel(BaseMLModel):
    """
    Abstract base class for tree-based ML classifiers.

    Provides the full train / predict / save / load implementation.
    Subclasses only define model construction and MLflow metadata.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.is_trained = False
        # self.model must be set by the subclass __init__ before calling train()

    # ── Abstract hooks ─────────────────────────────────────────────────────────

    @abstractmethod
    def _get_mlflow_experiment(self) -> str:
        """MLflow experiment name (e.g. 'xgboost', 'lightgbm')."""
        ...

    @abstractmethod
    def _get_mlflow_params(self) -> dict:
        """Model-specific hyperparameters to log in MLflow (excluding n_features/n_samples)."""
        ...

    @abstractmethod
    def _get_model_label(self) -> str:
        """Short display label for tqdm output (e.g. 'XGB', 'LGB', 'CB', 'RF')."""
        ...

    # ── Shared implementation ──────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        self.feature_names = list(X.columns)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self._get_mlflow_experiment())

        with mlflow.start_run():
            params = self._get_mlflow_params()
            params.update({"n_features": len(X.columns), "n_samples": len(X)})
            mlflow.log_params(params)

            tscv = TimeSeriesSplit(n_splits=5)
            accuracies, precisions, recalls = [], [], []
            label = self._get_model_label()

            with tqdm(enumerate(tscv.split(X)), total=5,
                      desc=f"  {label:<3} folds", unit="fold", leave=False,
                      dynamic_ncols=True) as fold_bar:
                for fold, (train_idx, val_idx) in fold_bar:
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    X_train_scaled = pd.DataFrame(
                        self.scaler.fit_transform(X_train), columns=self.feature_names
                    )
                    X_val_scaled = pd.DataFrame(
                        self.scaler.transform(X_val), columns=self.feature_names
                    )

                    self.model.fit(X_train_scaled, y_train)
                    y_pred = self.model.predict(X_val_scaled)

                    acc = accuracy_score(y_val, y_pred)
                    prec = precision_score(y_val, y_pred, zero_division=0)
                    rec = recall_score(y_val, y_pred, zero_division=0)
                    accuracies.append(acc)
                    precisions.append(prec)
                    recalls.append(rec)
                    fold_bar.set_postfix(
                        {"acc": f"{acc:.3f}", "prec": f"{prec:.3f}", "rec": f"{rec:.3f}"}
                    )

            metrics = {
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies)),
                "precision_mean": float(np.mean(precisions)),
                "recall_mean": float(np.mean(recalls)),
            }
            mlflow.log_metrics(metrics)

            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X), columns=self.feature_names
            )
            self.model.fit(X_scaled, y)
            self.is_trained = True

            tqdm.write(
                f"  ✓ {label:<5}acc={metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}"
            )
            return metrics

    def predict(self, X: pd.DataFrame, threshold: float = 0.35) -> tuple[int, float]:
        if not self.is_trained:
            raise RuntimeError("El model no està entrenat. Crida train() primer.")
        X_scaled = pd.DataFrame(
            self.scaler.transform(X), columns=self.feature_names
        )
        proba = self.model.predict_proba(X_scaled)[0][1]
        return int(proba >= threshold), float(proba)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
            }, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.is_trained = True
