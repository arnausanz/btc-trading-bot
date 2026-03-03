# bots/ml/lightgbm_model.py
import logging
import pickle
import numpy as np
import pandas as pd
import mlflow
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from core.config import MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)


class LightGBMModel:
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        scale_pos_weight: float = 2.0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.scale_pos_weight = scale_pos_weight
        self.feature_names: list[str] = []
        self.is_trained = False
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
        self.scaler = StandardScaler()

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        self.feature_names = list(X.columns)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("lightgbm")

        with mlflow.start_run():
            mlflow.log_params({
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "scale_pos_weight": self.scale_pos_weight,
                "n_features": len(X.columns),
                "n_samples": len(X),
            })

            tscv = TimeSeriesSplit(n_splits=5)
            accuracies, precisions, recalls = [], [], []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
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
                logger.info(f"  [{fold+1}/5] acc={acc:.3f} prec={prec:.3f} rec={rec:.3f}")

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

            logger.info(f"  ✓ acc={metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}")
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
            pickle.dump({"model": self.model, "scaler": self.scaler, "feature_names": self.feature_names}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.is_trained = True