# bots/ml/random_forest.py
import logging
import pickle
import numpy as np
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from core.config import MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)


class RandomForestModel:
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_names: list[str] = []
        self.is_trained = False
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.scaler = StandardScaler()

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        self.feature_names = list(X.columns)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("random_forest")

        with mlflow.start_run():
            mlflow.log_params({
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "n_features": len(X.columns),
                "n_samples": len(X),
            })

            tscv = TimeSeriesSplit(n_splits=5)
            accuracies, precisions, recalls = [], [], []

            with tqdm(
                enumerate(tscv.split(X)),
                total=5,
                desc="  RandomForest folds",
                unit="fold",
                dynamic_ncols=True,
                colour="green",
            ) as pbar:
                for fold, (train_idx, val_idx) in pbar:
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    X_train_scaled = self.scaler.fit_transform(X_train)
                    X_val_scaled = self.scaler.transform(X_val)

                    self.model.fit(X_train_scaled, y_train)
                    y_pred = self.model.predict(X_val_scaled)

                    acc = accuracy_score(y_val, y_pred)
                    prec = precision_score(y_val, y_pred, zero_division=0)
                    rec = recall_score(y_val, y_pred, zero_division=0)

                    accuracies.append(acc)
                    precisions.append(prec)
                    recalls.append(rec)
                    pbar.set_postfix(
                        acc=f"{acc:.3f}",
                        prec=f"{prec:.3f}",
                        rec=f"{rec:.3f}",
                        best_acc=f"{max(accuracies):.3f}",
                    )

            metrics = {
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies)),
                "precision_mean": float(np.mean(precisions)),
                "recall_mean": float(np.mean(recalls)),
            }
            mlflow.log_metrics(metrics)

            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True

            tqdm.write(f"  ✓ RandomForest → acc={metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}")
            return metrics

    def predict(self, X: pd.DataFrame, threshold: float = 0.35) -> tuple[int, float]:
        if not self.is_trained:
            raise RuntimeError("El model no està entrenat. Crida train() primer.")
        X_scaled = self.scaler.transform(X)
        proba_positive = self.model.predict_proba(X_scaled)[0][1]
        return int(proba_positive >= threshold), float(proba_positive)

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