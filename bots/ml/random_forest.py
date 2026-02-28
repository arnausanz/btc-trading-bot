# bots/ml/random_forest.py
import logging
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import pickle
import os

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


class RandomForestModel:
    """
    Model Random Forest per predir direcció del preu.
    Usa TimeSeriesSplit per evitar data leakage — mai entrenarem
    amb dades futures respecte al període de validació.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Entrena el model amb TimeSeriesSplit i registra els resultats a MLflow.
        Retorna les mètriques de validació.
        """
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("random_forest")

        with mlflow.start_run():
            mlflow.log_params({
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "n_features": len(X.columns),
                "n_samples": len(X),
            })

            # TimeSeriesSplit: respecta l'ordre temporal
            tscv = TimeSeriesSplit(n_splits=5)
            accuracies, precisions, recalls = [], [], []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
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
                logger.info(f"  Fold {fold+1}: acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}")

            metrics = {
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies)),
                "precision_mean": float(np.mean(precisions)),
                "recall_mean": float(np.mean(recalls)),
            }

            mlflow.log_metrics(metrics)

            # Entrena el model final amb totes les dades
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True

            logger.info(f"Entrenament completat: accuracy={metrics['accuracy_mean']:.3f}")
            return metrics

    def predict(self, X: pd.DataFrame, threshold: float = 0.35) -> tuple[int, float]:
        """
        Retorna (predicció, confiança).
        threshold: llindar de probabilitat per predir positiu (default 0.35 per classes desequilibrades)
        """
        if not self.is_trained:
            raise RuntimeError("El model no està entrenat. Crida train() primer.")
        X_scaled = self.scaler.transform(X)
        proba_positive = self.model.predict_proba(X_scaled)[0][1]
        pred = 1 if proba_positive >= threshold else 0
        return int(pred), float(proba_positive)

    def save(self, path: str) -> None:
        """Guarda el model i el scaler a disc."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)
        logger.info(f"Model guardat a {path}")

    def load(self, path: str) -> None:
        """Carrega el model i el scaler des de disc."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.is_trained = True
        logger.info(f"Model carregat des de {path}")