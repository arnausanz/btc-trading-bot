# bots/gate/regime_models/xgb_classifier.py
"""
XGBoost Classifier — Predicció de règim en temps real (Porta 1, Fase 2)

Entrena un XGBoost multiclasse (multi:softprob) sobre les 14 features de P1
amb les etiquetes generades per l'HMM (Viterbi) com a target.

Walk-forward validation obligatòria (≥5 folds, train ≥2 anys, test 3-6 mesos).
Optuna TPE + Hyperband per optimitzar la mlogloss MITJANA across folds.

Guarda: model final + best_params a models/gate_xgb_regime.pkl
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

# Noms de les 14 features de P1
P1_FEATURES = [
    "ema_ratio_50_200",
    "ema_slope_50",
    "ema_slope_200",
    "adx_14",
    "atr_14_percentile",
    "funding_rate_3d",
    "funding_rate_7d",
    "volume_sma_ratio",
    "volume_trend",
    "fear_greed",
    "fear_greed_7d_change",
    "rsi_14_daily",
    "returns_5d",
    "returns_20d",
]


class XGBRegimeClassifier:
    """
    XGBoost multiclasse per predicció de règim.
    Segueix el mateix patró que BaseTreeModel del projecte però adaptat
    per a multi-classe i walk-forward específic de règims.
    """

    def __init__(self, n_folds: int = 5, n_optuna_trials: int = 50, random_state: int = 42):
        self.n_folds        = n_folds
        self.n_trials       = n_optuna_trials
        self.random_state   = random_state
        self.model: xgb.XGBClassifier | None = None
        self.label_encoder  = LabelEncoder()
        self.best_params: dict = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,          # string labels ('STRONG_BULL', 'WEAK_BULL', ...)
        n_classes: int,
    ) -> dict:
        """
        Walk-forward training + Optuna. Retorna dict amb mètriques de validació.

        Args:
            X: DataFrame amb les 14 features de P1 (alineat amb y)
            y: array de string labels de règim (de l'HMM Viterbi)
            n_classes: nombre de classes (K de l'HMM)
        """
        # Encode string labels → integers
        y_enc = self.label_encoder.fit_transform(y)
        logger.info(f"Classes: {list(self.label_encoder.classes_)} ({n_classes})")

        # ── Definir folds walk-forward ────────────────────────────────────
        folds = self._walk_forward_splits(len(X), self.n_folds)
        logger.info(f"Walk-forward: {len(folds)} folds")

        # ── Optuna: minimitzar mlogloss MITJANA ───────────────────────────
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=self.random_state),
            pruner=HyperbandPruner(),
        )
        study.optimize(
            lambda trial: self._objective(trial, X, y_enc, folds, n_classes),
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        self.best_params = study.best_params
        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best mlogloss (mean WF): {study.best_value:.4f}")

        # ── Validació final amb best_params ──────────────────────────────
        metrics = self._validate(X, y_enc, folds, n_classes, self.best_params)
        logger.info(f"Validation — accuracy per fold: {[f'{a:.3f}' for a in metrics['accuracies']]}")
        logger.info(f"Validation — mean accuracy OOS: {metrics['mean_accuracy']:.3f}")
        logger.info(f"Validation — mlogloss std: {metrics['mlogloss_std']:.4f}")

        # ── Re-entrenar sobre tot el dataset amb best_params ─────────────
        self._fit_final(X, y_enc, n_classes, self.best_params)

        return metrics

    def predict_proba(self, X: pd.DataFrame) -> dict[str, float]:
        """
        Prediu probabilitats per a cada règim.
        Retorna {regime_name: probability}.
        """
        if self.model is None:
            raise RuntimeError("Model no entrenat. Crida fit() primer.")
        proba = self.model.predict_proba(X)[-1:]  # última fila (observació actual)
        classes = self.label_encoder.classes_
        return {cls: float(p) for cls, p in zip(classes, proba[0])}

    # ──────────────────────────────────────────────────────────────────────
    # Walk-forward splits
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _walk_forward_splits(
        n: int,
        n_folds: int,
        min_train_pct: float = 0.5,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Genera índexos de folds walk-forward.
        Cada fold: train = [0..split_i], test = [split_i..split_{i+1}]
        Mínim: 50% del dataset com a train en el primer fold.
        """
        test_size  = int(n * (1 - min_train_pct) / n_folds)
        min_train  = int(n * min_train_pct)
        folds: list[tuple[np.ndarray, np.ndarray]] = []

        for i in range(n_folds):
            test_start = min_train + i * test_size
            test_end   = test_start + test_size
            if test_end > n:
                break
            train_idx = np.arange(0, test_start)
            test_idx  = np.arange(test_start, test_end)
            folds.append((train_idx, test_idx))

        return folds

    # ──────────────────────────────────────────────────────────────────────
    # Optuna objective
    # ──────────────────────────────────────────────────────────────────────

    def _objective(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: np.ndarray,
        folds: list,
        n_classes: int,
    ) -> float:
        """Objectiu Optuna: mlogloss MITJANA dels folds walk-forward."""
        params = {
            "max_depth":          trial.suggest_int("max_depth", 3, 7),
            "min_child_weight":   trial.suggest_int("min_child_weight", 5, 50),
            "gamma":              trial.suggest_float("gamma", 0.1, 5.0),
            "subsample":          trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "reg_alpha":          trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1.0, 10.0),
            "learning_rate":      trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        }
        metrics = self._validate(X, y, folds, n_classes, params)
        return metrics["mean_mlogloss"]

    # ──────────────────────────────────────────────────────────────────────
    # Validació walk-forward
    # ──────────────────────────────────────────────────────────────────────

    def _validate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        folds: list,
        n_classes: int,
        params: dict,
    ) -> dict:
        """Executa la validació walk-forward amb els params donats."""
        mlogloss_list: list[float] = []
        accuracy_list: list[float] = []

        for train_idx, test_idx in folds:
            X_train = X.iloc[train_idx]
            y_train = y[train_idx]
            X_test  = X.iloc[test_idx]
            y_test  = y[test_idx]

            model = xgb.XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=n_classes,
                n_estimators=1000,
                early_stopping_rounds=50,
                use_label_encoder=False,
                random_state=self.random_state,
                verbosity=0,
                **params,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            proba = model.predict_proba(X_test)
            preds = np.argmax(proba, axis=1)
            mlogloss_list.append(log_loss(y_test, proba, labels=list(range(n_classes))))
            accuracy_list.append(accuracy_score(y_test, preds))

        return {
            "mean_mlogloss": float(np.mean(mlogloss_list)),
            "mlogloss_std":  float(np.std(mlogloss_list)),
            "accuracies":    accuracy_list,
            "mean_accuracy": float(np.mean(accuracy_list)),
        }

    def _fit_final(self, X: pd.DataFrame, y: np.ndarray, n_classes: int, params: dict) -> None:
        """Re-entrena sobre tot el dataset amb els millors hiperparàmetres."""
        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=n_classes,
            n_estimators=1000,
            use_label_encoder=False,
            random_state=self.random_state,
            verbosity=0,
            **params,
        )
        self.model.fit(X, y, verbose=False)
        logger.info("Final model fitted on full dataset.")

    # ──────────────────────────────────────────────────────────────────────
    # Persistència
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Guarda model + encoder + best_params."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model":         self.model,
                "label_encoder": self.label_encoder,
                "best_params":   self.best_params,
            }, f)
        logger.info(f"XGB regime classifier saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "XGBRegimeClassifier":
        """Carrega un model guardat i retorna una instància llesta per inferència."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.model         = data["model"]
        obj.label_encoder = data["label_encoder"]
        obj.best_params   = data.get("best_params", {})
        obj.n_folds       = 5
        obj.n_trials      = 50
        obj.random_state  = 42
        return obj
