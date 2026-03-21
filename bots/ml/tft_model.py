# bots/ml/tft_model.py
"""
Temporal Fusion Transformer (TFT) for BTC price direction classification.

Architecture overview:
────────────────────────────────────────────────────────────────────────
  TFT (Lim et al., 2021 — "Temporal Fusion Transformers for Interpretable
  Multi-horizon Time Series Forecasting") adapted for binary classification.

  Key components implemented:
    GRN  Gated Residual Network — shared building block throughout the model.
         Input → ELU → hidden → GLU (gate × value) → LayerNorm + skip.
         Controls how much of each signal passes to the next layer.

    VSN  Variable Selection Network — learns WHICH input features matter
         at each timestep. Each feature gets its own GRN; a second GRN
         over the flattened features produces soft selection weights.
         Result: d_model-dimensional context vector per timestep.

    Transformer encoder — multi-head self-attention across time steps.
         Learns WHEN in the lookback window events are relevant.
         Uses pre-norm (norm_first=True) for better gradient flow.

    Classification head — GRN + Linear → Sigmoid → P(price up in 24h).

Why TFT over GRU for BTC?
  GRU treats all features equally and weights all timesteps by recurrence.
  TFT explicitly learns feature importance (VSN) and temporal importance
  (attention), making it interpretable and more robust when only a subset
  of the 30+ technical indicators is predictive at any given moment.

References:
  Lim et al., 2021 — https://arxiv.org/abs/1912.09363
  "Temporal Fusion Transformers for Interpretable Multi-horizon
  Time Series Forecasting" — Google AI / Oxford
"""
import logging
import os

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.interfaces.base_ml_model import BaseMLModel
from data.processing.torch_dataset import TimeSeriesDataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Neural network modules
# ─────────────────────────────────────────────────────────────────────────────


class GatedResidualNetwork(nn.Module):
    """
    Core TFT building block.  Applies a gating mechanism that lets the network
    decide how much new information to accept vs. pass through unchanged.

    Gating:  sigmoid(W·h) × tanh-like(h) — controls information flow.
    Residual: output + skip(input) — preserves gradient signal.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size * 2)   # gate + value
        self.layer_norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)
        # Skip connection: match dimensions if needed
        self.skip = (
            nn.Linear(input_size, output_size, bias=False)
            if input_size != output_size
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        # Gated Linear Unit: split into gate and value, then gate the value
        gate, value = h.chunk(2, dim=-1)
        h = torch.sigmoid(gate) * value
        h = self.dropout(h)
        return self.layer_norm(h + self.skip(x))


class VariableSelectionNetwork(nn.Module):
    """
    Learns which input features are relevant at each timestep.

    Each feature gets its own GRN (independent embedding).
    A second GRN over the flat feature vector produces selection weights.
    The final output is the attention-weighted sum of feature embeddings.

    This gives the model interpretable feature importance per timestep.
    """

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        # One GRN per feature: 1 input dim → d_model output
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, d_model, d_model, dropout)
            for _ in range(n_features)
        ])
        # Selection GRN: n_features inputs → n_features selection weights
        self.selection_grn = GatedResidualNetwork(
            n_features, d_model, n_features, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, seq_len, d_model) — feature-weighted context
        """
        # Embed each feature independently: (batch, seq_len, n_features, d_model)
        embedded = torch.stack(
            [grn(x[..., i: i + 1]) for i, grn in enumerate(self.feature_grns)],
            dim=-2,
        )
        # Soft selection weights over features: (batch, seq_len, n_features)
        weights = torch.softmax(self.selection_grn(x), dim=-1)
        # Weighted sum of embeddings: (batch, seq_len, d_model)
        return (embedded * weights.unsqueeze(-1)).sum(dim=-2)


class TemporalFusionTransformerNet(nn.Module):
    """
    Simplified TFT for binary classification.

    Pipeline:
      x → VSN → TransformerEncoder → last-step GRN → classifier → P(up)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vsn = VariableSelectionNetwork(n_features, d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm: more stable for long sequences
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.grn_out = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch,) — probability of price up
        """
        z = self.vsn(x)           # (batch, seq_len, d_model)
        z = self.transformer(z)   # (batch, seq_len, d_model)
        z = z[:, -1, :]           # use last timestep: (batch, d_model)
        z = self.grn_out(z)       # gated refinement
        return self.classifier(z).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# TFTModel — wraps the network in the BaseMLModel interface
# ─────────────────────────────────────────────────────────────────────────────


class TFTModel(BaseMLModel):
    """
    Temporal Fusion Transformer for BTC price direction prediction.

    Follows the same interface as GRUModel and PatchTSTModel:
    DatasetBuilder → X, y → model.train(X, y) → model.save(path).
    Shares TimeSeriesDataset with GRU and PatchTST.
    """

    def __init__(
        self,
        seq_len: int = 96,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 5e-4,
        patience: int = 5,
    ):
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience

        self.feature_names: list[str] = []
        self.is_trained = False
        self.scaler = StandardScaler()
        self.net: TemporalFusionTransformerNet | None = None

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        torch.set_num_threads(1)   # avoid BLAS/OpenMP deadlock on macOS
        logger.info(f"TFTModel device: {self.device}")

    @property
    def lookback(self) -> int:
        return self.seq_len

    @classmethod
    def from_config(cls, config: dict) -> "TFTModel":
        m = config["model"]
        return cls(
            seq_len=m["seq_len"],
            d_model=m["d_model"],
            n_heads=m["n_heads"],
            n_layers=m["n_layers"],
            dropout=m["dropout"],
            epochs=m["epochs"],
            batch_size=m["batch_size"],
            learning_rate=m["learning_rate"],
            patience=m.get("patience", 5),
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_net(self, n_features: int) -> TemporalFusionTransformerNet:
        return TemporalFusionTransformerNet(
            n_features=n_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TimeSeriesDataset(X, y, self.seq_len)
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=0, pin_memory=False,
        )

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        from core.config import MLFLOW_TRACKING_URI

        self.feature_names = list(X.columns)
        n_features = len(self.feature_names)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("tft")
        mlflow.start_run()

        try:
            mlflow.log_params({
                "seq_len": self.seq_len,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "dropout": self.dropout,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "n_features": n_features,
                "n_samples": len(X),
            })

            X_arr = X.values
            y_arr = y.values
            n = len(X_arr)
            fold_size = n // 6
            accuracies, precisions, recalls = [], [], []

            with tqdm(
                range(5), desc="  TFT folds", unit="fold",
                leave=False, dynamic_ncols=True,
            ) as fold_bar:
                for fold in fold_bar:
                    split   = fold_size * (fold + 1)
                    val_end = min(split + fold_size, n)

                    X_train_raw = X_arr[:split]
                    y_train_raw = y_arr[:split]
                    X_val_raw   = X_arr[split:val_end]
                    y_val_raw   = y_arr[split:val_end]

                    if len(X_val_raw) <= self.seq_len:
                        continue

                    X_train_sc = self.scaler.fit_transform(X_train_raw)
                    X_val_sc   = self.scaler.transform(X_val_raw)

                    net = self._build_net(n_features)
                    acc, prec, rec = self._train_fold(
                        net, X_train_sc, y_train_raw,
                        X_val_sc, y_val_raw, fold_num=fold + 1,
                    )
                    accuracies.append(acc)
                    precisions.append(prec)
                    recalls.append(rec)
                    fold_bar.set_postfix({
                        "acc": f"{acc:.3f}", "prec": f"{prec:.3f}", "rec": f"{rec:.3f}",
                    })

            metrics = {
                "accuracy_mean":  float(np.mean(accuracies)),
                "accuracy_std":   float(np.std(accuracies)),
                "precision_mean": float(np.mean(precisions)),
                "recall_mean":    float(np.mean(recalls)),
            }
            mlflow.log_metrics(metrics)

            tqdm.write("  Training final TFT model...")
            X_scaled = self.scaler.fit_transform(X_arr)
            self.net = self._build_net(n_features)
            self._train_fold(
                self.net, X_scaled, y_arr,
                X_scaled[-fold_size:], y_arr[-fold_size:],
                fold_num=0,
            )
            self.is_trained = True

            tqdm.write(
                f"  ✓ TFT   acc={metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}"
            )
            return metrics

        finally:
            mlflow.end_run()

    def _train_fold(
        self,
        net: TemporalFusionTransformerNet,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        fold_num: int = 0,
    ) -> tuple[float, float, float]:
        train_loader = self._make_loader(X_train, y_train, shuffle=False)
        val_loader   = self._make_loader(X_val,   y_val,   shuffle=False)

        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.5
        )

        best_val_loss   = float("inf")
        patience_counter = 0
        best_state      = None
        label = "final" if fold_num == 0 else f"fold {fold_num}"

        with tqdm(
            range(self.epochs), desc=f"    TFT {label}", unit="ep",
            leave=False, dynamic_ncols=True,
        ) as epoch_bar:
            for _ in epoch_bar:
                # ── Training step ──────────────────────────────────────────
                net.train()
                epoch_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    optimizer.zero_grad()
                    preds = net(X_batch)
                    loss  = criterion(preds, y_batch)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()

                # ── Validation step ────────────────────────────────────────
                net.eval()
                val_losses = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        preds = net(X_batch.to(self.device))
                        val_losses.append(
                            criterion(preds, y_batch.to(self.device)).item()
                        )

                val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
                scheduler.step(val_loss)
                epoch_bar.set_postfix({
                    "loss": f"{epoch_loss / max(len(train_loader), 1):.4f}",
                    "val":  f"{val_loss:.4f}",
                    "lr":   f"{optimizer.param_groups[0]['lr']:.1e}",
                })

                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        epoch_bar.set_description(f"    TFT {label} (early stop)")
                        break

        if best_state:
            net.load_state_dict(best_state)

        return self._evaluate(net, val_loader)

    def _evaluate(
        self,
        net: TemporalFusionTransformerNet,
        loader: DataLoader,
        threshold: float = 0.5,
    ) -> tuple[float, float, float]:
        net.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                probs = net(X_batch.to(self.device)).cpu().numpy()
                all_preds.extend((probs >= threshold).astype(int))
                all_labels.extend(y_batch.numpy().astype(int))
        if not all_preds:
            return 0.0, 0.0, 0.0
        return (
            accuracy_score(all_labels, all_preds),
            precision_score(all_labels, all_preds, zero_division=0),
            recall_score(all_labels, all_preds, zero_division=0),
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> tuple[int, float]:
        if not self.is_trained or self.net is None:
            raise RuntimeError("Model not trained. Call train() first.")
        X_scaled = self.scaler.transform(X.values)
        if len(X_scaled) < self.seq_len:
            raise ValueError(f"predict() needs {self.seq_len} rows, got {len(X_scaled)}")
        seq = torch.tensor(
            X_scaled[-self.seq_len:], dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        self.net.eval()
        with torch.no_grad():
            proba = self.net(seq).item()
        return int(proba >= threshold), float(proba)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        torch.save({
            "net_state": self.net.state_dict(),
            "net_config": {
                "n_features": len(self.feature_names),
                "d_model":    self.d_model,
                "n_heads":    self.n_heads,
                "n_layers":   self.n_layers,
                "dropout":    self.dropout,
            },
            "scaler":        self.scaler,
            "feature_names": self.feature_names,
            "seq_len":       self.seq_len,
        }, path)
        logger.info(f"TFTModel saved to {path}")

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.feature_names = data["feature_names"]
        self.seq_len       = data["seq_len"]
        self.scaler        = data["scaler"]
        cfg = data["net_config"]
        self.net = TemporalFusionTransformerNet(**cfg).to(self.device)
        self.net.load_state_dict(data["net_state"])
        self.net.eval()
        self.is_trained = True
        logger.info(f"TFTModel loaded from {path}")
