# bots/ml/gru_model.py
import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

from core.interfaces.base_ml_model import BaseMLModel

logger = logging.getLogger(__name__)


class BidirectionalGRU(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        return out.squeeze(-1)


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int):
        return self.X[idx:idx + self.seq_len], self.y[idx + self.seq_len]


class GRUModel(BaseMLModel):
    def __init__(
        self,
        seq_len: int = 50,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        epochs: int = 15,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        patience: int = 4,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.feature_names: list[str] = []
        self.is_trained = False
        self.scaler = StandardScaler()
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.net: BidirectionalGRU | None = None

        torch.set_num_threads(1)  # avoid BLAS/OpenMP deadlock on macOS
        logger.info(f"GRUModel device: {self.device}")

    @property
    def lookback(self) -> int:
        return self.seq_len

    @classmethod
    def from_config(cls, config: dict) -> "GRUModel":
        return cls(
            seq_len=config["model"]["seq_len"],
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
            epochs=config["model"]["epochs"],
            batch_size=config["model"]["batch_size"],
            learning_rate=config["model"]["learning_rate"],
            patience=config["model"].get("patience", 4),
        )

    def _build_net(self, n_features: int) -> BidirectionalGRU:
        return BidirectionalGRU(
            n_features=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TimeSeriesDataset(X, y, self.seq_len)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        from core.config import MLFLOW_TRACKING_URI
        self.feature_names = list(X.columns)
        n_features = len(self.feature_names)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("gru_bidireccional")
        run = mlflow.start_run()

        try:
            mlflow.log_params({
                "seq_len": self.seq_len,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
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

            with tqdm(range(5), desc="  GRU folds", unit="fold",
                      leave=False, dynamic_ncols=True) as fold_bar:
                for fold in fold_bar:
                    split = fold_size * (fold + 1)
                    val_end = min(split + fold_size, n)

                    X_train_raw = X_arr[:split]
                    y_train_raw = y_arr[:split]
                    X_val_raw = X_arr[split:val_end]
                    y_val_raw = y_arr[split:val_end]

                    if len(X_val_raw) <= self.seq_len:
                        continue

                    X_train_sc = self.scaler.fit_transform(X_train_raw)
                    X_val_sc = self.scaler.transform(X_val_raw)

                    net = self._build_net(n_features)
                    acc, prec, rec = self._train_fold(
                        net, X_train_sc, y_train_raw,
                        X_val_sc, y_val_raw,
                        fold_num=fold + 1,
                    )
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

            tqdm.write("  Training final GRU model...")
            X_scaled = self.scaler.fit_transform(X_arr)
            self.net = self._build_net(n_features)
            self._train_fold(
                self.net, X_scaled, y_arr,
                X_scaled[-fold_size:], y_arr[-fold_size:],
                fold_num=0,
            )
            self.is_trained = True

            tqdm.write(f"  ✓ GRU  acc={metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}")
            return metrics

        finally:
            mlflow.end_run()

    def _train_fold(
        self,
        net: BidirectionalGRU,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        fold_num: int = 0,
    ) -> tuple[float, float, float]:
        train_loader = self._make_loader(X_train, y_train, shuffle=False)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.5
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        label = "final" if fold_num == 0 else f"fold {fold_num}"

        with tqdm(range(self.epochs), desc=f"    GRU {label}", unit="ep",
                  leave=False, dynamic_ncols=True) as epoch_bar:
            for epoch in epoch_bar:
                net.train()
                epoch_loss = 0.0

                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    optimizer.zero_grad()
                    preds = net(X_batch)
                    loss = criterion(preds, y_batch)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()

                # Validation
                net.eval()
                val_losses = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        preds = net(X_batch.to(self.device))
                        val_losses.append(criterion(preds, y_batch.to(self.device)).item())

                val_loss = float(np.mean(val_losses))
                scheduler.step(val_loss)
                epoch_bar.set_postfix({
                    "loss": f"{epoch_loss/len(train_loader):.4f}",
                    "val":  f"{val_loss:.4f}",
                    "lr":   f"{optimizer.param_groups[0]['lr']:.1e}",
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        epoch_bar.set_description(f"    GRU {label} (early stop)")
                        break

        if best_state:
            net.load_state_dict(best_state)

        return self._evaluate(net, val_loader)

    def _evaluate(
        self, net: BidirectionalGRU, loader: DataLoader, threshold: float = 0.35
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

    def predict(self, X: pd.DataFrame, threshold: float = 0.35) -> tuple[int, float]:
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

    def save(self, path: str) -> None:
        torch.save({
            "net_state": self.net.state_dict(),
            "net_config": {
                "n_features": len(self.feature_names),
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "seq_len": self.seq_len,
        }, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.feature_names = data["feature_names"]
        self.seq_len = data["seq_len"]
        self.scaler = data["scaler"]
        cfg = data["net_config"]
        self.net = BidirectionalGRU(**cfg).to(self.device)
        self.net.load_state_dict(data["net_state"])
        self.net.eval()
        self.is_trained = True