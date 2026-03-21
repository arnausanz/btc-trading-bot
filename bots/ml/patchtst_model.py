# bots/ml/patchtst_model.py
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from core.interfaces.base_ml_model import BaseMLModel
from data.processing.torch_dataset import TimeSeriesDataset

logger = logging.getLogger(__name__)


# ── Arquitectura ──────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Divides the temporal sequence into patches and projects them to d_model dimensions.
    Each patch is a group of patch_len consecutive candles.
    Equivalent to "tokens" in a text Transformer.

    Example: seq_len=96, patch_len=16, stride=8
    → (96 - 16) / 8 + 1 = 11 patches
    Each patch represents 16 candles with 8 overlap.
    """

    def __init__(self, n_features: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        # Linear projection of each patch (patch_len * n_features) → d_model
        self.projection = nn.Linear(patch_len * n_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        batch, seq_len, n_features = x.shape
        patches = x.unfold(1, self.patch_len, self.stride)
        patches = patches.permute(0, 1, 3, 2)
        patches = patches.reshape(batch, -1, self.patch_len * n_features).contiguous()
        return self.projection(patches)


class PatchTSTNet(nn.Module):
    """
    PatchTST for binary classification.
    Architecture:
        Input (seq_len, n_features)
        → PatchEmbedding → (n_patches, d_model)
        → Positional Encoding
        → Transformer Encoder (n_heads, n_layers)
        → Mean pooling over patches
        → FC → Sigmoid

    Why it outperforms GRU/LSTM:
    - Sees all patches simultaneously (self-attention), not sequentially
    - Captures long-range dependencies without vanishing gradient
    - Patches preserve local structure (16 candles) + global (between patches)
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        device: torch.device | None = None
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(n_features, patch_len, stride, d_model)

        # Number of resulting patches
        n_patches = (seq_len - patch_len) // stride + 1
        self.n_patches = n_patches

        # Positional encoding learnable (more flexible than sinusoidal)
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_patches, d_model, device=device))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN: more stable than Post-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = x.to(self.pos_embedding.device)     # ensure input is on the same device as the model
        out = self.patch_embedding(x)           # (batch, n_patches, d_model)
        out = out + self.pos_embedding           # add position
        out = self.transformer(out)              # (batch, n_patches, d_model)
        out = out.mean(dim=1)                    # mean pooling → (batch, d_model)
        out = self.dropout(out)
        out = self.classifier(out)               # (batch, 1)
        return out.squeeze(-1)                   # (batch,)


# ── Model wrapper ─────────────────────────────────────────────────────────────

class PatchTSTModel(BaseMLModel):
    """
    PatchTST wrapper with the same interface as GRUModel.
    Add to registry = one line. The rest is automatic.
    """

    def __init__(
        self,
        seq_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        patience: int = 5,
    ):
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
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
        self.net: PatchTSTNet | None = None

        torch.set_num_threads(1)  # avoid BLAS/OpenMP deadlock on macOS

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(f"PatchTSTModel device: {self.device}")

    @property
    def lookback(self) -> int:
        return self.seq_len

    @classmethod
    def from_config(cls, config: dict) -> "PatchTSTModel":
        m = config["model"]
        return cls(
            seq_len=m["seq_len"],
            patch_len=m["patch_len"],
            stride=m["stride"],
            d_model=m["d_model"],
            n_heads=m["n_heads"],
            n_layers=m["n_layers"],
            dropout=m["dropout"],
            epochs=m["epochs"],
            batch_size=m["batch_size"],
            learning_rate=m["learning_rate"],
            patience=m.get("patience", 5),
        )

    def _build_net(self, n_features: int) -> PatchTSTNet:
        return PatchTSTNet(
            n_features=n_features,
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            device=self.device,
        ).to(self.device)

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TimeSeriesDataset(X, y, self.seq_len)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        self.feature_names = list(X.columns)
        n_features = len(self.feature_names)

        # Validates that patches fit with seq_len
        n_patches = (self.seq_len - self.patch_len) // self.stride + 1
        assert n_patches > 0, (
            f"Invalid configuration: seq_len={self.seq_len}, "
            f"patch_len={self.patch_len}, stride={self.stride} → {n_patches} patches"
        )
        logger.info(
            f"  PatchTST: seq_len={self.seq_len}, patch_len={self.patch_len}, "
            f"stride={self.stride} → {n_patches} patches x d_model={self.d_model}"
        )

        X_arr = X.values
        y_arr = y.values
        n = len(X_arr)
        fold_size = n // 6
        accuracies, precisions, recalls = [], [], []

        with tqdm(range(5), desc="  TST folds", unit="fold",
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

        tqdm.write("  Training final PatchTST model...")
        X_scaled = self.scaler.fit_transform(X_arr)
        self.net = self._build_net(n_features)
        self._train_fold(
            self.net, X_scaled, y_arr,
            X_scaled[-fold_size:], y_arr[-fold_size:],
            fold_num=0,
        )
        self.is_trained = True

        tqdm.write(f"  ✓ TST  acc={metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}")
        return metrics

    def _train_fold(
        self,
        net: PatchTSTNet,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        fold_num: int = 0,
    ) -> tuple[float, float, float]:
        train_loader = self._make_loader(X_train, y_train, shuffle=False)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,  # AdamW better than Adam for Transformers
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,  # Cosine better than ReduceLROnPlateau for Transformers
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        label = "final" if fold_num == 0 else f"fold {fold_num}"

        with tqdm(range(self.epochs), desc=f"    TST {label}", unit="ep",
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

                scheduler.step()

                # Validation
                net.eval()
                val_losses = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        preds = net(X_batch.to(self.device))
                        val_losses.append(criterion(preds, y_batch.to(self.device)).item())

                val_loss = float(np.mean(val_losses))
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
                        epoch_bar.set_description(f"    TST {label} (early stop)")
                        break

        if best_state:
            net.load_state_dict(best_state)

        return self._evaluate(net, val_loader)

    def _evaluate(
        self, net: PatchTSTNet, loader: DataLoader, threshold: float = 0.5
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

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> tuple[int, float]:
        if not self.is_trained or self.net is None:
            raise RuntimeError("Model not trained. Call train() first.")
        X_scaled = self.scaler.transform(X.values)
        if len(X_scaled) < self.seq_len:
            raise ValueError(
                f"predict() needs {self.seq_len} rows, got {len(X_scaled)}"
            )
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
                "seq_len": self.seq_len,
                "patch_len": self.patch_len,
                "stride": self.stride,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "dropout": self.dropout,
            },
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "seq_len": self.seq_len,
        }, path)
        logger.info(f"PatchTSTModel saved to {path}")

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.feature_names = data["feature_names"]
        self.seq_len = data["seq_len"]
        self.scaler = data["scaler"]
        cfg = data["net_config"]
        self.net = PatchTSTNet(**cfg, device=None).to(self.device)
        self.net.load_state_dict(data["net_state"])
        self.net.eval()
        self.is_trained = True
        logger.info(f"PatchTSTModel loaded from {path}")