# data/processing/torch_dataset.py
"""
Shared PyTorch Dataset for time-series window sequences.

Used by GRU, PatchTST, and any future deep-learning model (TFT, N-BEATS, …).
Centralised here so the class is defined exactly once and all models reuse it.

Usage:
    from data.processing.torch_dataset import TimeSeriesDataset
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Sliding-window Dataset that converts (X, y) arrays into (sequence, label) pairs.

    Each sample is a window of `seq_len` consecutive feature vectors, with the
    label corresponding to the timestep immediately following the window.

    Args:
        X:       2-D array of shape (n_samples, n_features), already scaled.
        y:       1-D array of shape (n_samples,) with binary labels.
        seq_len: Length of the look-back window (number of timesteps per sample).

    Example:
        seq_len = 50, len(X) = 1000
        → 950 valid samples: indices 0..949
        → sample[i] = (X[i : i+50], y[i+50])
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int):
        return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len]
