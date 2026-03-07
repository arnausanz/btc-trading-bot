# core/interfaces/base_ml_model.py
from abc import ABC, abstractmethod
import pandas as pd


class BaseMLModel(ABC):
    """
    Base interface for all ML/DL models in the project.
    Ensures every model implements the minimum contract:
      - train() / predict() / save() / load()
      - feature_names: list of expected feature names
      - lookback: number of rows needed by predict() (1 for tree-based, seq_len for RNNs)
      - from_config(): YAML constructor to avoid parameter duplication

    Adding a new model = inherit BaseMLModel + add to registry.
    """

    feature_names: list[str] = []
    is_trained: bool = False

    @property
    def lookback(self) -> int:
        """
        Number of rows needed by predict().
        Tree-based: 1 (last candle only)
        RNNs/Transformers: seq_len (full temporal window)
        """
        return 1

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Trains the model and returns metrics:
        {accuracy_mean, accuracy_std, precision_mean, recall_mean}
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame, threshold: float = 0.35) -> tuple[int, float]:
        """
        Returns (prediction, confidence).
        X must have at least self.lookback rows.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Saves the model to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Loads the model from disk."""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "BaseMLModel":
        """
        Constructor from YAML config.
        Avoids duplicating parameters between the model and train_models.py.
        """
        ...