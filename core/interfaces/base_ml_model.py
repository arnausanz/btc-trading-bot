# core/interfaces/base_ml_model.py
from abc import ABC, abstractmethod
import pandas as pd


class BaseMLModel(ABC):
    """
    Interfície base per a tots els models ML/DL del projecte.
    Garanteix que qualsevol model implementa el contracte mínim:
      - train() / predict() / save() / load()
      - feature_names: llista de features que espera
      - lookback: quantes files necessita predict() (1 per tree-based, seq_len per RNNs)
      - from_config(): constructor des de YAML per evitar duplicació de paràmetres

    Afegir un model nou = crear classe que hereti BaseMLModel + afegir al registry.
    """

    feature_names: list[str] = []
    is_trained: bool = False

    @property
    def lookback(self) -> int:
        """
        Quantes files necessita predict().
        Tree-based: 1 (només l'última candle)
        RNNs/Transformers: seq_len (finestra temporal sencera)
        """
        return 1

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Entrena el model i retorna mètriques:
        {accuracy_mean, accuracy_std, precision_mean, recall_mean}
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame, threshold: float = 0.35) -> tuple[int, float]:
        """
        Retorna (predicció, confiança).
        X ha de tenir almenys self.lookback files.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Guarda el model a disc."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Carrega el model des de disc."""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "BaseMLModel":
        """
        Constructor des de config YAML.
        Evita duplicar paràmetres entre el model i train_models.py.
        """
        ...