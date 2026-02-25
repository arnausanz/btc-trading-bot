# core/interfaces/base_strategy.py
from abc import ABC, abstractmethod
import pandas as pd
from core.models import Signal


class BaseStrategy(ABC):
    """
    Interfície per a estratègies de trading clàssiques.
    Una Strategy encapsula la lògica de decisió basada en indicadors tècnics.
    És el component intern d'un Bot clàssic — el Bot crida la Strategy.
    """

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, bot_id: str) -> Signal:
        """
        Rep un DataFrame amb features calculades i retorna un Signal.
        df té com a mínim les columnes OHLCV + els indicadors tècnics.
        """
        ...