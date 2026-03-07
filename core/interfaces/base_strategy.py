# core/interfaces/base_strategy.py
from abc import ABC, abstractmethod
import pandas as pd
from core.models import Signal


class BaseStrategy(ABC):
    """
    Interface for classical trading strategies.
    A Strategy encapsulates decision logic based on technical indicators.
    It is the inner component of a classical Bot — the Bot calls the Strategy.
    """

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, bot_id: str) -> Signal:
        """
        Receives a DataFrame with computed features and returns a Signal.
        df must have at least the OHLCV columns + technical indicators.
        """
        ...