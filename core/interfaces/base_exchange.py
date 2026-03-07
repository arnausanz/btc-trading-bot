# core/interfaces/base_exchange.py
from abc import ABC, abstractmethod
from core.models import Order, Signal


class BaseExchange(ABC):
    """
    Interface that every exchange must implement.
    Bots are agnostic to paper vs live — they always communicate through this interface.
    """

    @abstractmethod
    def get_candles(self, symbol: str, timeframe: str, limit: int = 500) -> list:
        """Returns the last N market candles."""
        ...

    @abstractmethod
    def send_order(self, signal: Signal) -> Order:
        """Receives a Signal and executes the order on the market."""
        ...

    @abstractmethod
    def get_portfolio(self) -> dict:
        """Returns the current portfolio state. E.g.: {'BTC': 0.5, 'USDT': 10000.0}"""
        ...

    @abstractmethod
    def get_balance(self, currency: str) -> float:
        """Returns the available balance for a specific currency."""
        ...

    def get_portfolio_value(self) -> float:
        """Total portfolio value in USDT. Subclasses should override this."""
        raise NotImplementedError

    def set_current_price(self, price: float) -> None:
        """Updates the current price. Required for paper trading."""
        raise NotImplementedError