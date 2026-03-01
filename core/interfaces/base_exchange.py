# core/interfaces/base_exchange.py
from abc import ABC, abstractmethod
from core.models import Order, Signal


class BaseExchange(ABC):
    """
    Interfície que tot exchange ha d'implementar.
    Els bots no saben si estan en paper o en real — sempre parlen amb aquesta interfície.
    """

    @abstractmethod
    def get_candles(self, symbol: str, timeframe: str, limit: int = 500) -> list:
        """Retorna les últimes N candles del mercat."""
        ...

    @abstractmethod
    def send_order(self, signal: Signal) -> Order:
        """Rep un Signal i executa l'ordre al mercat."""
        ...

    @abstractmethod
    def get_portfolio(self) -> dict:
        """Retorna l'estat actual del portfolio. Ex: {'BTC': 0.5, 'USDT': 10000.0}"""
        ...

    @abstractmethod
    def get_balance(self, currency: str) -> float:
        """Retorna el balanç disponible d'una divisa concreta."""
        ...

    def get_portfolio_value(self) -> float:
        """Valor total del portfolio en USDT. Subclasses haurien de sobreescriure'l."""
        raise NotImplementedError

    def set_current_price(self, price: float) -> None:
        """Actualitza el preu actual. Necessari per a paper trading."""
        raise NotImplementedError