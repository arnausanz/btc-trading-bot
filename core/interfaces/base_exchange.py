# core/interfaces/base_exchange.py
from abc import ABC, abstractmethod
from core.models import Order, Signal, Candle


class BaseExchange(ABC):
    """
    Interfície que tot exchange ha d'implementar.
    Els bots no saben si estan en paper o en real — sempre parlen amb aquesta interfície.
    """

    @abstractmethod
    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> list[Candle]:
        """Retorna les últimes N candles del mercat."""
        ...

    @abstractmethod
    def send_order(self, signal: Signal) -> Order:
        """
        Rep un Signal i executa l'ordre al mercat.
        Retorna l'ordre amb l'estat actualitzat (filled, cancelled...).
        """
        ...

    @abstractmethod
    def get_portfolio(self) -> dict:
        """
        Retorna l'estat actual del portfolio.
        Exemple: {'BTC': 0.5, 'USDT': 10000.0}
        """
        ...

    @abstractmethod
    def get_balance(self, currency: str) -> float:
        """Retorna el balanç disponible d'una divisa concreta."""
        ...