# core/interfaces/base_bot.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from core.models import Signal


@dataclass
class ObservationSchema:
    """
    Declara quines dades necessita un bot per funcionar.
    El Runner usarà aquest schema per construir l'observació automàticament.
    """
    features: list[str]                    # columnes del DataFrame: 'close', 'rsi_14'...
    timeframes: list[str]                  # timeframes necessaris: '1h', '4h'...
    lookback: int                          # quantes candles enrere necessita veure
    extras: dict[str, Any] = field(default_factory=dict)  # fonts externes: sentiment, onchain...


class BaseBot(ABC):
    """
    Interfície que tot bot ha d'implementar.
    El Runner no sap quin tipus de bot és — només crida observation_schema i on_observation.
    """

    def __init__(self, bot_id: str, config: dict):
        self.bot_id = bot_id
        self.config = config

    @abstractmethod
    def observation_schema(self) -> ObservationSchema:
        """
        Declara quines dades necessita aquest bot.
        S'executa una vegada en inicialitzar el bot.
        """
        ...

    @abstractmethod
    def on_observation(self, observation: dict) -> Signal:
        """
        Rep l'observació i retorna una decisió.
        Aquest és el cervell del bot.
        """
        ...

    def on_start(self) -> None:
        """Hook opcional: s'executa quan el bot arrenca."""
        pass

    def on_stop(self) -> None:
        """Hook opcional: s'executa quan el bot s'atura."""
        pass