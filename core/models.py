# core/models.py
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, field_validator, model_validator
from typing import Union


# --- Enums ---

class Action(str, Enum):
    """Accions possibles d'un bot."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class OrderStatus(str, Enum):
    """Estat d'una ordre."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OrderSide(str, Enum):
    """Direcció d'una ordre."""
    BUY = "buy"
    SELL = "sell"


# --- Models ---

class Candle(BaseModel):
    exchange: str
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @model_validator(mode="after")
    def validate_ohlc_consistency(self):
        if self.high < self.low:
            raise ValueError("high no pot ser menor que low")
        if self.high < self.open or self.high < self.close:
            raise ValueError("high ha de ser el màxim de la candle")
        if self.low > self.open or self.low > self.close:
            raise ValueError("low ha de ser el mínim de la candle")
        return self

    @field_validator("volume")
    @classmethod
    def volume_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("volume no pot ser negatiu")
        return v


class Signal(BaseModel):
    bot_id: str
    timestamp: datetime
    action: Union[Action, float]  # Action per a bots clàssics, float [-1.0, 1.0] per a RL continu
    size: float
    confidence: float
    reason: str

    @model_validator(mode="after")
    def validate_action_and_size(self):
        # Si action és float (RL continu), ha d'estar entre -1.0 i 1.0
        if isinstance(self.action, float):
            if not -1.0 <= self.action <= 1.0:
                raise ValueError("action com a float ha d'estar entre -1.0 i 1.0")
        # size i confidence sempre entre 0 i 1
        if not 0.0 <= self.size <= 1.0:
            raise ValueError("size ha d'estar entre 0.0 i 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence ha d'estar entre 0.0 i 1.0")
        return self

class Order(BaseModel):
    """
    Ordre enviada a un exchange (paper o real).
    """
    id: str
    signal_id: str
    exchange: str
    symbol: str
    side: OrderSide
    status: OrderStatus
    price_target: float
    price_filled: float | None = None
    size: float                        # en BTC
    size_quote: float | None = None    # en USDT — útil per a ordres de mercat
    fees: float = 0.0
    created_at: datetime
    filled_at: datetime | None = None
    metadata: dict = {}                # camp lliure per a context addicional


class Trade(BaseModel):
    """
    Trade completat: des que s'obre fins que es tanca.
    Aquí és on calculem el PnL real.
    """
    id: str
    order_open_id: str      # referència per ID, no objecte embedded
    order_close_id: str
    symbol: str
    pnl_realized: float
    pnl_pct: float          # PnL en percentatge — útil per comparar entre trades
    duration_seconds: float
    metadata: dict = {}     # context de mercat en el moment del trade

    @property
    def is_profitable(self) -> bool:
        return self.pnl_realized > 0