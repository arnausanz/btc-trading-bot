# core/models.py
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, field_validator, model_validator


class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


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
    action: Action | float  # Action per a bots clàssics, float [-1.0, 1.0] per a RL continu
    size: float
    confidence: float
    reason: str

    @model_validator(mode="after")
    def validate_action_and_size(self):
        if isinstance(self.action, float):
            if not -1.0 <= self.action <= 1.0:
                raise ValueError("action com a float ha d'estar entre -1.0 i 1.0")
        if not 0.0 <= self.size <= 1.0:
            raise ValueError("size ha d'estar entre 0.0 i 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence ha d'estar entre 0.0 i 1.0")
        return self


class Order(BaseModel):
    id: str
    signal_id: str
    exchange: str
    symbol: str
    side: OrderSide
    status: OrderStatus
    price_target: float
    price_filled: float | None = None
    size: float
    size_quote: float | None = None
    fees: float = 0.0
    created_at: datetime
    filled_at: datetime | None = None
    metadata: dict = {}


class Trade(BaseModel):
    id: str
    order_open_id: str
    order_close_id: str
    symbol: str
    pnl_realized: float
    pnl_pct: float
    duration_seconds: float
    metadata: dict = {}

    @property
    def is_profitable(self) -> bool:
        return self.pnl_realized > 0


class FearGreedEntry(BaseModel):
    timestamp: datetime
    value: int
    classification: str

    @field_validator("value")
    @classmethod
    def value_in_range(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("value ha d'estar entre 0 i 100")
        return v


# --- Dades on-chain ---

class FundingRateEntry(BaseModel):
    """Funding rate d'un contracte perp de Binance (cada 8h)."""
    symbol: str
    timestamp: datetime
    rate: float  # e.g. 0.0001 = 0.01%; pot ser negatiu


class OpenInterestEntry(BaseModel):
    """Open interest d'un contracte perp de Binance a una granularitat donada."""
    symbol: str
    timeframe: str          # e.g. '1h'
    timestamp: datetime
    open_interest_btc: float   # quantitat en BTC
    open_interest_usdt: float  # valor en USDT

    @field_validator("open_interest_btc", "open_interest_usdt")
    @classmethod
    def must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError("open interest no pot ser negatiu")
        return v


# Mètriques vàlides de Blockchain.com Charts
BLOCKCHAIN_METRICS = frozenset({
    "hash-rate",           # TH/s — potència de mineria de la xarxa
    "n-unique-addresses",  # nombre d'adreces actives úniques diàries
    "transaction-fees",    # comissions totals diàries (en BTC)
})


class BlockchainMetricEntry(BaseModel):
    """Mètrica diària de la xarxa Bitcoin via Blockchain.com Charts API."""
    metric: str
    timestamp: datetime
    value: float

    @field_validator("metric")
    @classmethod
    def metric_must_be_known(cls, v):
        if v not in BLOCKCHAIN_METRICS:
            raise ValueError(
                f"metric '{v}' no és coneguda. Vàlides: {sorted(BLOCKCHAIN_METRICS)}"
            )
        return v

    @field_validator("value")
    @classmethod
    def value_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError("value no pot ser negatiu")
        return v