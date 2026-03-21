# core/db/models.py
from datetime import datetime
from sqlalchemy import String, Float, DateTime, Integer, JSON, ForeignKey, UniqueConstraint, Enum as SAEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from core.models import Action, OrderStatus, OrderSide


class Base(DeclarativeBase):
    pass


class CandleDB(Base):
    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    exchange: Mapped[str] = mapped_column(String(50), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)


class SignalDB(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bot_id: Mapped[str] = mapped_column(String(100), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    action: Mapped[str] = mapped_column(String(20), nullable=False)  # str per suportar float i enum
    size: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str] = mapped_column(String(500), nullable=False)


class OrderDB(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    signal_id: Mapped[str] = mapped_column(String(100), nullable=False)
    exchange: Mapped[str] = mapped_column(String(50), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(SAEnum(OrderSide), nullable=False)
    status: Mapped[str] = mapped_column(SAEnum(OrderStatus), nullable=False)
    price_target: Mapped[float] = mapped_column(Float, nullable=False)
    price_filled: Mapped[float | None] = mapped_column(Float, nullable=True)
    size: Mapped[float] = mapped_column(Float, nullable=False)
    size_quote: Mapped[float | None] = mapped_column(Float, nullable=True)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict)


class TradeDB(Base):
    __tablename__ = "trades"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    order_open_id: Mapped[str] = mapped_column(String(100), ForeignKey("orders.id"), nullable=False)
    order_close_id: Mapped[str] = mapped_column(String(100), ForeignKey("orders.id"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    pnl_realized: Mapped[float] = mapped_column(Float, nullable=False)
    pnl_pct: Mapped[float] = mapped_column(Float, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

class DemoTickDB(Base):
    """Registre de cada tick del DemoRunner."""
    __tablename__ = "demo_ticks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bot_id: Mapped[str] = mapped_column(String(100), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    action: Mapped[str] = mapped_column(String(20), nullable=False)
    portfolio_value: Mapped[float] = mapped_column(Float, nullable=False)
    usdt_balance: Mapped[float] = mapped_column(Float, nullable=False)
    btc_balance: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str] = mapped_column(String(500), nullable=False)


class DemoTradeDB(Base):
    """Registre de cada trade executat durant el Demo."""
    __tablename__ = "demo_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bot_id: Mapped[str] = mapped_column(String(100), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    action: Mapped[str] = mapped_column(String(10), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    size_btc: Mapped[float] = mapped_column(Float, nullable=False)
    size_usdt: Mapped[float] = mapped_column(Float, nullable=False)
    fees: Mapped[float] = mapped_column(Float, nullable=False)
    portfolio_value: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=True, default=0.0)
    reason: Mapped[str] = mapped_column(String(500), nullable=False)


class FearGreedDB(Base):
    """Fear & Greed Index diari d'alternative.me."""
    __tablename__ = "fear_greed"
    __table_args__ = (UniqueConstraint("timestamp", name="uq_fear_greed_timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    value: Mapped[int] = mapped_column(Integer, nullable=False)
    classification: Mapped[str] = mapped_column(String(50), nullable=False)


# --- Dades on-chain ---

class FundingRateDB(Base):
    """Funding rate cada 8h del contracte BTC/USDT:USDT de Binance USDT-M."""
    __tablename__ = "funding_rates"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_funding_rate_symbol_ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    rate: Mapped[float] = mapped_column(Float, nullable=False)


class OpenInterestDB(Base):
    """Open interest del contracte BTC/USDT:USDT de Binance a la granularitat indicada."""
    __tablename__ = "open_interest"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_open_interest_symbol_tf_ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open_interest_btc: Mapped[float] = mapped_column(Float, nullable=False)
    open_interest_usdt: Mapped[float] = mapped_column(Float, nullable=False)


class BlockchainMetricDB(Base):
    """Mètriques diàries de la xarxa Bitcoin via Blockchain.com Charts API."""
    __tablename__ = "blockchain_metrics"
    __table_args__ = (
        UniqueConstraint("metric", "timestamp", name="uq_blockchain_metric_ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    metric: Mapped[str] = mapped_column(String(50), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)


# ── Gate System ───────────────────────────────────────────────────────────────

class GatePositionDB(Base):
    """
    Posicions obertes del GateBot. Persistides entre reinicis.
    Cada registre = 1 posició llarga activa.
    """
    __tablename__ = "gate_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bot_id: Mapped[str] = mapped_column(String(100), nullable=False)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    stop_level: Mapped[float] = mapped_column(Float, nullable=False)    # s'actualitza amb trailing
    target_level: Mapped[float] = mapped_column(Float, nullable=False)
    highest_price: Mapped[float] = mapped_column(Float, nullable=False) # màxim des d'entrada
    size_usdt: Mapped[float] = mapped_column(Float, nullable=False)      # mida en USDT
    regime: Mapped[str] = mapped_column(String(20), nullable=False)      # règim en el moment d'entrada
    decel_counter: Mapped[int] = mapped_column(Integer, default=0)       # candles consecutives amb d2<0


class GateNearMissDB(Base):
    """
    Registre d'avaluacions on P1+P2+P3 han passat, independentment de si s'executa un trade.
    Permet analitzar quines portes tanquen més i per quin motiu (calibratge de llindars).
    Crida: cada cop que P3 retorna has_actionable_level=True.
    """
    __tablename__ = "gate_near_misses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    bot_id: Mapped[str] = mapped_column(String(100), nullable=False)

    # P1
    p1_regime: Mapped[str] = mapped_column(String(20), nullable=False)
    p1_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # P2
    p2_multiplier: Mapped[float] = mapped_column(Float, nullable=False)

    # P3
    p3_level_type: Mapped[str | None] = mapped_column(String(20), nullable=True)   # 'support'|'resistance'
    p3_level_strength: Mapped[float | None] = mapped_column(Float, nullable=True)
    p3_risk_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    p3_volume_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)

    # P4 — senyals individuals (True = actiu)
    p4_d1_ok: Mapped[bool | None] = mapped_column(nullable=True)
    p4_d2_ok: Mapped[bool | None] = mapped_column(nullable=True)
    p4_rsi_ok: Mapped[bool | None] = mapped_column(nullable=True)
    p4_macd_ok: Mapped[bool | None] = mapped_column(nullable=True)
    p4_score: Mapped[float | None] = mapped_column(Float, nullable=True)    # 0.0–1.0
    p4_triggered: Mapped[bool | None] = mapped_column(nullable=True)

    # P5
    p5_veto_reason: Mapped[str | None] = mapped_column(String(100), nullable=True)  # NULL = no vetat
    p5_position_size: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Resultat final
    executed: Mapped[bool] = mapped_column(nullable=False, default=False)