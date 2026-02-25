# tests/unit/test_models.py
import pytest
from datetime import datetime, timezone
from core.models import Candle, Signal, Action, Order, OrderSide, OrderStatus, Trade


def make_candle(**kwargs) -> Candle:
    """Factory per crear candles de test."""
    defaults = {
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "timestamp": datetime.now(timezone.utc),
        "open": 50000.0,
        "high": 51000.0,
        "low": 49000.0,
        "close": 50500.0,
        "volume": 100.0,
    }
    return Candle(**{**defaults, **kwargs})


def test_candle_valid():
    candle = make_candle()
    assert candle.close == 50500.0


def test_candle_high_lower_than_low_raises():
    with pytest.raises(Exception):
        make_candle(high=48000.0, low=49000.0)


def test_candle_negative_volume_raises():
    with pytest.raises(Exception):
        make_candle(volume=-1.0)


def test_signal_valid():
    signal = Signal(
        bot_id="dca_v1",
        timestamp=datetime.now(timezone.utc),
        action=Action.BUY,
        size=0.5,
        confidence=0.8,
        reason="EMA crossover detectat",
    )
    assert signal.action == Action.BUY


def test_signal_size_out_of_range_raises():
    with pytest.raises(Exception):
        Signal(
            bot_id="dca_v1",
            timestamp=datetime.now(timezone.utc),
            action=Action.BUY,
            size=1.5,  # invalid
            confidence=0.8,
            reason="test",
        )

def test_order_valid():
    order = Order(
        id="ord_001",
        signal_id="sig_001",
        exchange="paper",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        status=OrderStatus.FILLED,
        price_target=50000.0,
        price_filled=50100.0,
        size=0.1,
        created_at=datetime.now(timezone.utc),
    )
    assert order.status == OrderStatus.FILLED


def test_trade_valid():
    trade = Trade(
        id="trade_001",
        order_open_id="ord_001",
        order_close_id="ord_002",
        symbol="BTC/USDT",
        pnl_realized=150.0,
        pnl_pct=0.03,
        duration_seconds=3600.0,
    )
    assert trade.is_profitable is True


def test_trade_not_profitable():
    trade = Trade(
        id="trade_002",
        order_open_id="ord_003",
        order_close_id="ord_004",
        symbol="BTC/USDT",
        pnl_realized=-50.0,
        pnl_pct=-0.01,
        duration_seconds=1800.0,
    )
    assert trade.is_profitable is False