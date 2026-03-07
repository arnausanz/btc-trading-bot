# tests/smoke/test_data_models.py
"""Test Pydantic data models validation."""
import pytest
from datetime import datetime, timezone
from core.models import Candle, Signal, Action, Order, OrderSide, OrderStatus, Trade


def make_candle(**kwargs) -> Candle:
    """Factory for creating test candles."""
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
    """Candle with high < low should raise validation error."""
    with pytest.raises(ValueError):
        make_candle(high=48000.0, low=49000.0)


def test_candle_high_below_open_raises():
    """Candle with high below open should raise validation error."""
    with pytest.raises(ValueError):
        make_candle(high=49000.0, open=50000.0)


def test_candle_low_above_close_raises():
    """Candle with low above close should raise validation error."""
    with pytest.raises(ValueError):
        make_candle(low=51000.0, close=50000.0)


def test_candle_negative_volume_raises():
    """Candle with negative volume should raise validation error."""
    with pytest.raises(ValueError):
        make_candle(volume=-1.0)


def test_candle_zero_volume_allowed():
    """Candle with zero volume is allowed."""
    candle = make_candle(volume=0.0)
    assert candle.volume == 0.0


def test_signal_valid():
    signal = Signal(
        bot_id="dca_v1",
        timestamp=datetime.now(timezone.utc),
        action=Action.BUY,
        size=0.5,
        confidence=0.8,
        reason="EMA crossover detected",
    )
    assert signal.action == Action.BUY


def test_signal_size_out_of_range_raises():
    """Signal size > 1.0 should raise validation error."""
    with pytest.raises(ValueError):
        Signal(
            bot_id="dca_v1",
            timestamp=datetime.now(timezone.utc),
            action=Action.BUY,
            size=1.5,
            confidence=0.8,
            reason="test",
        )


def test_signal_size_negative_raises():
    """Signal size < 0.0 should raise validation error."""
    with pytest.raises(ValueError):
        Signal(
            bot_id="dca_v1",
            timestamp=datetime.now(timezone.utc),
            action=Action.BUY,
            size=-0.5,
            confidence=0.8,
            reason="test",
        )


def test_signal_confidence_out_of_range_raises():
    """Signal confidence > 1.0 should raise validation error."""
    with pytest.raises(ValueError):
        Signal(
            bot_id="dca_v1",
            timestamp=datetime.now(timezone.utc),
            action=Action.BUY,
            size=0.5,
            confidence=1.5,
            reason="test",
        )


def test_signal_action_as_float():
    """Signal can use float action for continuous RL agents."""
    signal = Signal(
        bot_id="rl_v1",
        timestamp=datetime.now(timezone.utc),
        action=0.5,  # continuous action
        size=0.5,
        confidence=0.8,
        reason="RL signal",
    )
    assert signal.action == 0.5


def test_signal_action_float_out_of_range_raises():
    """Signal float action outside [-1.0, 1.0] should raise validation error."""
    with pytest.raises(ValueError):
        Signal(
            bot_id="rl_v1",
            timestamp=datetime.now(timezone.utc),
            action=1.5,  # outside range
            size=0.5,
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


def test_order_pending_status():
    order = Order(
        id="ord_002",
        signal_id="sig_002",
        exchange="paper",
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        status=OrderStatus.PENDING,
        price_target=51000.0,
        size=0.1,
        created_at=datetime.now(timezone.utc),
    )
    assert order.status == OrderStatus.PENDING


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


def test_trade_breakeven_not_profitable():
    """Trade with 0 PnL should not be considered profitable."""
    trade = Trade(
        id="trade_003",
        order_open_id="ord_005",
        order_close_id="ord_006",
        symbol="BTC/USDT",
        pnl_realized=0.0,
        pnl_pct=0.0,
        duration_seconds=1800.0,
    )
    assert trade.is_profitable is False
