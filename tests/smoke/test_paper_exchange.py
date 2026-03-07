# tests/smoke/test_paper_exchange.py
"""Test paper exchange simulations without real data."""
from datetime import datetime, timezone
from exchanges.paper import PaperExchange
from core.models import Signal, Action


def make_exchange() -> PaperExchange:
    """Factory for creating a PaperExchange instance."""
    return PaperExchange(config_path="config/exchanges/paper.yaml")


def make_signal(action: Action, size: float = 1.0) -> Signal:
    """Factory for creating a Signal."""
    return Signal(
        bot_id="test_bot",
        timestamp=datetime.now(timezone.utc),
        action=action,
        size=size,
        confidence=1.0,
        reason="test",
    )


def test_initial_portfolio():
    exchange = make_exchange()
    assert exchange.get_balance("USDT") == 10_000.0
    assert exchange.get_balance("BTC") == 0.0


def test_buy_order():
    exchange = make_exchange()
    exchange.set_current_price(50_000.0)
    order = exchange.send_order(make_signal(Action.BUY, size=1.0))
    assert order.status.value == "filled"
    assert exchange.get_balance("BTC") > 0
    assert exchange.get_balance("USDT") < 10_000.0


def test_buy_partial():
    """Test buying with size < 1.0 (partial capital)."""
    exchange = make_exchange()
    exchange.set_current_price(50_000.0)
    initial_usdt = exchange.get_balance("USDT")
    order = exchange.send_order(make_signal(Action.BUY, size=0.5))
    assert order.status.value == "filled"
    assert exchange.get_balance("BTC") > 0
    assert exchange.get_balance("USDT") < initial_usdt
    # Should have about 50% less USDT
    assert exchange.get_balance("USDT") > initial_usdt * 0.4


def test_sell_order():
    exchange = make_exchange()
    exchange.set_current_price(50_000.0)
    exchange.send_order(make_signal(Action.BUY, size=1.0))
    order = exchange.send_order(make_signal(Action.SELL, size=1.0))
    assert order.status.value == "filled"
    assert exchange.get_balance("BTC") == 0.0


def test_hold_order_cancelled():
    """HOLD action should result in CANCELLED order."""
    exchange = make_exchange()
    exchange.set_current_price(50_000.0)
    order = exchange.send_order(make_signal(Action.HOLD, size=0.0))
    assert order.status.value == "cancelled"


def test_portfolio_value():
    exchange = make_exchange()
    exchange.set_current_price(50_000.0)
    exchange.send_order(make_signal(Action.BUY, size=0.5))
    value = exchange.get_portfolio_value()
    assert 9_900.0 < value < 10_100.0


def test_multiple_buys():
    """Test multiple buy orders."""
    exchange = make_exchange()
    exchange.set_current_price(50_000.0)
    exchange.send_order(make_signal(Action.BUY, size=0.3))
    exchange.send_order(make_signal(Action.BUY, size=0.3))
    btc_balance = exchange.get_balance("BTC")
    assert btc_balance > 0
    usdt_balance = exchange.get_balance("USDT")
    assert usdt_balance < 10_000.0


def test_buy_then_sell_partial():
    """Test buying and then selling partially."""
    exchange = make_exchange()
    exchange.set_current_price(50_000.0)
    exchange.send_order(make_signal(Action.BUY, size=1.0))
    initial_btc = exchange.get_balance("BTC")
    exchange.send_order(make_signal(Action.SELL, size=0.5))
    final_btc = exchange.get_balance("BTC")
    assert final_btc < initial_btc
    assert exchange.get_balance("USDT") > 0
