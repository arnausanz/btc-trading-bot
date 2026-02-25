# tests/unit/test_paper_exchange.py
from datetime import datetime, timezone
from exchanges.paper import PaperExchange
from core.models import Signal, Action


def make_exchange() -> PaperExchange:
    return PaperExchange(config_path="config/exchanges/paper.yaml")


def make_signal(action: Action, size: float = 1.0) -> Signal:
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


def test_sell_order():
    exchange = make_exchange()
    exchange.set_current_price(50_000.0)
    exchange.send_order(make_signal(Action.BUY, size=1.0))
    order = exchange.send_order(make_signal(Action.SELL, size=1.0))
    assert order.status.value == "filled"
    assert exchange.get_balance("BTC") == 0.0


def test_hold_order_cancelled():
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