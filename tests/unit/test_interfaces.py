# tests/unit/test_interfaces.py
from datetime import datetime, timezone
from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.interfaces.base_exchange import BaseExchange
from core.models import Signal, Action, Order, OrderSide, OrderStatus, Candle


class DummyBot(BaseBot):
    """Bot mínim per testejar la interfície."""

    def observation_schema(self) -> ObservationSchema:
        return ObservationSchema(
            features=["close", "rsi_14"],
            timeframes=["1h"],
            lookback=50,
        )

    def on_observation(self, observation: dict) -> Signal:
        return Signal(
            bot_id=self.bot_id,
            timestamp=datetime.now(timezone.utc),
            action=Action.HOLD,
            size=0.0,
            confidence=1.0,
            reason="test signal",
        )


class DummyExchange(BaseExchange):
    """Exchange mínim per testejar la interfície."""

    def get_candles(self, symbol, timeframe, limit=500) -> list[Candle]:
        return []

    def send_order(self, signal: Signal) -> Order:
        return Order(
            id="ord_test",
            signal_id="sig_test",
            exchange="dummy",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            status=OrderStatus.FILLED,
            price_target=50000.0,
            price_filled=50000.0,
            size=0.1,
            created_at=datetime.now(timezone.utc),
        )

    def get_portfolio(self) -> dict:
        return {"BTC": 0.0, "USDT": 10000.0}

    def get_balance(self, currency: str) -> float:
        return {"BTC": 0.0, "USDT": 10000.0}.get(currency, 0.0)


def test_bot_returns_signal():
    bot = DummyBot(bot_id="dummy_v1", config={})
    schema = bot.observation_schema()
    assert "close" in schema.features
    assert schema.lookback == 50

    signal = bot.on_observation({})
    assert signal.action == Action.HOLD


def test_exchange_returns_portfolio():
    exchange = DummyExchange()
    portfolio = exchange.get_portfolio()
    assert "USDT" in portfolio
    assert portfolio["USDT"] == 10000.0