# exchanges/paper.py
import uuid
import logging
from datetime import datetime, timezone
from core.interfaces.base_exchange import BaseExchange
from core.models import Signal, Order, Candle, Action, OrderSide, OrderStatus
import yaml
import os

logger = logging.getLogger(__name__)


class PaperExchange(BaseExchange):
    """
    Simulador d'exchange per a paper trading i backtesting.
    Simula fills, fees i slippage sense connectar-se a cap exchange real.
    """

    def __init__(self, config_path: str = "config/exchanges/paper.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.fee_rate = config["fee_rate"]
        self.slippage_rate = config["slippage_rate"]
        self._portfolio: dict[str, float] = {
            "USDT": config["initial_capital"],
            "BTC": 0.0,
        }
        self._current_price: float = 0.0
        self._orders: list[Order] = []
        logger.info(f"PaperExchange inicialitzat amb {config['initial_capital']} USDT")

    def set_current_price(self, price: float) -> None:
        """El Runner actualitza el preu actual a cada tick."""
        self._current_price = price

    def get_candles(self, symbol: str, timeframe: str, limit: int = 500) -> list[Candle]:
        """En paper trading les candles venen de la DB, no de l'exchange."""
        return []

    def send_order(self, signal: Signal) -> Order:
        """
        Simula l'execució d'un Signal.
        Aplica slippage i fees, actualitza el portfolio.
        """
        if self._current_price <= 0:
            raise ValueError("El preu actual no està configurat. Crida set_current_price primer.")

        now = datetime.now(timezone.utc)

        if signal.action == Action.HOLD:
            return Order(
                id=str(uuid.uuid4()),
                signal_id=signal.bot_id,
                exchange="paper",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                status=OrderStatus.CANCELLED,
                price_target=self._current_price,
                size=0.0,
                created_at=now,
            )

        # Aplica slippage: comprar és més car, vendre és més barat
        if signal.action == Action.BUY:
            fill_price = self._current_price * (1 + self.slippage_rate)
            side = OrderSide.BUY
            capital_to_use = self._portfolio["USDT"] * signal.size
            btc_amount = capital_to_use / fill_price
            fees = capital_to_use * self.fee_rate

            if capital_to_use > self._portfolio["USDT"]:
                logger.warning("Capital insuficient per executar l'ordre")
                return Order(
                    id=str(uuid.uuid4()),
                    signal_id=signal.bot_id,
                    exchange="paper",
                    symbol="BTC/USDT",
                    side=side,
                    status=OrderStatus.FAILED,
                    price_target=self._current_price,
                    size=0.0,
                    created_at=now,
                )

            self._portfolio["USDT"] -= (capital_to_use + fees)
            self._portfolio["BTC"] += btc_amount

        else:  # SELL
            fill_price = self._current_price * (1 - self.slippage_rate)
            side = OrderSide.SELL
            btc_amount = self._portfolio["BTC"] * signal.size
            capital_received = btc_amount * fill_price
            fees = capital_received * self.fee_rate

            if btc_amount > self._portfolio["BTC"]:
                logger.warning("BTC insuficient per executar l'ordre")
                return Order(
                    id=str(uuid.uuid4()),
                    signal_id=signal.bot_id,
                    exchange="paper",
                    symbol="BTC/USDT",
                    side=side,
                    status=OrderStatus.FAILED,
                    price_target=self._current_price,
                    size=0.0,
                    created_at=now,
                )

            self._portfolio["BTC"] -= btc_amount
            self._portfolio["USDT"] += (capital_received - fees)

        order = Order(
            id=str(uuid.uuid4()),
            signal_id=signal.bot_id,
            exchange="paper",
            symbol="BTC/USDT",
            side=side,
            status=OrderStatus.FILLED,
            price_target=self._current_price,
            price_filled=fill_price,
            size=btc_amount,
            fees=fees,
            created_at=now,
            filled_at=now,
        )

        self._orders.append(order)
        logger.debug(  # canvia INFO per DEBUG
            f"Ordre executada: {side.value} {btc_amount:.6f} BTC "
            f"@ {fill_price:.2f} USDT (fees: {fees:.4f} USDT)"
        )
        return order

    def get_portfolio(self) -> dict:
        return self._portfolio.copy()

    def get_balance(self, currency: str) -> float:
        return self._portfolio.get(currency, 0.0)

    def get_portfolio_value(self) -> float:
        """Valor total del portfolio en USDT al preu actual."""
        return self._portfolio["USDT"] + self._portfolio["BTC"] * self._current_price

    def restore_state(self, usdt_balance: float, btc_balance: float) -> None:
        """Restaura el portfolio des d'un estat guardat a la DB."""
        self._portfolio["USDT"] = usdt_balance
        self._portfolio["BTC"] = btc_balance
        logger.info(
            f"Estat restaurat: {usdt_balance:.2f} USDT | {btc_balance:.6f} BTC"
        )