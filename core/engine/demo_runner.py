# core/engine/demo_runner.py
import logging
import time
import yaml
from datetime import datetime, timezone
from data.processing.technical import compute_features
from data.observation.builder import ObservationBuilder
from exchanges.paper import PaperExchange
from core.interfaces.base_bot import BaseBot
from core.models import Signal, Action, Order

logger = logging.getLogger(__name__)


class DemoRunner:
    """
    Executa múltiples bots en paral·lel amb dades reals en temps real.
    Usa PaperExchange — cap diners reals involucrats.
    Cada bot té el seu propi exchange independent (portfolios separats).
    """

    def __init__(self, config_path: str = "config/demo.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.symbol = self.config["demo"]["symbol"]
        self.timeframe = self.config["demo"]["timeframe"]
        self.update_interval = self.config["demo"]["update_interval_seconds"]

        self.bots: list[BaseBot] = []
        self.exchanges: dict[str, PaperExchange] = {}
        self.builders: dict[str, ObservationBuilder] = {}
        self.histories: dict[str, list[dict]] = {}

        self._load_bots()

    def _load_bots(self) -> None:
        """Carrega els bots definits al config."""
        from bots.classical.trend_bot import TrendBot
        from bots.classical.dca_bot import DCABot
        from bots.classical.grid_bot import GridBot
        from bots.classical.hold_bot import HoldBot

        bot_classes = {
            "trend.yaml": TrendBot,
            "dca.yaml": DCABot,
            "grid.yaml": GridBot,
            "hold.yaml": HoldBot,
        }

        for bot_config in self.config["bots"]:
            if not bot_config["enabled"]:
                continue

            config_path = bot_config["config_path"]
            bot_name = config_path.split("/")[-1]
            bot_class = bot_classes.get(bot_name)

            if not bot_class:
                logger.warning(f"Bot desconegut: {bot_name}")
                continue

            bot = bot_class(config_path=config_path)
            exchange = PaperExchange(
                config_path=self.config["exchange"]["config_path"]
            )
            builder = ObservationBuilder()

            self.bots.append(bot)
            self.exchanges[bot.bot_id] = exchange
            self.builders[bot.bot_id] = builder
            self.histories[bot.bot_id] = []

            logger.info(f"Bot carregat: {bot.bot_id}")

    def _fetch_latest_price(self) -> float:
        """Obté el preu actual de BTC via ccxt."""
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
        ticker = exchange.fetch_ticker(self.symbol)
        return float(ticker["last"])

    def _process_tick(self, current_price: float) -> None:
        """Processa un tick per a tots els bots."""
        for bot in self.bots:
            bot_id = bot.bot_id
            exchange = self.exchanges[bot_id]
            builder = self.builders[bot_id]

            try:
                schema = bot.observation_schema()

                # Carrega dades si és el primer tick
                if not builder._cache:
                    builder.load(schema=schema, symbol=self.symbol)
                    bot.on_start()

                df = builder.get_dataframe(
                    symbol=self.symbol,
                    timeframe=self.timeframe
                )
                index = len(df) - 1

                exchange.set_current_price(current_price)
                observation = builder.build(
                    schema=schema,
                    symbol=self.symbol,
                    index=index,
                )
                observation["portfolio"] = exchange.get_portfolio()

                signal = bot.on_observation(observation)
                order = exchange.send_order(signal)

                portfolio_value = exchange.get_portfolio_value()

                tick_data = {
                    "timestamp": datetime.now(timezone.utc),
                    "price": current_price,
                    "signal": signal.action,
                    "order_status": order.status,
                    "portfolio_value": portfolio_value,
                    "reason": signal.reason,
                }
                self.histories[bot_id].append(tick_data)

                if signal.action != Action.HOLD:
                    logger.info(
                        f"[{bot_id}] {signal.action.value.upper()} @ {current_price:.2f} | "
                        f"Portfolio: {portfolio_value:.2f} USDT | "
                        f"Motiu: {signal.reason}"
                    )

            except Exception as e:
                logger.error(f"Error processant tick per {bot_id}: {e}")

    def get_status(self) -> dict:
        """Retorna l'estat actual de tots els bots."""
        status = {}
        for bot in self.bots:
            bot_id = bot.bot_id
            exchange = self.exchanges[bot_id]
            history = self.histories[bot_id]

            status[bot_id] = {
                "portfolio_value": exchange.get_portfolio_value(),
                "btc_balance": exchange.get_balance("BTC"),
                "usdt_balance": exchange.get_balance("USDT"),
                "total_ticks": len(history),
                "last_signal": history[-1]["signal"] if history else None,
                "last_update": history[-1]["timestamp"] if history else None,
            }
        return status

    def run(self) -> None:
        """Loop principal del demo — corre indefinidament."""
        logger.info(f"Demo Runner iniciat: {[b.bot_id for b in self.bots]}")
        logger.info(f"Interval d'actualització: {self.update_interval}s")

        while True:
            try:
                current_price = self._fetch_latest_price()
                logger.info(f"Preu actual BTC: {current_price:.2f} USDT")
                self._process_tick(current_price=current_price)

                # Mostra estat cada tick
                status = self.get_status()
                for bot_id, s in status.items():
                    logger.info(
                        f"  [{bot_id}] Portfolio: {s['portfolio_value']:.2f} USDT | "
                        f"BTC: {s['btc_balance']:.6f} | "
                        f"USDT: {s['usdt_balance']:.2f}"
                    )

                time.sleep(self.update_interval)

            except KeyboardInterrupt:
                logger.info("Demo aturat per l'usuari.")
                break
            except Exception as e:
                logger.error(f"Error al loop principal: {e}")
                time.sleep(10)