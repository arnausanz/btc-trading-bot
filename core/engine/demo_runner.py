# core/engine/demo_runner.py
import logging
import os
import time
import yaml
import zoneinfo
from datetime import datetime, timezone
from data.observation.builder import ObservationBuilder
from exchanges.paper import PaperExchange
from core.interfaces.base_bot import BaseBot
from core.models import Action
from core.db.demo_repository import DemoRepository

logger = logging.getLogger(__name__)

TZ = zoneinfo.ZoneInfo("Europe/Madrid")

_ML_TYPES = {"random_forest", "xgboost", "lightgbm", "catboost", "gru", "patchtst"}
_RL_TYPES = {"ppo", "sac"}


def _get_best_config_path(base_config_path: str) -> str:
    """
    Check if an optimized version of the config exists and return the best path.

    For base_config_path like "config/bots/dca.yaml", checks for "config/bots/dca_optimized.yaml".
    For models like "config/training/rf_optimized.yaml", checks for optimized variant.

    Args:
        base_config_path: The original config path from demo.yaml

    Returns:
        The optimized config path if it exists, otherwise the base config path
    """
    dir_name = os.path.dirname(base_config_path)
    file_name = os.path.basename(base_config_path)
    name_without_ext = file_name.replace(".yaml", "")
    optimized_path = os.path.join(dir_name, f"{name_without_ext}_optimized.yaml")

    if os.path.exists(optimized_path):
        logger.info(f"Using optimized config: {optimized_path}")
        return optimized_path

    logger.info(f"Using default config: {base_config_path}")
    return base_config_path


def _load_bot_from_config(config_path: str) -> BaseBot:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_type = config.get("model_type", "")
    if model_type in _ML_TYPES:
        from bots.ml.ml_bot import MLBot
        return MLBot(config_path=config_path)
    if model_type in _RL_TYPES:
        from bots.rl.rl_bot import RLBot
        return RLBot(config_path=config_path)
    bot_name = config_path.split("/")[-1]
    from bots.classical.trend_bot import TrendBot
    from bots.classical.dca_bot   import DCABot
    from bots.classical.grid_bot  import GridBot
    from bots.classical.hold_bot  import HoldBot
    _CLASSICAL = {"trend.yaml": TrendBot, "dca.yaml": DCABot, "grid.yaml": GridBot, "hold.yaml": HoldBot}
    bot_class = _CLASSICAL.get(bot_name)
    if bot_class:
        return bot_class(config_path=config_path)
    raise ValueError(f"Cannot detect bot type: {config_path}")


class DemoRunner:
    def __init__(self, config_path: str = "config/demo.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.symbol          = self.config["demo"]["symbol"]
        self.timeframe       = self.config["demo"]["timeframe"]
        self.update_interval = self.config["demo"]["update_interval_seconds"]
        self.bots:      list[BaseBot]                = []
        self.exchanges: dict[str, PaperExchange]     = {}
        self.builders:  dict[str, ObservationBuilder] = {}
        self.histories: dict[str, list[dict]]        = {}
        self._max_portfolio:      dict[str, float]   = {}
        self._last_daily_summary: datetime | None    = None
        self._last_candle_hour:   dict[str, int]     = {}
        self.repo = DemoRepository()
        if self.config["telegram"]["enabled"]:
            from monitoring.telegram_notifier import TelegramNotifier
            self.notifier = TelegramNotifier()
        else:
            self.notifier = None
        self._load_bots()
        if self.notifier:
            self.notifier.set_status_fn(self.get_status)
            self.notifier.set_trades_fn(self._get_all_trades)
            self.notifier.start_listener()

    def _load_bots(self) -> None:
        for bot_config in self.config["bots"]:
            if not bot_config.get("enabled", True):
                continue
            base_config_path = bot_config["config_path"]
            # Check for optimized config version
            config_path = _get_best_config_path(base_config_path)
            try:
                bot = _load_bot_from_config(config_path)
            except Exception as e:
                logger.error(f"Cannot load bot {config_path}: {e}")
                continue
            exchange = PaperExchange(config_path=self.config["exchange"]["config_path"])
            builder  = ObservationBuilder()
            last_state = self.repo.get_last_state(bot_id=bot.bot_id)
            if last_state:
                exchange.restore_state(usdt_balance=last_state["usdt_balance"], btc_balance=last_state["btc_balance"])
                if hasattr(bot, "_in_position"):
                    bot._in_position = last_state["btc_balance"] > 1e-6
                if hasattr(bot, "_bought"):
                    bot._bought = last_state["btc_balance"] > 1e-6
                logger.info(f"Bot restored: {bot.bot_id} | Portfolio: {last_state['portfolio_value']:.2f} USDT | {last_state['timestamp'].astimezone(TZ).strftime('%Y-%m-%d %H:%M')}")
            else:
                logger.info(f"New bot: {bot.bot_id} | Initial capital: 10,000 USDT")
            self.bots.append(bot)
            self.exchanges[bot.bot_id]         = exchange
            self.builders[bot.bot_id]          = builder
            self.histories[bot.bot_id]         = []
            self._last_candle_hour[bot.bot_id] = -1

    def _fetch_latest_price(self) -> float:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
        return float(exchange.fetch_ticker(self.symbol)["last"])

    @staticmethod
    def _current_hour_bucket() -> int:
        return int(datetime.now(timezone.utc).timestamp()) // 3600

    def _process_tick(self, current_price: float) -> None:
        current_hour = self._current_hour_bucket()
        now          = datetime.now(timezone.utc)
        for bot in self.bots:
            bot_id   = bot.bot_id
            exchange = self.exchanges[bot_id]
            builder  = self.builders[bot_id]
            try:
                schema = bot.observation_schema()
                if not builder._cache:
                    builder.load(schema=schema, symbol=self.symbol)
                    bot.on_start()
                exchange.set_current_price(current_price)
                is_new_candle = (current_hour != self._last_candle_hour[bot_id])
                if is_new_candle:
                    self._last_candle_hour[bot_id] = current_hour
                    df    = builder.get_dataframe(symbol=self.symbol, timeframe=self.timeframe)
                    index = len(df) - 1
                    observation              = builder.build(schema=schema, symbol=self.symbol, index=index)
                    observation["portfolio"] = exchange.get_portfolio()
                    signal = bot.on_observation(observation)
                    order  = exchange.send_order(signal)
                    if signal.action.value != "hold" and order.status.value == "filled":
                        pv = exchange.get_portfolio_value()
                        self.repo.save_trade(bot_id=bot_id, timestamp=now, action=signal.action.value,
                            price=current_price, size_btc=order.size, size_usdt=order.size * current_price,
                            fees=order.fees, portfolio_value=pv, reason=signal.reason)
                        if self.notifier:
                            self.notifier.notify_trade(bot_id=bot_id, action=signal.action.value,
                                price=current_price, portfolio_value=pv, reason=signal.reason)
                        logger.info(f"[{bot_id}] {signal.action.value.upper()} @ {current_price:.2f} | Portfolio: {exchange.get_portfolio_value():.2f} USDT")
                    action_str = signal.action.value
                else:
                    action_str = "hold"
                portfolio_value = exchange.get_portfolio_value()
                self.repo.save_tick(bot_id=bot_id, timestamp=now, price=current_price, action=action_str,
                    portfolio_value=portfolio_value, usdt_balance=exchange.get_balance("USDT"),
                    btc_balance=exchange.get_balance("BTC"),
                    reason="new candle" if is_new_candle else "between candles")
                self.histories[bot_id].append({"timestamp": now, "price": current_price,
                    "signal": action_str, "portfolio_value": portfolio_value, "new_candle": is_new_candle})
            except Exception as e:
                logger.error(f"Error processing tick for {bot_id}: {e}", exc_info=True)

    def get_status(self) -> dict:
        status = {}
        for bot in self.bots:
            bot_id  = bot.bot_id; exchange = self.exchanges[bot_id]; history = self.histories[bot_id]
            status[bot_id] = {"portfolio_value": exchange.get_portfolio_value(),
                "btc_balance": exchange.get_balance("BTC"), "usdt_balance": exchange.get_balance("USDT"),
                "total_ticks": len(history), "last_signal": history[-1]["signal"] if history else None,
                "last_update": history[-1]["timestamp"] if history else None,
                "new_candle":  history[-1].get("new_candle") if history else None}
        return status

    def _get_all_trades(self) -> list[dict]:
        all_trades = []
        for bot in self.bots:
            trades = self.repo.get_trades(bot_id=bot.bot_id)
            for t in trades: t["bot_id"] = bot.bot_id
            all_trades.extend(trades)
        all_trades.sort(key=lambda x: x["timestamp"])
        return all_trades

    def run(self) -> None:
        logger.info(f"Demo Runner v2 started: {[b.bot_id for b in self.bots]}")
        logger.info("Candle sync active — on_observation() called once per hour (candle closed)")
        if self.notifier:
            self.notifier.notify_startup([b.bot_id for b in self.bots])
        logger.info(f"Monitoring interval: {self.update_interval}s")
        while True:
            try:
                current_price = self._fetch_latest_price()
                logger.info(f"Current BTC price: {current_price:.2f} USDT")
                self._process_tick(current_price=current_price)
                status = self.get_status()
                for bot_id, s in status.items():
                    logger.info(f"  [{bot_id}] Portfolio: {s['portfolio_value']:.2f} USDT | BTC: {s['btc_balance']:.6f} | USDT: {s['usdt_balance']:.2f}")
                ticks_per_hour = max(1, 3600 // self.update_interval)
                if self.histories and len(list(self.histories.values())[0]) % ticks_per_hour == 0:
                    if self.notifier: self.notifier.notify_status(self.get_status())
                for bot in self.bots:
                    bot_id = bot.bot_id; pv = self.exchanges[bot_id].get_portfolio_value()
                    max_pv = self._max_portfolio.get(bot_id, 10_000)
                    self._max_portfolio[bot_id] = max(max_pv, pv)
                    drawdown = (pv - max_pv) / max_pv * 100
                    if drawdown < -10 and self.notifier:
                        self.notifier.notify_drawdown_alert(bot_id=bot_id, drawdown_pct=drawdown, portfolio_value=pv)
                now = datetime.now(timezone.utc)
                if (self.notifier and now.hour == 8 and now.minute == 0 and
                    (self._last_daily_summary is None or (now - self._last_daily_summary).total_seconds() > 3600)):
                    self.notifier.notify_daily_summary(self.get_status())
                    self._last_daily_summary = now
                time.sleep(self.update_interval)
            except KeyboardInterrupt:
                logger.info("Demo stopped by user."); break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(10)
