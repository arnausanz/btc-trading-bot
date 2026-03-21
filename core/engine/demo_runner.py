# core/engine/demo_runner.py
import logging
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

_ML_TYPES = {"random_forest", "xgboost", "lightgbm", "catboost", "gru", "patchtst", "tft"}
_RL_TYPES = {
    "ppo", "sac",
    "ppo_professional", "sac_professional",
    "td3_professional", "td3_multiframe",
}

# Maps timeframe string → seconds per candle.
# Used to compute per-bot candle buckets so bots with different timeframes
# (e.g. GateBot at 4H vs classic bots at 1H) detect new candles correctly.
_TF_SECONDS: dict[str, int] = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
    "8h": 28800, "12h": 43200, "1d": 86400,
}


def _get_best_config_path(base_config_path: str) -> str:
    """
    Returns the config path to use for loading a bot.

    best_params from Optuna are stored inline in the base YAML (section
    ``best_params``). apply_best_params() is called by each bot/trainer —
    no separate *_optimized.yaml file is needed.
    """
    logger.info(f"Using config: {base_config_path}")
    return base_config_path


def _load_bot_from_config(config_path: str) -> BaseBot:
    """
    Load a bot from its YAML config.

    Detection order:
    1. model_type in _ML_TYPES  → MLBot
    2. model_type in _RL_TYPES  → RLBot
    3. 'module' + 'class_name' present  → generic dynamic load
       (classical bots, EnsembleBot, any future bot type)

    Adding a new bot type never requires editing this file — just add
    'module' and 'class_name' to its YAML.
    """
    import importlib
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_type = config.get("model_type", "")

    if model_type in _ML_TYPES:
        from bots.ml.ml_bot import MLBot
        return MLBot(config_path=config_path)

    if model_type in _RL_TYPES:
        from bots.rl.rl_bot import RLBot
        return RLBot(config_path=config_path)

    module_path = config.get("module")
    class_name  = config.get("class_name")
    if module_path and class_name:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls(config_path=config_path)

    raise ValueError(
        f"Cannot detect bot type for '{config_path}'. "
        "The YAML must have 'module' + 'class_name' or a recognised model_type."
    )


class DemoRunner:
    def __init__(self, config_path: str = "config/demo.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.symbol          = self.config["demo"]["symbol"]
        self.timeframe       = self.config["demo"]["timeframe"]
        self.update_interval = self.config["demo"]["update_interval_seconds"]

        # Alert thresholds from config (with sensible defaults)
        tg_cfg = self.config.get("telegram", {})
        self._inactivity_hours   = float(tg_cfg.get("inactivity_hours",   4.0))
        self._drawdown_alert_pct = float(tg_cfg.get("drawdown_alert_pct", -10.0))
        self._data_stale_min     = float(tg_cfg.get("data_stale_minutes", 90.0))

        self.bots:      list[BaseBot]                 = []
        self.exchanges: dict[str, PaperExchange]      = {}
        self.builders:  dict[str, ObservationBuilder] = {}
        self.histories: dict[str, list[dict]]         = {}

        self._max_portfolio:         dict[str, float]            = {}
        self._last_daily_summary:    datetime | None             = None
        # Per-bot candle bucket (replaces _last_candle_hour).
        # Bucket size depends on the bot's primary timeframe (_TF_SECONDS).
        # All existing bots use "1h" → bucket = epoch // 3600 (identical to before).
        self._last_candle_bucket:    dict[str, int]              = {}
        self._last_trade_time:       dict[str, datetime | None]  = {}  # inactivity tracking
        self._inactivity_alerted:    dict[str, bool]             = {}  # avoid repeat alerts
        self._last_data_stale_alert: datetime | None             = None  # avoid spam

        self.repo = DemoRepository()

        if tg_cfg.get("enabled", False):
            from monitoring.telegram_notifier import TelegramNotifier
            self.notifier = TelegramNotifier(
                inactivity_hours=self._inactivity_hours,
                drawdown_alert_pct=self._drawdown_alert_pct,
                data_stale_minutes=self._data_stale_min,
            )
        else:
            self.notifier = None

        self._load_bots()

        if self.notifier:
            self.notifier.set_status_fn(self.get_status)
            self.notifier.set_trades_fn(self._get_all_trades)
            self.notifier.set_all_histories_fn(self._get_all_histories)
            self.notifier.set_health_fn(self._get_health)
            self.notifier.start_listener()

    # ──────────────────────────────────────────────────────────────────────
    # Bot loading
    # ──────────────────────────────────────────────────────────────────────

    def _load_bots(self) -> None:
        for bot_config in self.config["bots"]:
            if not bot_config.get("enabled", True):
                continue
            base_config_path = bot_config["config_path"]
            config_path      = _get_best_config_path(base_config_path)
            try:
                bot = _load_bot_from_config(config_path)
            except Exception as e:
                logger.error(f"Cannot load bot {config_path}: {e}")
                continue

            exchange   = PaperExchange(config_path=self.config["exchange"]["config_path"])
            builder    = ObservationBuilder()
            last_state = self.repo.get_last_state(bot_id=bot.bot_id)

            if last_state:
                exchange.restore_state(
                    usdt_balance=last_state["usdt_balance"],
                    btc_balance=last_state["btc_balance"],
                )
                if hasattr(bot, "_in_position"):
                    bot._in_position = last_state["btc_balance"] > 1e-6
                if hasattr(bot, "_bought"):
                    bot._bought = last_state["btc_balance"] > 1e-6
                logger.info(
                    f"Bot restored: {bot.bot_id} | "
                    f"Portfolio: {last_state['portfolio_value']:.2f} USDT | "
                    f"{last_state['timestamp'].astimezone(TZ).strftime('%Y-%m-%d %H:%M')}"
                )
            else:
                logger.info(f"New bot: {bot.bot_id} | Initial capital: 10,000 USDT")

            self.bots.append(bot)
            self.exchanges[bot.bot_id]           = exchange
            self.builders[bot.bot_id]            = builder
            self.histories[bot.bot_id]           = []
            self._last_candle_bucket[bot.bot_id] = -1
            self._last_trade_time[bot.bot_id]    = None
            self._inactivity_alerted[bot.bot_id] = False

    # ──────────────────────────────────────────────────────────────────────
    # Price fetching
    # ──────────────────────────────────────────────────────────────────────

    def _fetch_latest_price(self) -> float:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
        return float(exchange.fetch_ticker(self.symbol)["last"])

    @staticmethod
    def _candle_bucket(tf_seconds: int) -> int:
        """Returns the current candle bucket for the given timeframe (epoch // tf_secs)."""
        return int(datetime.now(timezone.utc).timestamp()) // tf_seconds

    # ──────────────────────────────────────────────────────────────────────
    # Tick processing
    # ──────────────────────────────────────────────────────────────────────

    def _process_tick(self, current_price: float) -> None:
        now = datetime.now(timezone.utc)

        for bot in self.bots:
            bot_id   = bot.bot_id
            exchange = self.exchanges[bot_id]
            builder  = self.builders[bot_id]
            try:
                schema = bot.observation_schema()

                # Determine candle bucket for this bot's primary timeframe.
                # Bots with 1h primary TF → bucket = epoch//3600 (same as before).
                # GateBot with 4h primary TF → bucket = epoch//14400.
                primary_tf = schema.timeframes[0]
                tf_secs    = _TF_SECONDS.get(primary_tf, 3600)
                cur_bucket = self._candle_bucket(tf_secs)

                is_first_load = not builder._cache
                if is_first_load:
                    builder.load(schema=schema, symbol=self.symbol)
                    bot.on_start()
                    # Re-apply restored state: on_start() resets _in_position/_bought
                    # but _load_bots() already set the correct values from DB.
                    # We re-apply here so the bot knows its true position on restart.
                    last_state = self.repo.get_last_state(bot_id=bot_id)
                    if last_state:
                        if hasattr(bot, "_in_position"):
                            bot._in_position = last_state["btc_balance"] > 1e-6
                        if hasattr(bot, "_bought"):
                            bot._bought = last_state["btc_balance"] > 1e-6
                        logger.info(
                            f"[{bot_id}] Position state re-applied after on_start(): "
                            f"in_position={last_state['btc_balance'] > 1e-6}"
                        )

                exchange.set_current_price(current_price)
                is_new_candle = (cur_bucket != self._last_candle_bucket[bot_id])

                if is_new_candle:
                    self._last_candle_bucket[bot_id] = cur_bucket
                    # Reload data from DB so the bot sees freshly closed candles.
                    # Without this, the builder cache would be stale from startup
                    # and the bot would always predict on the same old data.
                    builder._cache.clear()
                    builder.load(schema=schema, symbol=self.symbol)
                    df    = builder.get_dataframe(symbol=self.symbol, timeframe=primary_tf)
                    index = len(df) - 1
                    observation              = builder.build(schema=schema, symbol=self.symbol, index=index)
                    observation["portfolio"] = exchange.get_portfolio()
                    signal = bot.on_observation(observation)
                    order  = exchange.send_order(signal)

                    if signal.action.value != "hold" and order.status.value == "filled":
                        pv = exchange.get_portfolio_value()
                        self.repo.save_trade(
                            bot_id=bot_id,
                            timestamp=now,
                            action=signal.action.value,
                            price=current_price,
                            size_btc=order.size,
                            size_usdt=order.size * current_price,
                            fees=order.fees,
                            portfolio_value=pv,
                            reason=signal.reason,
                            confidence=signal.confidence,     # ← new field
                        )
                        # Update inactivity tracking
                        self._last_trade_time[bot_id]    = now
                        self._inactivity_alerted[bot_id] = False  # reset alert flag

                        if self.notifier:
                            self.notifier.notify_trade(
                                bot_id=bot_id,
                                action=signal.action.value,
                                price=current_price,
                                portfolio_value=pv,
                                reason=signal.reason,
                                confidence=signal.confidence,  # ← new field
                                size_btc=order.size,           # ← new field
                            )
                        logger.info(
                            f"[{bot_id}] {signal.action.value.upper()} @ {current_price:.2f} | "
                            f"Portfolio: {exchange.get_portfolio_value():.2f} USDT"
                        )
                    action_str = signal.action.value
                else:
                    action_str = "hold"

                portfolio_value = exchange.get_portfolio_value()
                self.repo.save_tick(
                    bot_id=bot_id,
                    timestamp=now,
                    price=current_price,
                    action=action_str,
                    portfolio_value=portfolio_value,
                    usdt_balance=exchange.get_balance("USDT"),
                    btc_balance=exchange.get_balance("BTC"),
                    reason="new candle" if is_new_candle else "between candles",
                )
                self.histories[bot_id].append({
                    "timestamp":       now,
                    "price":           current_price,
                    "signal":          action_str,
                    "portfolio_value": portfolio_value,
                    "new_candle":      is_new_candle,
                })
            except Exception as e:
                logger.error(f"Error processing tick for {bot_id}: {e}", exc_info=True)
                if self.notifier:
                    self.notifier.notify_error(bot_id, str(e))

    # ──────────────────────────────────────────────────────────────────────
    # Data functions injected into TelegramNotifier
    # ──────────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        status = {}
        for bot in self.bots:
            bot_id   = bot.bot_id
            exchange = self.exchanges[bot_id]
            history  = self.histories[bot_id]
            status[bot_id] = {
                "portfolio_value": exchange.get_portfolio_value(),
                "btc_balance":     exchange.get_balance("BTC"),
                "usdt_balance":    exchange.get_balance("USDT"),
                "total_ticks":     len(history),
                "last_signal":     history[-1]["signal"]    if history else None,
                "last_update":     history[-1]["timestamp"] if history else None,
                "new_candle":      history[-1].get("new_candle") if history else None,
            }
        return status

    def _get_all_trades(self) -> list[dict]:
        """Returns all trades for all active bots, sorted by timestamp."""
        bot_ids = [b.bot_id for b in self.bots]
        return self.repo.get_all_trades(bot_ids)

    def _get_all_histories(self) -> dict[str, list[dict]]:
        """
        Returns portfolio history keyed by bot_id.

        Uses in-memory histories for speed when available; falls back to DB
        for richer data (price column needed for B&H alpha).
        """
        bot_ids = [b.bot_id for b in self.bots]
        return self.repo.get_all_portfolios(bot_ids)

    def _get_health(self) -> dict:
        """Assembles health info for /health command."""
        import time as _time

        # DB ping
        db_ok = self.repo.ping_db()

        # Last OHLCV candle timestamp
        last_candle = self.repo.get_candle_last_update(
            symbol=self.symbol, timeframe=self.timeframe
        )

        # Fear & Greed
        fg_entry = self.repo.get_fear_greed_last()
        fg = None
        if fg_entry:
            fg = {
                "timestamp":      fg_entry["timestamp"],
                "value":          fg_entry["value"],
                "classification": fg_entry["classification"],
            }

        # Binance API latency (simple round-trip measurement)
        binance_ms = None
        try:
            import ccxt
            exc = ccxt.binance({"enableRateLimit": False})
            t0  = _time.time()
            exc.fetch_ticker(self.symbol)
            binance_ms = (_time.time() - t0) * 1000
        except Exception:
            pass

        # MLflow: just check the tracking URI is reachable (best effort)
        mlflow_ok = False
        try:
            import mlflow
            mlflow.set_tracking_uri(mlflow.get_tracking_uri())
            mlflow_ok = True
        except Exception:
            pass

        # Minutes until next health check (1 hour)
        ticks_per_hour = max(1, 3600 // self.update_interval)
        ticks_done     = len(list(self.histories.values())[0]) if self.histories else 0
        ticks_to_next  = ticks_per_hour - (ticks_done % ticks_per_hour)
        next_check_min = round(ticks_to_next * self.update_interval / 60)

        return {
            "db_ok":              db_ok,
            "last_candle_ts":     last_candle,
            "fear_greed":         fg,
            "binance_latency_ms": binance_ms,
            "mlflow_ok":          mlflow_ok,
            "next_check_minutes": next_check_min,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Proactive alert checks (called in the main loop)
    # ──────────────────────────────────────────────────────────────────────

    def _check_inactivity_alerts(self, now: datetime) -> None:
        """Sends a push alert if a bot hasn't traded in > inactivity_hours."""
        if not self.notifier:
            return
        for bot in self.bots:
            bot_id    = bot.bot_id
            last_trade = self._last_trade_time.get(bot_id)
            if last_trade is None:
                continue  # No trades yet — don't alert
            age_h = (now - last_trade).total_seconds() / 3600
            if age_h > self._inactivity_hours and not self._inactivity_alerted.get(bot_id):
                self.notifier.notify_inactivity_alert(bot_id=bot_id, hours=age_h)
                self._inactivity_alerted[bot_id] = True

    def _check_drawdown_alerts(self, now: datetime) -> None:
        """Sends a push alert if a bot's max drawdown crosses the threshold."""
        if not self.notifier:
            return
        for bot in self.bots:
            bot_id = bot.bot_id
            pv     = self.exchanges[bot_id].get_portfolio_value()
            max_pv = self._max_portfolio.get(bot_id, 10_000.0)
            self._max_portfolio[bot_id] = max(max_pv, pv)
            drawdown = (pv - max_pv) / max_pv * 100.0
            if drawdown < self._drawdown_alert_pct:
                self.notifier.notify_drawdown_alert(
                    bot_id=bot_id, drawdown_pct=drawdown, portfolio_value=pv
                )

    def _check_data_stale_alert(self, now: datetime) -> None:
        """Sends a push alert if OHLCV candles are too old."""
        if not self.notifier:
            return
        # Only alert once per hour to avoid spam
        if (self._last_data_stale_alert and
                (now - self._last_data_stale_alert).total_seconds() < 3600):
            return
        last_candle = self.repo.get_candle_last_update(
            symbol=self.symbol, timeframe=self.timeframe
        )
        if last_candle:
            age_min = (now - last_candle).total_seconds() / 60
            if age_min > self._data_stale_min:
                self.notifier.notify_data_stale("OHLCV", age_min)
                self._last_data_stale_alert = now

    # ──────────────────────────────────────────────────────────────────────
    # Main run loop
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info(f"Demo Runner v2 started: {[b.bot_id for b in self.bots]}")
        logger.info("Candle sync: on_observation() called once per hour (candle closed)")
        if self.notifier:
            self.notifier.notify_startup([b.bot_id for b in self.bots])
        logger.info(f"Monitoring interval: {self.update_interval}s")

        while True:
            try:
                current_price = self._fetch_latest_price()
                logger.info(f"Current BTC price: {current_price:.2f} USDT")
                self._process_tick(current_price=current_price)

                # Log current portfolio state
                status = self.get_status()
                for bot_id, s in status.items():
                    logger.info(
                        f"  [{bot_id}] Portfolio: {s['portfolio_value']:.2f} USDT | "
                        f"BTC: {s['btc_balance']:.6f} | USDT: {s['usdt_balance']:.2f}"
                    )

                now = datetime.now(timezone.utc)

                # ── Proactive alerts ───────────────────────────────────────
                self._check_inactivity_alerts(now)
                self._check_drawdown_alerts(now)
                self._check_data_stale_alert(now)

                # No hourly auto-push for /status: it is available on demand via command.

                # ── Daily summary at 08:00 UTC ─────────────────────────────
                if (
                    self.notifier
                    and now.hour == 8
                    and now.minute == 0
                    and (
                        self._last_daily_summary is None
                        or (now - self._last_daily_summary).total_seconds() > 3600
                    )
                ):
                    self.notifier.notify_daily_summary(self.get_status())
                    self._last_daily_summary = now

                time.sleep(self.update_interval)

            except KeyboardInterrupt:
                logger.info("Demo stopped by user.")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                if self.notifier:
                    self.notifier.notify_error("DemoRunner main loop", str(e))
                time.sleep(10)
