# monitoring/telegram_notifier.py
import logging
import os
import threading
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import time

load_dotenv()

logger = logging.getLogger(__name__)

TZ = ZoneInfo("Europe/Madrid")


class TelegramNotifier:
    """
    Sends notifications and listens for incoming commands.
    Supported commands:
      /status — current status of all bots
      /trades — last executed trades
    """

    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not self.token or not self.chat_id:
            raise ValueError("TELEGRAM_TOKEN and TELEGRAM_CHAT_ID must be in .env")

        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self._last_update_id = 0
        self._get_status_fn = None  # injected by DemoRunner
        self._get_trades_fn = None  # injected by DemoRunner
        self._listener_thread = None

    def set_status_fn(self, fn) -> None:
        """Injects the function that returns the status of bots."""
        self._get_status_fn = fn

    def set_trades_fn(self, fn) -> None:
        """Injects the function that returns the latest trades."""
        self._get_trades_fn = fn

    def send(self, message: str) -> bool:
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False

    def notify_trade(
        self,
        bot_id: str,
        action: str,
        price: float,
        portfolio_value: float,
        reason: str,
    ) -> None:
        emoji = "🟢" if action == "buy" else "🔴"
        action_text = "BUY" if action == "buy" else "SELL"
        now = datetime.now(TZ).strftime("%H:%M")
        self.send(
            f"{emoji} <b>{bot_id}</b> — {action_text} [{now}]\n"
            f"💰 Preu: <b>{price:,.2f} USDT</b>\n"
            f"📊 Portfolio: <b>{portfolio_value:,.2f} USDT</b>\n"
            f"💬 {reason}"
        )

    def notify_status(self, status: dict) -> None:
        lines = [f"📈 <b>Bot Status</b> [{datetime.now(TZ).strftime('%H:%M')}]\n"]
        for bot_id, s in status.items():
            pnl = s["portfolio_value"] - 10_000
            pnl_pct = pnl / 10_000 * 100
            emoji = "🟢" if pnl >= 0 else "🔴"
            in_position = "📦 In Position" if s["btc_balance"] > 0 else "💵 In USDT"
            lines.append(
                f"{emoji} <b>{bot_id}</b>\n"
                f"   Portfolio: <b>{s['portfolio_value']:,.2f} USDT</b> ({pnl_pct:+.2f}%)\n"
                f"   {in_position} | BTC: {s['btc_balance']:.6f}\n"
            )
        self.send("\n".join(lines))

    def notify_daily_summary(self, status: dict) -> None:
        now = datetime.now(TZ).strftime("%Y-%m-%d")
        lines = [f"🌅 <b>Daily Summary — {now}</b>\n"]
        for bot_id, s in status.items():
            pnl = s["portfolio_value"] - 10_000
            pnl_pct = pnl / 10_000 * 100
            emoji = "🟢" if pnl >= 0 else "🔴"
            lines.append(
                f"{emoji} <b>{bot_id}</b>: "
                f"{s['portfolio_value']:,.2f} USDT ({pnl_pct:+.2f}%)"
            )
        self.send("\n".join(lines))

    def notify_drawdown_alert(self, bot_id: str, drawdown_pct: float, portfolio_value: float) -> None:
        self.send(
            f"⚠️ <b>ALERTA DRAWDOWN</b>\n"
            f"Bot: <b>{bot_id}</b>\n"
            f"Drawdown: <b>{drawdown_pct:.2f}%</b>\n"
            f"Portfolio actual: <b>{portfolio_value:,.2f} USDT</b>"
        )

    def notify_startup(self, bot_ids: list[str]) -> None:
        bots = "\n".join([f"  • {b}" for b in bot_ids])
        self.send(
            f"🚀 <b>Demo Runner Started</b>\n"
            f"Active bots:\n{bots}"
        )

    def _handle_command(self, text: str) -> None:
        """Processes an incoming command."""
        text = text.strip().lower()

        if text == "/status":
            if self._get_status_fn:
                self.notify_status(self._get_status_fn())
            else:
                self.send("⚠️ Status not available.")

        elif text == "/trades":
            if self._get_trades_fn:
                trades = self._get_trades_fn()
                if not trades:
                    self.send("No trades executed yet.")
                else:
                    lines = ["📋 <b>Latest Trades</b>\n"]
                    for t in trades[-10:]:  # last 10
                        ts = t["timestamp"].astimezone(TZ).strftime("%m-%d %H:%M")
                        emoji = "🟢" if t["action"] == "buy" else "🔴"
                        lines.append(
                            f"{emoji} [{ts}] <b>{t['bot_id']}</b> "
                            f"{t['action'].upper()} @ {t['price']:,.0f}"
                        )
                    self.send("\n".join(lines))
            else:
                self.send("⚠️ Trades not available.")

        elif text == "/help":
            self.send(
                "📖 <b>Available Commands</b>\n"
                "/status — current status of bots\n"
                "/trades — last 10 trades\n"
                "/help — this help message"
            )

        else:
            self.send(f"Unknown command: {text}\nUse /help to see available commands.")

    def _poll_updates(self) -> None:
        """Listens for Telegram updates in a separate thread with automatic retry."""
        logger.info("Telegram listener started")
        while True:
            try:
                response = requests.get(
                    f"{self.base_url}/getUpdates",
                    params={
                        "offset": self._last_update_id + 1,
                        "timeout": 30,
                    },
                    timeout=35,
                )
                data = response.json()
                for update in data.get("result", []):
                    self._last_update_id = update["update_id"]
                    message = update.get("message", {})
                    text = message.get("text", "")
                    if text.startswith("/"):
                        logger.info(f"Command received: {text}")
                        self._handle_command(text)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                # Silent reconnection — normal with long polling
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error in Telegram listener: {e}")
                time.sleep(5)

    def start_listener(self) -> None:
        """Starts the listener in a background thread."""
        self._listener_thread = threading.Thread(
            target=self._poll_updates,
            daemon=True,
        )
        self._listener_thread.start()