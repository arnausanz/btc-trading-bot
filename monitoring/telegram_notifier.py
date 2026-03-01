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
    Envia notificacions i escolta comandos entrants.
    Comandos suportats:
      /status — estat actual de tots els bots
      /trades — últims trades executats
    """

    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not self.token or not self.chat_id:
            raise ValueError("TELEGRAM_TOKEN i TELEGRAM_CHAT_ID han d'estar al .env")

        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self._last_update_id = 0
        self._get_status_fn = None  # injectat pel DemoRunner
        self._get_trades_fn = None  # injectat pel DemoRunner
        self._listener_thread = None

    def set_status_fn(self, fn) -> None:
        """Injecta la funció que retorna l'estat dels bots."""
        self._get_status_fn = fn

    def set_trades_fn(self, fn) -> None:
        """Injecta la funció que retorna els últims trades."""
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
            logger.error(f"Error enviant missatge Telegram: {e}")
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
        action_text = "COMPRA" if action == "buy" else "VENDA"
        now = datetime.now(TZ).strftime("%H:%M")
        self.send(
            f"{emoji} <b>{bot_id}</b> — {action_text} [{now}]\n"
            f"💰 Preu: <b>{price:,.2f} USDT</b>\n"
            f"📊 Portfolio: <b>{portfolio_value:,.2f} USDT</b>\n"
            f"💬 {reason}"
        )

    def notify_status(self, status: dict) -> None:
        lines = [f"📈 <b>Estat dels bots</b> [{datetime.now(TZ).strftime('%H:%M')}]\n"]
        for bot_id, s in status.items():
            pnl = s["portfolio_value"] - 10_000
            pnl_pct = pnl / 10_000 * 100
            emoji = "🟢" if pnl >= 0 else "🔴"
            in_position = "📦 En posició" if s["btc_balance"] > 0 else "💵 En USDT"
            lines.append(
                f"{emoji} <b>{bot_id}</b>\n"
                f"   Portfolio: <b>{s['portfolio_value']:,.2f} USDT</b> ({pnl_pct:+.2f}%)\n"
                f"   {in_position} | BTC: {s['btc_balance']:.6f}\n"
            )
        self.send("\n".join(lines))

    def notify_daily_summary(self, status: dict) -> None:
        now = datetime.now(TZ).strftime("%Y-%m-%d")
        lines = [f"🌅 <b>Resum diari — {now}</b>\n"]
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
            f"🚀 <b>Demo Runner iniciat</b>\n"
            f"Bots actius:\n{bots}"
        )

    def _handle_command(self, text: str) -> None:
        """Processa un comando entrant."""
        text = text.strip().lower()

        if text == "/status":
            if self._get_status_fn:
                self.notify_status(self._get_status_fn())
            else:
                self.send("⚠️ Status no disponible.")

        elif text == "/trades":
            if self._get_trades_fn:
                trades = self._get_trades_fn()
                if not trades:
                    self.send("Cap trade executat encara.")
                else:
                    lines = ["📋 <b>Últims trades</b>\n"]
                    for t in trades[-10:]:  # últims 10
                        ts = t["timestamp"].astimezone(TZ).strftime("%m-%d %H:%M")
                        emoji = "🟢" if t["action"] == "buy" else "🔴"
                        lines.append(
                            f"{emoji} [{ts}] <b>{t['bot_id']}</b> "
                            f"{t['action'].upper()} @ {t['price']:,.0f}"
                        )
                    self.send("\n".join(lines))
            else:
                self.send("⚠️ Trades no disponibles.")

        elif text == "/help":
            self.send(
                "📖 <b>Comandos disponibles</b>\n"
                "/status — estat actual dels bots\n"
                "/trades — últims 10 trades\n"
                "/help — aquesta ajuda"
            )

        else:
            self.send(f"Comando desconegut: {text}\nUsa /help per veure els comandos.")

    def _poll_updates(self) -> None:
        """Escolta updates de Telegram en un thread separat amb retry automàtic."""
        logger.info("Telegram listener iniciat")
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
                        logger.info(f"Comando rebut: {text}")
                        self._handle_command(text)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                # Reconexió silenciosa — és normal amb long polling
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error al listener Telegram: {e}")
                time.sleep(5)

    def start_listener(self) -> None:
        """Arrenca el listener en un thread background."""
        self._listener_thread = threading.Thread(
            target=self._poll_updates,
            daemon=True,
        )
        self._listener_thread.start()