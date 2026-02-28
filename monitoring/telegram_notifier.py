# core/monitoring/telegram_notifier.py
import logging
import os
from dotenv import load_dotenv
import requests

load_dotenv()

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Envia notificacions al bot de Telegram.
    Les credencials venen del .env — mai hardcoded.
    """

    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not self.token or not self.chat_id:
            raise ValueError(
                "TELEGRAM_TOKEN i TELEGRAM_CHAT_ID han d'estar al .env"
            )

        self.base_url = f"https://api.telegram.org/bot{self.token}"

    def send(self, message: str) -> bool:
        """Envia un missatge. Retorna True si ha anat bé."""
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
        message = (
            f"🤖 <b>{bot_id}</b>\n"
            f"{'🟢 COMPRA' if action == 'buy' else '🔴 VENDA'}\n"
            f"💰 Preu: <b>{price:,.2f} USDT</b>\n"
            f"📊 Portfolio: <b>{portfolio_value:,.2f} USDT</b>\n"
            f"💬 {reason}"
        )
        self.send(message)

    def notify_status(self, status: dict) -> None:
        lines = ["📈 <b>Estat dels bots</b>\n"]
        for bot_id, s in status.items():
            pnl = s["portfolio_value"] - 10_000
            pnl_pct = pnl / 10_000 * 100
            emoji = "🟢" if pnl >= 0 else "🔴"
            lines.append(
                f"{emoji} <b>{bot_id}</b>\n"
                f"   Portfolio: {s['portfolio_value']:,.2f} USDT "
                f"({pnl_pct:+.2f}%)\n"
                f"   BTC: {s['btc_balance']:.6f}\n"
            )
        self.send("\n".join(lines))

    def notify_startup(self, bot_ids: list[str]) -> None:
        bots = "\n".join([f"  • {b}" for b in bot_ids])
        self.send(
            f"🚀 <b>Demo Runner iniciat</b>\n"
            f"Bots actius:\n{bots}"
        )