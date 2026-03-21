# monitoring/telegram_notifier.py
"""
Telegram Bot Monitor for the BTC Demo Trading System.

Commands
--------
/status    — compact table: all bots, return %, alpha vs B&H, semaphore
/portfolio — inline-keyboard tabs with full per-bot detail
/compare   — matplotlib chart (return evolution day-0) + sorted text summary
/trades    — last N trades (all bots combined), with confidence & P&L
/health    — data-source freshness + bot status + system components
/help      — command list

Proactive alerts (no command needed)
-------------------------------------
• Bot inactivity  > inactivity_hours threshold
• OHLCV data     > data_stale_minutes threshold
• Max drawdown   < drawdown_alert_pct threshold
• Unhandled errors in main runner

Architecture notes
------------------
Uses raw requests + long-polling (NOT python-telegram-bot async SDK) so it
runs cleanly in the DemoRunner's background daemon thread without any
event-loop conflicts.  Inline-keyboard callbacks are handled in the same
polling loop by inspecting ``callback_query`` updates.
"""

import io
import logging
import math
import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)

TZ = ZoneInfo("Europe/Madrid")
INITIAL_CAPITAL = 10_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers  (pure functions, no I/O)
# ─────────────────────────────────────────────────────────────────────────────

def _bot_return_pct(portfolio_value: float) -> float:
    return (portfolio_value / INITIAL_CAPITAL - 1.0) * 100.0


def _bh_return_pct(history: list[dict]) -> float:
    """Buy-&-Hold return from bot-start BTC price to latest BTC price."""
    if not history or len(history) < 2:
        return 0.0
    start = history[0]["price"]
    end   = history[-1]["price"]
    return (end / start - 1.0) * 100.0 if start > 0 else 0.0


def _max_drawdown_pct(history: list[dict]) -> float:
    """Maximum peak-to-trough drawdown from portfolio history."""
    if not history:
        return 0.0
    peak   = INITIAL_CAPITAL
    max_dd = 0.0
    for row in history:
        pv   = row["portfolio_value"]
        peak = max(peak, pv)
        dd   = (pv - peak) / peak * 100.0
        if dd < max_dd:
            max_dd = dd
    return max_dd


def _sharpe_ratio(history: list[dict]) -> float | None:
    """Annualised Sharpe ratio (risk-free = 0) from daily portfolio values."""
    if len(history) < 10:
        return None
    # Sample the last value of each calendar day
    daily: dict[str, float] = {}
    for row in history:
        ts = row["timestamp"]
        key = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
        daily[key] = row["portfolio_value"]
    values = list(daily.values())
    if len(values) < 3:
        return None
    returns = [(values[i] - values[i - 1]) / values[i - 1] for i in range(1, len(values))]
    mean_r  = sum(returns) / len(returns)
    var_r   = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std_r   = var_r ** 0.5
    if std_r == 0:
        return None
    return (mean_r / std_r) * math.sqrt(365)


def _win_rate(trades: list[dict]) -> float | None:
    """
    Win-rate percentage from BUY/SELL round-trips.
    A SELL is a win if portfolio_value grew from the preceding BUY.
    """
    buys  = [t for t in trades if t["action"] == "buy"]
    sells = [t for t in trades if t["action"] == "sell"]
    if not sells:
        return None
    wins = 0
    for sell in sells:
        preceding = [b for b in buys if b["timestamp"] < sell["timestamp"]]
        if preceding and sell["portfolio_value"] > preceding[-1]["portfolio_value"]:
            wins += 1
    return wins / len(sells) * 100.0


def _trade_pnl(trade: dict, bot_trades: list[dict]) -> float | None:
    """P&L for a SELL = portfolio_value_sell - portfolio_value_preceding_buy."""
    if trade["action"] == "buy":
        return None
    buys = [t for t in bot_trades if t["action"] == "buy" and t["timestamp"] < trade["timestamp"]]
    if not buys:
        return None
    return trade["portfolio_value"] - buys[-1]["portfolio_value"]


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class TelegramNotifier:
    """
    Telegram monitor bot.  All blocking I/O runs in a daemon thread.

    Injection points (set before calling start_listener):
        set_status_fn(fn)          fn() -> dict[bot_id, status_dict]
        set_trades_fn(fn)          fn() -> list[trade_dict]  (all bots combined)
        set_all_histories_fn(fn)   fn() -> dict[bot_id, list[history_row]]
        set_health_fn(fn)          fn() -> health_dict
    """

    def __init__(
        self,
        inactivity_hours: float = 4.0,
        drawdown_alert_pct: float = -10.0,
        data_stale_minutes: float = 90.0,
    ):
        self.token   = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not self.token or not self.chat_id:
            raise ValueError("TELEGRAM_TOKEN and TELEGRAM_CHAT_ID must be set in .env")

        self.inactivity_hours   = inactivity_hours
        self.drawdown_alert_pct = drawdown_alert_pct
        self.data_stale_minutes = data_stale_minutes
        self.base_url           = f"https://api.telegram.org/bot{self.token}"
        self._last_update_id    = 0
        self._listener_thread: threading.Thread | None = None

        # Injected data functions
        self._get_status_fn         = None   # () -> dict
        self._get_trades_fn         = None   # () -> list[dict]
        self._get_all_histories_fn  = None   # () -> dict[bot_id, list]
        self._get_health_fn         = None   # () -> dict

    # ── injection ──────────────────────────────────────────────────────────

    def set_status_fn(self, fn):          self._get_status_fn = fn
    def set_trades_fn(self, fn):          self._get_trades_fn = fn
    def set_all_histories_fn(self, fn):   self._get_all_histories_fn = fn
    def set_health_fn(self, fn):          self._get_health_fn = fn

    # ── low-level Telegram API ─────────────────────────────────────────────

    def _post(self, method: str, **kwargs) -> dict | None:
        try:
            r = requests.post(f"{self.base_url}/{method}", timeout=15, **kwargs)
            data = r.json()
            if not data.get("ok"):
                logger.warning(f"Telegram {method} not ok: {data.get('description')}")
            return data
        except Exception as exc:
            logger.error(f"Telegram API error ({method}): {exc}")
            return None

    def send(self, text: str, chat_id: str | None = None) -> dict | None:
        """Send an HTML-formatted message."""
        return self._post("sendMessage", json={
            "chat_id":    chat_id or self.chat_id,
            "text":       text,
            "parse_mode": "HTML",
        })

    def _send_with_keyboard(
        self,
        text: str,
        buttons: list[list[dict]],
        chat_id: str | None = None,
    ) -> dict | None:
        return self._post("sendMessage", json={
            "chat_id":      chat_id or self.chat_id,
            "text":         text,
            "parse_mode":   "HTML",
            "reply_markup": {"inline_keyboard": buttons},
        })

    def _edit_message(
        self,
        message_id: int,
        text: str,
        buttons: list[list[dict]],
        chat_id: str | None = None,
    ) -> None:
        self._post("editMessageText", json={
            "chat_id":      chat_id or self.chat_id,
            "message_id":   message_id,
            "text":         text,
            "parse_mode":   "HTML",
            "reply_markup": {"inline_keyboard": buttons},
        })

    def _answer_callback(self, callback_id: str) -> None:
        self._post("answerCallbackQuery", json={"callback_query_id": callback_id})

    def _send_photo(self, image_bytes: bytes, caption: str = "") -> None:
        try:
            requests.post(
                f"{self.base_url}/sendPhoto",
                data={"chat_id": self.chat_id, "caption": caption, "parse_mode": "HTML"},
                files={"photo": ("compare.png", image_bytes, "image/png")},
                timeout=30,
            )
        except Exception as exc:
            logger.error(f"Error sending photo: {exc}")

    # ── /status ────────────────────────────────────────────────────────────

    def _handle_status(self) -> None:
        status    = self._get_status_fn() if self._get_status_fn else {}
        histories = self._get_all_histories_fn() if self._get_all_histories_fn else {}

        if not status:
            self.send("⚠️ No hi ha bots actius.")
            return

        now_str = datetime.now(TZ).strftime("%d/%m %H:%M")
        header  = f"{'Bot':<20}{'Retorn':>9}{'Alpha':>9}"
        sep     = "─" * 38
        rows    = []
        alerts  = []

        for bot_id, s in status.items():
            pv      = s["portfolio_value"]
            history = histories.get(bot_id, [])
            bot_ret = _bot_return_pct(pv)
            bh_ret  = _bh_return_pct(history)
            alpha   = bot_ret - bh_ret

            if alpha > 0:
                emoji = "🟢"
            elif alpha >= -1.0:
                emoji = "🟡"
            else:
                emoji = "🔴"

            short = (bot_id[:17] + "…") if len(bot_id) > 18 else bot_id
            rows.append(f"{emoji} {short:<18}{bot_ret:>+8.1f}%{alpha:>+8.1f}%")

            # Collect inactivity alerts
            last_upd = s.get("last_update")
            if last_upd:
                age_h = (datetime.now(timezone.utc) - last_upd).total_seconds() / 3600
                if age_h > self.inactivity_hours:
                    alerts.append(f"⚠️ {bot_id}: sense activitat fa {age_h:.1f}h")

        body = "\n".join([header, sep] + rows)
        self.send(f"<pre>📊 BOTS DEMO  [{now_str}]\n\n{body}</pre>")

        if alerts:
            self.send("⚠️ <b>Alertes actives</b>\n" + "\n".join(alerts))

    # ── /portfolio ─────────────────────────────────────────────────────────

    def _portfolio_detail_text(
        self,
        bot_id: str,
        status: dict,
        histories: dict,
        all_trades: list[dict],
    ) -> str:
        s          = status.get(bot_id, {})
        history    = histories.get(bot_id, [])
        bot_trades = [t for t in all_trades if t.get("bot_id") == bot_id]

        pv       = s.get("portfolio_value", INITIAL_CAPITAL)
        btc      = s.get("btc_balance", 0.0)
        usdt     = s.get("usdt_balance", pv)
        bot_ret  = _bot_return_pct(pv)
        bh_ret   = _bh_return_pct(history)
        alpha    = bot_ret - bh_ret
        max_dd   = _max_drawdown_pct(history)
        sharpe   = _sharpe_ratio(history)
        win_pct  = _win_rate(bot_trades)

        start_dt  = history[0]["timestamp"] if history else None
        start_str = start_dt.astimezone(TZ).strftime("%d/%m/%Y") if start_dt else "—"

        cur_price = history[-1]["price"] if history else 1.0
        btc_usd   = btc * cur_price
        btc_pct   = btc_usd / pv * 100 if pv > 0 else 0.0
        usdt_pct  = 100.0 - btc_pct

        sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "—"
        win_str    = f"{win_pct:.0f}%" if win_pct is not None else "—"

        lines = [
            "─" * 32,
            f"🤖 {bot_id}",
            f"   inici: {start_str}",
            "─" * 32,
            f"{'💰 Capital':<16}{pv:>12,.2f} $",
            f"{'📈 Retorn':<16}{bot_ret:>+11.1f} %",
            f"{'🔄 Alpha B&H':<16}{alpha:>+11.1f} %",
            f"{'📉 Max DD':<16}{max_dd:>+11.1f} %",
            f"{'⚡ Sharpe':<16}{sharpe_str:>12}",
            f"{'🎯 Win Rate':<16}{win_str:>12}",
            "─" * 32,
            "📦 Posició actual:",
            f"   BTC  {btc:>12.6f}  ({btc_pct:.1f}%)",
            f"   USDT {usdt:>12,.2f}  ({usdt_pct:.1f}%)",
        ]

        # Inactivity warning
        last_upd = s.get("last_update")
        if last_upd:
            age_h = (datetime.now(timezone.utc) - last_upd).total_seconds() / 3600
            if age_h > self.inactivity_hours:
                lines.append(f"\n⚠️ Sense activitat fa {age_h:.1f}h")

        return "<pre>" + "\n".join(lines) + "</pre>"

    @staticmethod
    def _portfolio_buttons(bot_ids: list[str], selected: str) -> list[list[dict]]:
        """Build inline keyboard: one button per bot, selected one bracketed."""
        all_buttons = []
        for bid in bot_ids:
            label = f"[ {bid} ]" if bid == selected else bid
            all_buttons.append({"text": label, "callback_data": f"portfolio:{bid}"})
        # Up to 3 per row
        return [all_buttons[i:i + 3] for i in range(0, len(all_buttons), 3)]

    def _handle_portfolio(self, chat_id: str) -> None:
        status    = self._get_status_fn() if self._get_status_fn else {}
        histories = self._get_all_histories_fn() if self._get_all_histories_fn else {}
        trades    = self._get_trades_fn() if self._get_trades_fn else []

        if not status:
            self.send("⚠️ No hi ha bots actius.", chat_id=chat_id)
            return

        bot_ids = list(status.keys())
        first   = bot_ids[0]
        text    = self._portfolio_detail_text(first, status, histories, trades)
        buttons = self._portfolio_buttons(bot_ids, selected=first)
        self._send_with_keyboard(text, buttons, chat_id=chat_id)

    def _handle_portfolio_callback(
        self, chat_id: str, message_id: int, bot_id: str
    ) -> None:
        status    = self._get_status_fn() if self._get_status_fn else {}
        histories = self._get_all_histories_fn() if self._get_all_histories_fn else {}
        trades    = self._get_trades_fn() if self._get_trades_fn else []

        if bot_id not in status:
            return  # Stale callback — bot no longer active

        bot_ids = list(status.keys())
        text    = self._portfolio_detail_text(bot_id, status, histories, trades)
        buttons = self._portfolio_buttons(bot_ids, selected=bot_id)
        self._edit_message(message_id, text, buttons, chat_id=chat_id)

    # ── /compare ───────────────────────────────────────────────────────────

    def _generate_compare_chart(self, histories: dict[str, list[dict]]) -> bytes | None:
        """Generate a dark-theme line chart of normalised returns. Returns PNG bytes."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import matplotlib.ticker as mticker
        except ImportError:
            logger.warning("matplotlib not installed — /compare chart unavailable")
            return None

        try:
            PALETTE = [
                "#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77",
                "#845ef7", "#ff9f43", "#0abde3", "#ee5a24",
                "#10ac84", "#c44569",
            ]
            fig, ax = plt.subplots(figsize=(11, 5))
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#161b22")

            bh_plotted = False
            for idx, (bot_id, history) in enumerate(histories.items()):
                if not history:
                    continue
                ts      = [row["timestamp"] for row in history]
                returns = [_bot_return_pct(row["portfolio_value"]) for row in history]
                color   = PALETTE[idx % len(PALETTE)]
                ax.plot(ts, returns, label=bot_id, color=color, linewidth=1.6, alpha=0.9)

                # B&H reference line — plotted once using the oldest bot's start
                if not bh_plotted:
                    start_p  = history[0]["price"]
                    bh_rets  = [(row["price"] / start_p - 1.0) * 100 for row in history]
                    ax.plot(ts, bh_rets, label="BTC B&H", color="#8b949e",
                            linewidth=1.4, linestyle="--", alpha=0.75)
                    bh_plotted = True

            ax.axhline(y=0, color="#30363d", linewidth=0.8)
            ax.set_xlabel("Data", color="#c9d1d9", fontsize=9, labelpad=4)
            ax.set_ylabel("Retorn (%)", color="#c9d1d9", fontsize=9, labelpad=4)
            ax.set_title(
                "Evolució del retorn acumulat  (day-0 = 0%)",
                color="#f0f6fc", fontsize=11, pad=10,
            )
            ax.tick_params(colors="#8b949e", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
            ax.grid(True, color="#21262d", linewidth=0.5, linestyle=":")

            legend = ax.legend(
                loc="upper left", fontsize=8, ncol=3,
                framealpha=0.5, facecolor="#161b22", edgecolor="#30363d",
            )
            for txt in legend.get_texts():
                txt.set_color("#c9d1d9")

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=130, facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            return buf.read()
        except Exception as exc:
            logger.error(f"Chart generation error: {exc}")
            return None

    def _handle_compare(self) -> None:
        status    = self._get_status_fn() if self._get_status_fn else {}
        histories = self._get_all_histories_fn() if self._get_all_histories_fn else {}

        if not status:
            self.send("⚠️ No hi ha dades per comparar.")
            return

        # Build sorted ranking text
        rows: list[tuple[str, float, float]] = []
        for bot_id, s in status.items():
            history = histories.get(bot_id, [])
            bot_ret = _bot_return_pct(s["portfolio_value"])
            bh_ret  = _bh_return_pct(history)
            rows.append((bot_id, bot_ret, bh_ret))
        rows.sort(key=lambda x: x[1], reverse=True)

        medals = ["🥇", "🥈", "🥉"]
        header = f"{'Bot':<20}{'Retorn':>9}{'B&H ref':>9}"
        sep    = "─" * 38
        lines  = [header, sep]
        for i, (bot_id, bot_ret, bh_ret) in enumerate(rows):
            prefix = medals[i] if i < 3 else f" {i + 1}."
            short  = (bot_id[:17] + "…") if len(bot_id) > 18 else bot_id
            lines.append(f"{prefix} {short:<17}{bot_ret:>+8.1f}%{bh_ret:>+8.1f}%")

        caption = "<pre>📊 RÀNQUING DE RETORNS\n\n" + "\n".join(lines) + "</pre>"

        chart_bytes = self._generate_compare_chart(histories)
        if chart_bytes:
            self._send_photo(chart_bytes, caption=caption)
        else:
            self.send(caption)
            self.send(
                "ℹ️ Per veure el gràfic instal·la matplotlib:\n"
                "<code>poetry add matplotlib</code>"
            )

    # ── /trades ────────────────────────────────────────────────────────────

    def _handle_trades(self, n: int = 15) -> None:
        all_trades = self._get_trades_fn() if self._get_trades_fn else []

        if not all_trades:
            self.send("📋 Encara no hi ha trades executats.")
            return

        # Sort descending, take last N
        sorted_desc = sorted(all_trades, key=lambda t: t["timestamp"], reverse=True)
        recent = sorted_desc[:n]

        # Group by bot for P&L calculation
        trades_by_bot: dict[str, list[dict]] = defaultdict(list)
        for t in all_trades:
            trades_by_bot[t["bot_id"]].append(t)

        # Day summary
        today       = datetime.now(TZ).date()
        today_trades = [
            t for t in all_trades
            if t["timestamp"].astimezone(TZ).date() == today
        ]
        day_pnl = 0.0
        for t in today_trades:
            p = _trade_pnl(t, trades_by_bot[t["bot_id"]])
            if p is not None:
                day_pnl += p

        now_str = datetime.now(TZ).strftime("%d/%m %H:%M")

        # Header
        col_h = f"{'Hora':<12}{'Bot':<16}{'T':>2}{'BTC':>9}{'Preu':>9}{'P&L':>9}{'Conf':>6}"
        sep   = "─" * 63
        rows  = []
        for t in recent:
            ts_str   = t["timestamp"].astimezone(TZ).strftime("%m-%d %H:%M")
            emoji    = "🟢" if t["action"] == "buy" else "🔴"
            short    = (t["bot_id"][:14] + "…") if len(t["bot_id"]) > 15 else t["bot_id"]
            size_btc = t.get("size_btc", 0.0)
            price    = t.get("price", 0.0)
            conf     = t.get("confidence", 0.0)
            pnl      = _trade_pnl(t, trades_by_bot[t["bot_id"]])

            pnl_str  = f"{pnl:>+8.0f}$" if pnl is not None else "       —"
            conf_str = f"{conf * 100:.0f}%" if conf and conf > 0 else "  —"
            rows.append(
                f"{ts_str} {emoji}{short:<14}{size_btc:>9.4f}{price:>9,.0f}{pnl_str}{conf_str:>6}"
            )

        # Day summary line
        pnl_sign = "+" if day_pnl >= 0 else ""
        summary  = f"Avui: {len(today_trades)} trades  |  P&L net: {pnl_sign}{day_pnl:,.0f} $"

        body = "\n".join([col_h, sep] + rows + [sep, summary])
        self.send(f"<pre>📋 ÚLTIMS TRADES  [{now_str}]\n\n{body}</pre>")

    # ── /health ────────────────────────────────────────────────────────────

    def _handle_health(self) -> None:
        health    = self._get_health_fn() if self._get_health_fn else {}
        status    = self._get_status_fn() if self._get_status_fn else {}
        now       = datetime.now(timezone.utc)
        now_str   = datetime.now(TZ).strftime("%d/%m %H:%M")

        lines: list[str] = []

        # ── Data sources ──────────────────────────────────────────────────
        lines.append("── Fonts de dades ──────────────────────")

        last_candle = health.get("last_candle_ts")
        if last_candle:
            age_min = (now - last_candle).total_seconds() / 60
            ok      = age_min < self.data_stale_minutes
            lines.append(
                f"{'✅' if ok else '⚠️'} OHLCV         fa {age_min:.0f} min"
            )
        else:
            lines.append("⚠️ OHLCV         sense dades")

        fg = health.get("fear_greed")
        if fg:
            age_h = (now - fg["timestamp"]).total_seconds() / 3600
            ok    = age_h < 26
            lines.append(
                f"{'✅' if ok else '⚠️'} Fear&Greed    {fg['value']} "
                f"({fg['classification']})  fa {age_h:.0f}h"
            )
        else:
            lines.append("⚠️ Fear&Greed    sense dades")

        db_ok = health.get("db_ok", False)
        lines.append(
            f"{'✅' if db_ok else '⚠️'} TimescaleDB   "
            f"{'connectat' if db_ok else 'ERROR'}"
        )

        binance_ms = health.get("binance_latency_ms")
        if binance_ms is not None:
            ok = binance_ms < 500
            lines.append(f"{'✅' if ok else '⚠️'} Binance API   {binance_ms:.0f} ms")
        else:
            lines.append("⚠️ Binance API   no mesurat")

        # ── Bot states ────────────────────────────────────────────────────
        lines.append("")
        lines.append("── Bots ─────────────────────────────────")

        n_ok = n_warn = 0
        for bot_id, s in status.items():
            last_upd = s.get("last_update")
            if last_upd:
                age_h = (now - last_upd).total_seconds() / 3600
                ok    = age_h < self.inactivity_hours
            else:
                age_h = float("inf")
                ok    = False

            short = (bot_id[:22] + "…") if len(bot_id) > 23 else bot_id
            if ok:
                n_ok += 1
                lines.append(f"🟢 {short:<24} nominal")
            else:
                n_warn += 1
                lines.append(f"⚠️ {short:<24} inactiu {age_h:.1f}h")

        # ── Summary ───────────────────────────────────────────────────────
        lines.append("")
        lines.append("─" * 40)
        next_check = health.get("next_check_minutes", 60)
        lines.append(f"Resum: {n_ok} OK  |  {n_warn} avis{'os' if n_warn != 1 else ''}")
        lines.append(f"Proper check automàtic en {next_check} min")

        self.send(
            f"<pre>🏥 HEALTH  [{now_str}]\n\n" + "\n".join(lines) + "</pre>"
        )

    # ── proactive notifications ────────────────────────────────────────────

    def notify_trade(
        self,
        bot_id: str,
        action: str,
        price: float,
        portfolio_value: float,
        reason: str,
        confidence: float = 0.0,
        size_btc: float = 0.0,
    ) -> None:
        emoji      = "🟢" if action == "buy" else "🔴"
        action_txt = "BUY" if action == "buy" else "SELL"
        bot_ret    = _bot_return_pct(portfolio_value)
        now_str    = datetime.now(TZ).strftime("%H:%M")
        conf_str   = f"{confidence * 100:.0f}%" if confidence > 0 else "—"
        self.send(
            f"{emoji} <b>{bot_id}</b> — {action_txt} [{now_str}]\n"
            f"💰 Preu: <b>{price:,.2f} USDT</b>  ·  {size_btc:.4f} BTC\n"
            f"📊 Portfolio: <b>{portfolio_value:,.2f} $</b> ({bot_ret:+.1f}%)\n"
            f"🎯 Confiança: {conf_str}\n"
            f"💬 {reason}"
        )

    def notify_inactivity_alert(self, bot_id: str, hours: float) -> None:
        self.send(
            f"⚠️ <b>INACTIVITAT</b>\n"
            f"Bot <b>{bot_id}</b> sense trades fa <b>{hours:.1f}h</b>"
        )

    def notify_drawdown_alert(
        self, bot_id: str, drawdown_pct: float, portfolio_value: float
    ) -> None:
        bot_ret = _bot_return_pct(portfolio_value)
        self.send(
            f"⚠️ <b>ALERTA DRAWDOWN</b>\n"
            f"Bot: <b>{bot_id}</b>\n"
            f"Drawdown: <b>{drawdown_pct:.1f}%</b>\n"
            f"Portfolio: <b>{portfolio_value:,.2f} $</b> ({bot_ret:+.1f}%)"
        )

    def notify_data_stale(self, source: str, age_minutes: float) -> None:
        self.send(
            f"⚠️ <b>DADES OBSOLETES</b>\n"
            f"Font: <b>{source}</b>\n"
            f"Última actualització: fa <b>{age_minutes:.0f} min</b>"
        )

    def notify_error(self, component: str, error_msg: str) -> None:
        self.send(
            f"🔴 <b>ERROR</b> a <b>{component}</b>\n"
            f"<code>{error_msg[:300]}</code>"
        )

    def notify_startup(self, bot_ids: list[str]) -> None:
        bots_str = "\n".join(f"  • {b}" for b in bot_ids)
        self.send(
            f"🚀 <b>Demo Runner iniciat</b>\n"
            f"Bots actius:\n{bots_str}\n\n"
            f"Comandes: /status  /portfolio  /compare  /trades  /health  /help"
        )

    def notify_daily_summary(self, status: dict) -> None:
        histories = self._get_all_histories_fn() if self._get_all_histories_fn else {}
        now_str   = datetime.now(TZ).strftime("%Y-%m-%d")

        rows: list[tuple] = []
        for bot_id, s in status.items():
            history = histories.get(bot_id, [])
            bot_ret = _bot_return_pct(s["portfolio_value"])
            bh_ret  = _bh_return_pct(history)
            alpha   = bot_ret - bh_ret
            rows.append((bot_id, bot_ret, alpha, s["portfolio_value"]))
        rows.sort(key=lambda x: x[1], reverse=True)

        lines = [f"🌅 <b>Resum diari — {now_str}</b>\n"]
        for bot_id, bot_ret, alpha, pv in rows:
            emoji = "🟢" if alpha > 0 else ("🟡" if alpha >= -1 else "🔴")
            lines.append(
                f"{emoji} <b>{bot_id}</b>: {pv:,.2f} $  "
                f"({bot_ret:+.1f}%  α {alpha:+.1f}%)"
            )
        self.send("\n".join(lines))

    # ── command routing ────────────────────────────────────────────────────

    def _handle_command(self, text: str, chat_id: str) -> None:
        cmd = text.strip().lower().split()[0]
        try:
            if cmd == "/status":
                self._handle_status()
            elif cmd == "/portfolio":
                self._handle_portfolio(chat_id)
            elif cmd == "/compare":
                self._handle_compare()
            elif cmd == "/trades":
                self._handle_trades()
            elif cmd == "/health":
                self._handle_health()
            elif cmd == "/help":
                self.send(
                    "📖 <b>Comandes disponibles</b>\n\n"
                    "/status      — taula resum de tots els bots\n"
                    "/portfolio   — detall interactiu per bot (botons inline)\n"
                    "/compare     — gràfic d'evolució + rànquing\n"
                    "/trades      — últims trades amb P&amp;L i confiança\n"
                    "/health      — estat de fonts de dades i components\n"
                    "/help        — aquest missatge"
                )
            else:
                self.send(
                    f"Comanda desconeguda: <code>{text}</code>\n"
                    "Escriu /help per veure les comandes disponibles."
                )
        except Exception as exc:
            logger.error(f"Error handling command {cmd}: {exc}", exc_info=True)
            self.send(f"⚠️ Error intern processant <code>{cmd}</code>.")

    # ── polling loop ───────────────────────────────────────────────────────

    def _poll_updates(self) -> None:
        logger.info("Telegram listener started")
        while True:
            try:
                r = requests.get(
                    f"{self.base_url}/getUpdates",
                    params={"offset": self._last_update_id + 1, "timeout": 30},
                    timeout=35,
                )
                data = r.json()
                for update in data.get("result", []):
                    self._last_update_id = update["update_id"]

                    # ── regular message (command) ─────────────────────────
                    msg  = update.get("message", {})
                    text = msg.get("text", "")
                    if text.startswith("/"):
                        chat_id = str(msg.get("chat", {}).get("id", self.chat_id))
                        logger.info(f"Command: {text!r} from chat {chat_id}")
                        self._handle_command(text, chat_id)

                    # ── inline keyboard callback ──────────────────────────
                    cq = update.get("callback_query")
                    if cq:
                        cb_id      = cq["id"]
                        cb_data    = cq.get("data", "")
                        cb_chat_id = str(
                            cq.get("message", {})
                              .get("chat", {})
                              .get("id", self.chat_id)
                        )
                        cb_msg_id  = cq.get("message", {}).get("message_id")

                        # Always ack the callback immediately
                        self._answer_callback(cb_id)

                        if cb_data.startswith("portfolio:") and cb_msg_id:
                            selected_bot = cb_data[len("portfolio:"):]
                            logger.info(f"Portfolio callback: {selected_bot}")
                            self._handle_portfolio_callback(
                                cb_chat_id, cb_msg_id, selected_bot
                            )

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                # Expected during long-polling; silent retry
                time.sleep(5)
            except Exception as exc:
                logger.error(f"Telegram listener error: {exc}")
                time.sleep(5)

    def start_listener(self) -> None:
        """Start the polling loop in a background daemon thread."""
        self._listener_thread = threading.Thread(
            target=self._poll_updates, daemon=True, name="telegram-listener"
        )
        self._listener_thread.start()
