# monitoring/telegram_notifier.py
"""
Telegram Bot Monitor for the BTC Demo Trading System.

Commands
--------
/status    — compact table: all bots, return %, alpha vs B&H, capital €, refresh button
/portfolio — inline-keyboard tabs with full per-bot detail + chart button + refresh
/compare   — matplotlib chart (return evolution day-0) + sorted text table (refresh)
/trades    — last N trades (all bots combined), with confidence & P&L, refresh button
/health    — data-source freshness + bot status + system components, refresh button
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

Refresh pattern
---------------
Every informational command (/status, /compare-table, /trades, /health) sends
a message with a "🔄 Actualitzar" inline button.  When pressed, the same
message is edited in-place with fresh data — no need to retype the command.
/portfolio already had tab-switching; its refresh re-fetches data for the
currently selected bot.
/compare sends two messages: a photo snapshot + a live text table.  Refresh
updates the text table in-place; a new chart is sent on the next /compare.
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
    daily: dict[str, float] = {}
    for row in history:
        ts  = row["timestamp"]
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
    """Win-rate % from BUY/SELL round-trips."""
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
        self._get_status_fn         = None
        self._get_trades_fn         = None
        self._get_all_histories_fn  = None
        self._get_health_fn         = None

        # Injected control functions
        self._pause_fn     = None
        self._resume_fn    = None
        self._close_fn     = None
        self._restart_fn   = None
        self._is_paused_fn = None
        self._reset_fn     = None

    # ── injection ──────────────────────────────────────────────────────────

    def set_status_fn(self, fn):          self._get_status_fn = fn
    def set_trades_fn(self, fn):          self._get_trades_fn = fn
    def set_all_histories_fn(self, fn):   self._get_all_histories_fn = fn
    def set_health_fn(self, fn):          self._get_health_fn = fn
    def set_pause_fn(self, fn):           self._pause_fn = fn
    def set_resume_fn(self, fn):          self._resume_fn = fn
    def set_close_fn(self, fn):           self._close_fn = fn
    def set_restart_fn(self, fn):         self._restart_fn = fn
    def set_is_paused_fn(self, fn):       self._is_paused_fn = fn
    def set_reset_fn(self, fn):           self._reset_fn = fn

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
        """Send an HTML-formatted message. Returns Telegram response dict."""
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

    def _send_photo(self, image_bytes: bytes, caption: str = "", chat_id: str | None = None) -> None:
        try:
            requests.post(
                f"{self.base_url}/sendPhoto",
                data={
                    "chat_id":    chat_id or self.chat_id,
                    "caption":    caption,
                    "parse_mode": "HTML",
                },
                files={"photo": ("chart.png", image_bytes, "image/png")},
                timeout=30,
            )
        except Exception as exc:
            logger.error(f"Error sending photo: {exc}")

    @staticmethod
    def _refresh_btn(callback: str) -> list[list[dict]]:
        """Single-button keyboard row with a refresh action."""
        return [[{"text": "🔄 Actualitzar", "callback_data": callback}]]

    # ─────────────────────────────────────────────────────────────────────
    # Content generators — return (text, buttons) for initial send & refresh
    # ─────────────────────────────────────────────────────────────────────

    def _status_content(self) -> tuple[str, list[list[dict]]]:
        """Compact table: bot, return %, alpha, capital €."""
        status    = self._get_status_fn() if self._get_status_fn else {}
        histories = self._get_all_histories_fn() if self._get_all_histories_fn else {}
        now_str   = datetime.now(TZ).strftime("%d/%m %H:%M")

        if not status:
            return "⚠️ No hi ha bots actius.", []

        # Header: Bot(19) Retorn(8) Alpha(8) €(10) = ~45 chars
        header = f"{'Bot':<19}{'Retorn':>8}{'Alpha':>8}{'€':>10}"
        sep    = "─" * 45
        rows   = []
        alerts = []

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

            short = (bot_id[:16] + "…") if len(bot_id) > 17 else bot_id
            rows.append(
                f"{emoji} {short:<17}{bot_ret:>+7.1f}%{alpha:>+7.1f}%{pv:>9,.0f}€"
            )

            last_upd = s.get("last_update")
            if last_upd:
                age_h = (datetime.now(timezone.utc) - last_upd).total_seconds() / 3600
                if age_h > self.inactivity_hours:
                    alerts.append(f"⚠️ {bot_id}: sense activitat fa {age_h:.1f}h")

        body  = "\n".join([header, sep] + rows)
        extra = ("\n\n" + "\n".join(alerts)) if alerts else ""
        text  = f"<pre>📊 BOTS DEMO  [{now_str}]\n\n{body}{extra}</pre>"
        return text, self._refresh_btn("refresh_status")

    def _compare_table_content(self) -> tuple[str, list[list[dict]]]:
        """
        Sorted ranking table: bot, return, alpha vs B&H, capital €.
        Sent separately from the chart photo to avoid Telegram's 1024-char caption limit.
        """
        status    = self._get_status_fn() if self._get_status_fn else {}
        histories = self._get_all_histories_fn() if self._get_all_histories_fn else {}
        now_str   = datetime.now(TZ).strftime("%d/%m %H:%M")

        if not status:
            return "⚠️ No hi ha dades per comparar.", []

        rows: list[tuple] = []
        for bot_id, s in status.items():
            history = histories.get(bot_id, [])
            bot_ret = _bot_return_pct(s["portfolio_value"])
            bh_ret  = _bh_return_pct(history)
            alpha   = bot_ret - bh_ret
            rows.append((bot_id, bot_ret, alpha, s["portfolio_value"]))
        rows.sort(key=lambda x: x[1], reverse=True)

        medals = ["🥇", "🥈", "🥉"]
        # Header: Bot(20) Retorn(8) Alpha(8) €(10) = ~46 chars
        header = f"{'Bot':<20}{'Retorn':>8}{'Alpha':>8}{'€':>10}"
        sep    = "─" * 46
        lines  = [header, sep]
        for i, (bot_id, bot_ret, alpha, pv) in enumerate(rows):
            prefix = medals[i] if i < 3 else f" {i + 1}."
            short  = (bot_id[:17] + "…") if len(bot_id) > 18 else bot_id
            lines.append(
                f"{prefix} {short:<17}{bot_ret:>+7.1f}%{alpha:>+7.1f}%{pv:>9,.0f}€"
            )

        text = f"<pre>📊 RÀNQUING  [{now_str}]\n\n" + "\n".join(lines) + "</pre>"
        return text, self._refresh_btn("refresh_compare")

    def _trades_content(self, n: int = 15) -> tuple[str, list[list[dict]]]:
        """Last N trades — compact format that fits mobile screens (<50 chars/row)."""
        all_trades = self._get_trades_fn() if self._get_trades_fn else []
        now_str    = datetime.now(TZ).strftime("%d/%m %H:%M")

        if not all_trades:
            return "📋 Encara no hi ha trades executats.", self._refresh_btn("refresh_trades")

        sorted_desc = sorted(all_trades, key=lambda t: t["timestamp"], reverse=True)
        recent      = sorted_desc[:n]

        trades_by_bot: dict[str, list[dict]] = defaultdict(list)
        for t in all_trades:
            trades_by_bot[t["bot_id"]].append(t)

        today        = datetime.now(TZ).date()
        today_trades = [
            t for t in all_trades
            if t["timestamp"].astimezone(TZ).date() == today
        ]
        day_pnl = 0.0
        for t in today_trades:
            p = _trade_pnl(t, trades_by_bot[t["bot_id"]])
            if p is not None:
                day_pnl += p

        # Columns: Data(11) Bot(14) Tipus(5) Preu(8) P&L(8) Conf(5) = ~51 chars
        col_h = f"{'Data':<11} {'Bot':<12} {'T':>5} {'Preu':>8} {'P&L':>7} {'Cf':>4}"
        sep   = "─" * 51
        rows  = []
        for t in recent:
            ts_str  = t["timestamp"].astimezone(TZ).strftime("%m/%d %H:%M")
            emoji   = "🟢" if t["action"] == "buy" else "🔴"
            short   = (t["bot_id"][:12] + "…") if len(t["bot_id"]) > 13 else t["bot_id"]
            action  = "BUY" if t["action"] == "buy" else "SEL"
            price   = t.get("price", 0.0)
            conf    = t.get("confidence", 0.0)
            pnl     = _trade_pnl(t, trades_by_bot[t["bot_id"]])
            pnl_str = f"{pnl:>+7,.0f}$" if pnl is not None else "      —"
            c_str   = f"{conf * 100:.0f}%" if conf and conf > 0 else "  —"
            rows.append(
                f"{ts_str} {emoji}{short:<12}{action:>5}{price:>9,.0f}{pnl_str}{c_str:>5}"
            )

        sign    = "+" if day_pnl >= 0 else ""
        summary = f"Avui: {len(today_trades)} trades  |  P&L: {sign}{day_pnl:,.0f}$"
        body    = "\n".join([col_h, sep] + rows + [sep, summary])
        text    = f"<pre>📋 ÚLTIMS TRADES  [{now_str}]\n\n{body}</pre>"
        return text, self._refresh_btn("refresh_trades")

    def _health_content(self) -> tuple[str, list[list[dict]]]:
        """Health check: data sources + bot states + system summary."""
        health  = self._get_health_fn() if self._get_health_fn else {}
        status  = self._get_status_fn() if self._get_status_fn else {}
        now     = datetime.now(timezone.utc)
        now_str = datetime.now(TZ).strftime("%d/%m %H:%M")
        lines: list[str] = []

        lines.append("── Fonts de dades ──────────────────────")
        last_candle    = health.get("last_candle_ts")
        expected_candle = health.get("expected_candle_ts")
        last_update_ts  = health.get("last_data_update_ts")
        if last_candle and expected_candle:
            # ✅ if we have the candle we're supposed to have (or newer).
            # The last_candle open-time should equal expected_candle open-time.
            # Allow 1 period of tolerance (e.g. data updated between closes).
            has_expected = last_candle >= expected_candle
            # How long ago did the fetcher run?
            fetch_age = (
                f"actualitzat fa {(now - last_update_ts).total_seconds() / 60:.0f} min"
                if last_update_ts else "fetcher no executat"
            )
            candle_str = last_candle.strftime("%H:%M UTC")
            # How long ago did the last candle CLOSE (open_time + 1 period)?
            close_ago_min = (now - last_candle).total_seconds() / 60 - 60
            close_ago_min = max(0, close_ago_min)
            lines.append(
                f"{'✅' if has_expected else '⚠️'} OHLCV         {fetch_age}\n"
                f"   └ última candle: {candle_str}"
                f"  (tancada fa {close_ago_min:.0f} min)"
            )
        else:
            lines.append("⚠️ OHLCV         sense dades")

        fg = health.get("fear_greed")
        if fg:
            age_h = (now - fg["timestamp"]).total_seconds() / 3600
            ok_fg = age_h < 26
            lines.append(
                f"{'✅' if ok_fg else '⚠️'} Fear&Greed    "
                f"{fg['value']} ({fg['classification']})  fa {age_h:.0f}h"
            )
        else:
            lines.append("⚠️ Fear&Greed    sense dades")

        db_ok = health.get("db_ok", False)
        lines.append(f"{'✅' if db_ok else '⚠️'} TimescaleDB   {'connectat' if db_ok else 'ERROR'}")

        binance_ms = health.get("binance_latency_ms")
        if binance_ms is not None:
            ok_b = binance_ms < 500
            lines.append(f"{'✅' if ok_b else '⚠️'} Binance API   {binance_ms:.0f} ms")
        else:
            lines.append("⚠️ Binance API   no mesurat")

        lines.append("")
        lines.append("── Bots ─────────────────────────────────")

        n_ok = n_warn = 0
        for bot_id, s in status.items():
            last_upd = s.get("last_update")
            if last_upd:
                age_h = (now - last_upd).total_seconds() / 3600
                ok_b  = age_h < self.inactivity_hours
            else:
                age_h = float("inf")
                ok_b  = False

            short = (bot_id[:22] + "…") if len(bot_id) > 23 else bot_id
            if ok_b:
                n_ok += 1
                lines.append(f"🟢 {short:<24} nominal")
            else:
                n_warn += 1
                lines.append(f"⚠️ {short:<24} inactiu {age_h:.1f}h")

        lines.append("")
        lines.append("─" * 40)
        next_check = health.get("next_check_minutes", 60)
        lines.append(f"Resum: {n_ok} OK  |  {n_warn} avis{'os' if n_warn != 1 else ''}")
        lines.append(f"Proper check automàtic en {next_check} min")

        text = f"<pre>🏥 HEALTH  [{now_str}]\n\n" + "\n".join(lines) + "</pre>"
        return text, self._refresh_btn("refresh_health")

    # ─────────────────────────────────────────────────────────────────────
    # Chart generators
    # ─────────────────────────────────────────────────────────────────────

    def _generate_compare_chart(self, histories: dict[str, list[dict]]) -> bytes | None:
        """
        Dark-theme line chart of normalised returns for all bots.
        Larger (14×7 in, 150 dpi) with outside legend to avoid overlapping lines.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import matplotlib.ticker as mticker
        except ImportError:
            logger.warning("matplotlib not installed — chart unavailable")
            return None

        try:
            PALETTE = [
                "#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77",
                "#845ef7", "#ff9f43", "#0abde3", "#ee5a24",
                "#10ac84", "#c44569",
            ]
            # Larger canvas + room on the right for the legend
            fig, ax = plt.subplots(figsize=(14, 7))
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#161b22")

            bh_plotted = False
            for idx, (bot_id, history) in enumerate(histories.items()):
                if not history:
                    continue
                ts      = [row["timestamp"] for row in history]
                returns = [_bot_return_pct(row["portfolio_value"]) for row in history]
                color   = PALETTE[idx % len(PALETTE)]
                ax.plot(ts, returns, label=bot_id, color=color, linewidth=2.0, alpha=0.9)

                # Annotate current value at the right end of each line
                if returns:
                    ax.annotate(
                        f"{returns[-1]:+.1f}%",
                        xy=(ts[-1], returns[-1]),
                        xytext=(6, 0),
                        textcoords="offset points",
                        color=color,
                        fontsize=8,
                        va="center",
                    )

                if not bh_plotted:
                    start_p = history[0]["price"]
                    bh_rets = [(row["price"] / start_p - 1.0) * 100 for row in history]
                    ax.plot(ts, bh_rets, label="BTC B&H", color="#8b949e",
                            linewidth=1.4, linestyle="--", alpha=0.75)
                    bh_plotted = True

            ax.axhline(y=0, color="#30363d", linewidth=0.8)
            ax.set_xlabel("Data", color="#c9d1d9", fontsize=9, labelpad=4)
            ax.set_ylabel("Retorn (%)", color="#c9d1d9", fontsize=9, labelpad=4)
            ax.set_title(
                "Evolució del retorn acumulat  (day-0 = 0%)",
                color="#f0f6fc", fontsize=12, pad=12,
            )
            ax.tick_params(colors="#8b949e", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
            ax.grid(True, color="#21262d", linewidth=0.5, linestyle=":")

            # Legend outside chart to the right — doesn't overlap lines
            legend = ax.legend(
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=9,
                framealpha=0.6,
                facecolor="#161b22",
                edgecolor="#30363d",
            )
            for txt in legend.get_texts():
                txt.set_color("#c9d1d9")

            # Extra right margin for legend
            plt.tight_layout(rect=[0, 0, 0.84, 1])
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            return buf.read()
        except Exception as exc:
            logger.error(f"Chart generation error: {exc}")
            return None

    def _generate_single_bot_chart(
        self, bot_id: str, history: list[dict]
    ) -> bytes | None:
        """
        Equity curve for a single bot with the B&H reference.
        Used by the 📈 Gràfic button in /portfolio.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import matplotlib.ticker as mticker
        except ImportError:
            return None

        if not history or len(history) < 2:
            return None

        try:
            ts      = [row["timestamp"] for row in history]
            returns = [_bot_return_pct(row["portfolio_value"]) for row in history]
            start_p = history[0]["price"]
            bh_rets = [(row["price"] / start_p - 1.0) * 100 for row in history]

            fig, ax = plt.subplots(figsize=(12, 5))
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#161b22")

            ax.plot(ts, returns, label=bot_id, color="#00d4ff", linewidth=2.2, alpha=0.95)
            ax.fill_between(
                ts, returns, 0,
                where=[r >= 0 for r in returns],
                color="#00d4ff", alpha=0.08,
            )
            ax.fill_between(
                ts, returns, 0,
                where=[r < 0 for r in returns],
                color="#ff6b6b", alpha=0.10,
            )
            ax.plot(ts, bh_rets, label="BTC B&H", color="#8b949e",
                    linewidth=1.4, linestyle="--", alpha=0.75)
            ax.axhline(y=0, color="#30363d", linewidth=0.8)

            # Mark BUY/SELL trades
            trades_by_bot = {}
            if self._get_trades_fn:
                all_trades = self._get_trades_fn()
                trades_by_bot = {
                    t["timestamp"]: t for t in all_trades if t["bot_id"] == bot_id
                }
            for ts_trade, trade in trades_by_bot.items():
                # Find closest history point
                closest_ret = None
                min_diff = float("inf")
                for row in history:
                    diff = abs((row["timestamp"] - ts_trade).total_seconds())
                    if diff < min_diff:
                        min_diff = diff
                        closest_ret = _bot_return_pct(row["portfolio_value"])
                if closest_ret is not None:
                    color  = "#6bcb77" if trade["action"] == "buy" else "#ff6b6b"
                    marker = "^" if trade["action"] == "buy" else "v"
                    ax.plot(ts_trade, closest_ret, marker=marker, color=color,
                            markersize=8, zorder=5)

            # Final return annotation
            final_ret = returns[-1]
            ax.annotate(
                f"{final_ret:+.2f}%",
                xy=(ts[-1], final_ret),
                xytext=(-50, 10 if final_ret >= 0 else -18),
                textcoords="offset points",
                color="#00d4ff",
                fontsize=10,
                fontweight="bold",
            )

            ax.set_xlabel("Data", color="#c9d1d9", fontsize=9, labelpad=4)
            ax.set_ylabel("Retorn (%)", color="#c9d1d9", fontsize=9, labelpad=4)
            ax.set_title(f"Corba d'equity — {bot_id}", color="#f0f6fc", fontsize=11, pad=10)
            ax.tick_params(colors="#8b949e", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
            ax.grid(True, color="#21262d", linewidth=0.5, linestyle=":")

            legend = ax.legend(fontsize=9, framealpha=0.5,
                               facecolor="#161b22", edgecolor="#30363d")
            for txt in legend.get_texts():
                txt.set_color("#c9d1d9")

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            return buf.read()
        except Exception as exc:
            logger.error(f"Single-bot chart error ({bot_id}): {exc}")
            return None

    # ─────────────────────────────────────────────────────────────────────
    # Command handlers
    # ─────────────────────────────────────────────────────────────────────

    def _handle_status(self, chat_id: str) -> None:
        text, buttons = self._status_content()
        self._send_with_keyboard(text, buttons, chat_id=chat_id)

    def _handle_compare(self, chat_id: str) -> None:
        """Send chart (photo) + separate text table with refresh button."""
        histories = self._get_all_histories_fn() if self._get_all_histories_fn else {}

        if not (self._get_status_fn and self._get_status_fn()):
            self.send("⚠️ No hi ha dades per comparar.", chat_id=chat_id)
            return

        # 1 — Photo (static snapshot, no caption, no keyboard)
        chart_bytes = self._generate_compare_chart(histories)
        if chart_bytes:
            self._send_photo(chart_bytes, caption="", chat_id=chat_id)
        else:
            self.send(
                "ℹ️ Gràfic no disponible. Instal·la matplotlib:\n"
                "<code>poetry add matplotlib</code>",
                chat_id=chat_id,
            )

        # 2 — Text table with refresh button (can be updated in-place)
        text, buttons = self._compare_table_content()
        self._send_with_keyboard(text, buttons, chat_id=chat_id)

    def _handle_trades(self, chat_id: str, n: int = 15) -> None:
        text, buttons = self._trades_content(n)
        self._send_with_keyboard(text, buttons, chat_id=chat_id)

    def _handle_health(self, chat_id: str) -> None:
        text, buttons = self._health_content()
        self._send_with_keyboard(text, buttons, chat_id=chat_id)

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

        # Days active
        if start_dt:
            days = (datetime.now(timezone.utc) - start_dt).total_seconds() / 86400
            days_str = f"{days:.0f} dies"
        else:
            days_str = "—"

        cur_price = history[-1]["price"] if history else 1.0
        btc_usd   = btc * cur_price
        btc_pct   = btc_usd / pv * 100 if pv > 0 else 0.0
        usdt_pct  = 100.0 - btc_pct

        sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "—"
        win_str    = f"{win_pct:.0f}%" if win_pct is not None else "—"
        n_trades   = len([t for t in bot_trades if t["action"] in ("buy", "sell")])

        lines = [
            "─" * 32,
            f"🤖 {bot_id}",
            f"   Inici: {start_str}  ({days_str})",
            "─" * 32,
            f"{'💰 Capital':<16}{pv:>12,.2f} €",
            f"{'📈 Retorn':<16}{bot_ret:>+11.2f} %",
            f"{'🔄 Alpha B&H':<16}{alpha:>+11.2f} %",
            f"{'📉 Max DD':<16}{max_dd:>+11.2f} %",
            f"{'⚡ Sharpe':<16}{sharpe_str:>12}",
            f"{'🎯 Win Rate':<16}{win_str:>12}",
            f"{'📊 Trades':<16}{n_trades:>12}",
            "─" * 32,
            "📦 Posició actual:",
            f"   BTC  {btc:>12.6f}  ({btc_pct:.1f}%)",
            f"   USDT {usdt:>12,.2f}  ({usdt_pct:.1f}%)",
        ]

        last_upd = s.get("last_update")
        if last_upd:
            age_h = (datetime.now(timezone.utc) - last_upd).total_seconds() / 3600
            if age_h > self.inactivity_hours:
                lines.append(f"\n⚠️ Sense activitat fa {age_h:.1f}h")

        return "<pre>" + "\n".join(lines) + "</pre>"

    @staticmethod
    def _portfolio_buttons(bot_ids: list[str], selected: str) -> list[list[dict]]:
        """
        Inline keyboard: one button per bot (3 per row) + bottom row with
        📈 Gràfic and 🔄 Actualitzar for the currently selected bot.
        """
        all_btns = []
        for bid in bot_ids:
            label = f"[ {bid} ]" if bid == selected else bid
            all_btns.append({"text": label, "callback_data": f"portfolio:{bid}"})
        rows = [all_btns[i:i + 3] for i in range(0, len(all_btns), 3)]
        # Action row — always refers to the selected bot
        rows.append([
            {"text": "📈 Gràfic",       "callback_data": f"portfolio_chart:{selected}"},
            {"text": "🔄 Actualitzar",  "callback_data": f"portfolio_refresh:{selected}"},
        ])
        return rows

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
        """Edit the portfolio message in-place when a bot tab is pressed."""
        status    = self._get_status_fn() if self._get_status_fn else {}
        histories = self._get_all_histories_fn() if self._get_all_histories_fn else {}
        trades    = self._get_trades_fn() if self._get_trades_fn else []

        if bot_id not in status:
            return

        bot_ids = list(status.keys())
        text    = self._portfolio_detail_text(bot_id, status, histories, trades)
        buttons = self._portfolio_buttons(bot_ids, selected=bot_id)
        self._edit_message(message_id, text, buttons, chat_id=chat_id)

    def _handle_portfolio_chart_callback(
        self, chat_id: str, bot_id: str
    ) -> None:
        """Send the equity curve chart for a single bot as a new photo message."""
        histories = self._get_all_histories_fn() if self._get_all_histories_fn else {}
        history   = histories.get(bot_id, [])

        if not history or len(history) < 2:
            self.send(
                f"ℹ️ <b>{bot_id}</b>: no hi ha prou dades per generar el gràfic.",
                chat_id=chat_id,
            )
            return

        chart_bytes = self._generate_single_bot_chart(bot_id, history)
        if chart_bytes:
            self._send_photo(chart_bytes, caption=f"📈 Corba d'equity — <b>{bot_id}</b>", chat_id=chat_id)
        else:
            self.send(
                "ℹ️ Gràfic no disponible (matplotlib no instal·lat).",
                chat_id=chat_id,
            )

    # ─────────────────────────────────────────────────────────────────────
    # Control commands
    # ─────────────────────────────────────────────────────────────────────

    def _valid_bot_ids(self) -> list[str]:
        status = self._get_status_fn() if self._get_status_fn else {}
        return list(status.keys())

    def _handle_pause_command(self, chat_id: str, bot_id: str) -> None:
        if bot_id not in self._valid_bot_ids():
            self.send(
                f"⚠️ Bot <code>{bot_id}</code> no trobat.\n"
                f"Bots actius: {self._valid_bot_ids()}",
                chat_id=chat_id,
            )
            return
        paused = self._is_paused_fn(bot_id) if self._is_paused_fn else False
        if paused:
            self.send(
                f"ℹ️ <b>{bot_id}</b> ja està pausat. Usa /resume {bot_id} per reprendre'l.",
                chat_id=chat_id,
            )
            return
        self._send_with_keyboard(
            f"⏸ Pausar <b>{bot_id}</b>?\n\n"
            f"El bot deixarà d'executar senyals. Les posicions obertes es mantenen.",
            [[
                {"text": "✅ Confirmar", "callback_data": f"confirm_pause:{bot_id}"},
                {"text": "❌ Cancel·lar", "callback_data": "cancel_action"},
            ]],
            chat_id=chat_id,
        )

    def _handle_resume_command(self, chat_id: str, bot_id: str) -> None:
        if bot_id not in self._valid_bot_ids():
            self.send(
                f"⚠️ Bot <code>{bot_id}</code> no trobat.\n"
                f"Bots actius: {self._valid_bot_ids()}",
                chat_id=chat_id,
            )
            return
        paused = self._is_paused_fn(bot_id) if self._is_paused_fn else False
        if not paused:
            self.send(f"ℹ️ <b>{bot_id}</b> no està pausat.", chat_id=chat_id)
            return
        self._send_with_keyboard(
            f"▶️ Reprendre <b>{bot_id}</b>?",
            [[
                {"text": "✅ Confirmar", "callback_data": f"confirm_resume:{bot_id}"},
                {"text": "❌ Cancel·lar", "callback_data": "cancel_action"},
            ]],
            chat_id=chat_id,
        )

    def _handle_close_command(self, chat_id: str, bot_id: str) -> None:
        if bot_id not in self._valid_bot_ids():
            self.send(
                f"⚠️ Bot <code>{bot_id}</code> no trobat.\n"
                f"Bots actius: {self._valid_bot_ids()}",
                chat_id=chat_id,
            )
            return
        status = self._get_status_fn() if self._get_status_fn else {}
        btc    = status.get(bot_id, {}).get("btc_balance", 0.0)
        if btc < 1e-6:
            self.send(
                f"ℹ️ <b>{bot_id}</b> no té cap posició oberta (BTC = 0).",
                chat_id=chat_id,
            )
            return
        self._send_with_keyboard(
            f"🚨 Tancar posició de <b>{bot_id}</b>?\n\n"
            f"Es vendrà tot el BTC ({btc:.6f} BTC) al preu de mercat actual.",
            [[
                {"text": "✅ Confirmar", "callback_data": f"confirm_close:{bot_id}"},
                {"text": "❌ Cancel·lar", "callback_data": "cancel_action"},
            ]],
            chat_id=chat_id,
        )

    def _handle_reset_command(self, chat_id: str, bot_id: str | None) -> None:
        status = self._get_status_fn() if self._get_status_fn else {}
        if bot_id and bot_id not in status:
            self.send(
                f"⚠️ Bot <code>{bot_id}</code> no trobat.\n"
                f"Bots actius: {list(status.keys())}",
                chat_id=chat_id,
            )
            return

        targets = [bot_id] if bot_id else list(status.keys())
        lines   = []
        for bid in targets:
            hist   = (self._get_all_histories_fn() or {}).get(bid, [])
            trades = [t for t in (self._get_trades_fn() or []) if t.get("bot_id") == bid]
            lines.append(f"  • <b>{bid}</b>: {len(hist)} ticks, {len(trades)} trades")

        scope = f"<b>{bot_id}</b>" if bot_id else "<b>TOTS els bots</b>"
        cb    = f"confirm_reset:{bot_id}" if bot_id else "confirm_reset_all"
        self._send_with_keyboard(
            f"🗑 Reiniciar capital de {scope}?\n\n"
            f"S'esborrarà tot l'historial:\n" + "\n".join(lines) + "\n\n"
            "El runner es reiniciarà i els bots afectats tornaran a 10.000€.",
            [[
                {"text": "✅ Confirmar", "callback_data": cb},
                {"text": "❌ Cancel·lar", "callback_data": "cancel_action"},
            ]],
            chat_id=chat_id,
        )

    def _handle_restart_command(self, chat_id: str) -> None:
        self._send_with_keyboard(
            "🔄 Reiniciar el demo runner?\n\n"
            "El procés es reiniciarà i recuperarà l'estat de la DB (capital i posicions conservats).",
            [[
                {"text": "✅ Confirmar", "callback_data": "confirm_restart"},
                {"text": "❌ Cancel·lar", "callback_data": "cancel_action"},
            ]],
            chat_id=chat_id,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Command routing
    # ─────────────────────────────────────────────────────────────────────

    def _handle_command(self, text: str, chat_id: str) -> None:
        parts = text.strip().split()
        cmd   = parts[0].lower()
        arg   = parts[1] if len(parts) > 1 else ""
        try:
            if cmd == "/status":
                self._handle_status(chat_id)
            elif cmd == "/portfolio":
                self._handle_portfolio(chat_id)
            elif cmd == "/compare":
                self._handle_compare(chat_id)
            elif cmd == "/trades":
                self._handle_trades(chat_id)
            elif cmd == "/health":
                self._handle_health(chat_id)
            elif cmd == "/pause":
                if not arg:
                    self.send("Ús: /pause &lt;bot_id&gt;", chat_id=chat_id)
                else:
                    self._handle_pause_command(chat_id, arg)
            elif cmd == "/resume":
                if not arg:
                    self.send("Ús: /resume &lt;bot_id&gt;", chat_id=chat_id)
                else:
                    self._handle_resume_command(chat_id, arg)
            elif cmd == "/close":
                if not arg:
                    self.send("Ús: /close &lt;bot_id&gt;", chat_id=chat_id)
                else:
                    self._handle_close_command(chat_id, arg)
            elif cmd == "/restart":
                self._handle_restart_command(chat_id)
            elif cmd == "/reset":
                self._handle_reset_command(chat_id, arg if arg else None)
            elif cmd == "/help":
                self.send(
                    "📖 <b>Comandes disponibles</b>\n\n"
                    "<b>Informació</b>\n"
                    "/status      — resum de tots els bots (retorn, alpha, €)\n"
                    "/portfolio   — detall interactiu per bot + 📈 corba equity\n"
                    "/compare     — gràfic evolució + rànquing complet\n"
                    "/trades      — últims trades amb P&amp;L i confiança\n"
                    "/health      — estat de fonts de dades i sistemes\n\n"
                    "<b>Control</b>\n"
                    "/pause &lt;bot&gt;   — pausar un bot (manté posicions)\n"
                    "/resume &lt;bot&gt;  — reprendre un bot pausat\n"
                    "/close &lt;bot&gt;   — vendre tota la posició ara\n"
                    "/restart      — reiniciar el runner (estat conservat)\n"
                    "/reset [bot]  — tornar al capital inicial (esborra historial)\n\n"
                    "ℹ️ Tots els missatges d'informació tenen botó 🔄 per refrescar-los sense reescriure la comanda.",
                    chat_id=chat_id,
                )
            else:
                self.send(
                    f"Comanda desconeguda: <code>{text}</code>\n"
                    "Escriu /help per veure les comandes disponibles.",
                    chat_id=chat_id,
                )
        except Exception as exc:
            logger.error(f"Error handling command {cmd}: {exc}", exc_info=True)
            self.send(f"⚠️ Error intern processant <code>{cmd}</code>.", chat_id=chat_id)

    # ─────────────────────────────────────────────────────────────────────
    # Proactive notifications
    # ─────────────────────────────────────────────────────────────────────

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

        # Optional F&G context
        fg_line = ""
        if self._get_health_fn:
            try:
                fg = (self._get_health_fn() or {}).get("fear_greed")
                if fg:
                    fg_line = f"\n🧠 F&amp;G: <b>{fg['value']}</b> ({fg['classification']})"
            except Exception:
                pass

        self.send(
            f"{emoji} <b>{bot_id}</b> — {action_txt} [{now_str}]\n"
            f"💰 Preu: <b>{price:,.2f} USDT</b>  ·  {size_btc:.4f} BTC\n"
            f"📊 Portfolio: <b>{portfolio_value:,.2f} €</b> ({bot_ret:+.1f}%)\n"
            f"🎯 Confiança: {conf_str}{fg_line}\n"
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
            f"Portfolio: <b>{portfolio_value:,.2f} €</b> ({bot_ret:+.1f}%)"
        )

    def notify_circuit_breaker(
        self,
        bot_id: str,
        drawdown_pct: float,
        threshold_pct: float,
        portfolio_value: float,
    ) -> None:
        bot_ret = _bot_return_pct(portfolio_value)
        self.send(
            f"🔴 <b>CIRCUIT BREAKER ACTIVAT</b>\n"
            f"Bot: <b>{bot_id}</b> — AUTO-PAUSAT\n"
            f"Drawdown: <b>{drawdown_pct:.1f}%</b> (llindar: {threshold_pct:.1f}%)\n"
            f"Portfolio: <b>{portfolio_value:,.2f} €</b> ({bot_ret:+.1f}%)\n"
            f"Repren manualment: /resume {bot_id}"
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
        """
        Daily push with bot rankings + BTC performance today + trade count.
        """
        histories  = self._get_all_histories_fn() if self._get_all_histories_fn else {}
        all_trades = self._get_trades_fn() if self._get_trades_fn else []
        now_str    = datetime.now(TZ).strftime("%Y-%m-%d")
        today      = datetime.now(TZ).date()

        # Bot ranking
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
                f"{emoji} <b>{bot_id}</b>: {pv:,.2f}€  "
                f"({bot_ret:+.1f}%  α {alpha:+.1f}%)"
            )

        # BTC change today
        btc_today_pct = None
        for history in histories.values():
            today_rows = [
                r for r in history
                if r["timestamp"].astimezone(TZ).date() == today
            ]
            if len(today_rows) >= 2:
                p0 = today_rows[0]["price"]
                p1 = today_rows[-1]["price"]
                if p0 > 0:
                    btc_today_pct = (p1 / p0 - 1.0) * 100
                break

        # Trades today
        today_trades = [
            t for t in all_trades
            if t["timestamp"].astimezone(TZ).date() == today
        ]
        n_buy  = sum(1 for t in today_trades if t["action"] == "buy")
        n_sell = sum(1 for t in today_trades if t["action"] == "sell")

        lines.append("")
        btc_str = f"{btc_today_pct:+.2f}%" if btc_today_pct is not None else "—"
        lines.append(
            f"📈 BTC avui: <b>{btc_str}</b>  "
            f"|  Trades: <b>{len(today_trades)}</b> "
            f"({n_buy} BUY / {n_sell} SELL)"
        )

        # F&G snapshot
        if self._get_health_fn:
            try:
                fg = (self._get_health_fn() or {}).get("fear_greed")
                if fg:
                    lines.append(f"🧠 F&amp;G: <b>{fg['value']}</b> ({fg['classification']})")
            except Exception:
                pass

        self.send("\n".join(lines))

    # ─────────────────────────────────────────────────────────────────────
    # Polling loop
    # ─────────────────────────────────────────────────────────────────────

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

                        self._answer_callback(cb_id)

                        # ── Refresh callbacks (edit message in-place) ─────

                        if cb_data == "refresh_status" and cb_msg_id:
                            text, buttons = self._status_content()
                            self._edit_message(cb_msg_id, text, buttons, chat_id=cb_chat_id)

                        elif cb_data == "refresh_compare" and cb_msg_id:
                            text, buttons = self._compare_table_content()
                            self._edit_message(cb_msg_id, text, buttons, chat_id=cb_chat_id)

                        elif cb_data == "refresh_trades" and cb_msg_id:
                            text, buttons = self._trades_content()
                            self._edit_message(cb_msg_id, text, buttons, chat_id=cb_chat_id)

                        elif cb_data == "refresh_health" and cb_msg_id:
                            text, buttons = self._health_content()
                            self._edit_message(cb_msg_id, text, buttons, chat_id=cb_chat_id)

                        # ── Portfolio: tab switch ─────────────────────────

                        elif cb_data.startswith("portfolio:") and cb_msg_id:
                            selected_bot = cb_data[len("portfolio:"):]
                            logger.info(f"Portfolio tab: {selected_bot}")
                            self._handle_portfolio_callback(cb_chat_id, cb_msg_id, selected_bot)

                        elif cb_data.startswith("portfolio_refresh:") and cb_msg_id:
                            selected_bot = cb_data[len("portfolio_refresh:"):]
                            logger.info(f"Portfolio refresh: {selected_bot}")
                            self._handle_portfolio_callback(cb_chat_id, cb_msg_id, selected_bot)

                        elif cb_data.startswith("portfolio_chart:"):
                            selected_bot = cb_data[len("portfolio_chart:"):]
                            logger.info(f"Portfolio chart: {selected_bot}")
                            self._handle_portfolio_chart_callback(cb_chat_id, selected_bot)

                        # ── Control confirmations ─────────────────────────

                        elif cb_data.startswith("confirm_pause:"):
                            bot_id = cb_data[len("confirm_pause:"):]
                            if self._pause_fn:
                                self._pause_fn(bot_id)
                            self.send(
                                f"⏸ <b>{bot_id}</b> pausat. Usa /resume {bot_id} per reprendre'l.",
                                chat_id=cb_chat_id,
                            )

                        elif cb_data.startswith("confirm_resume:"):
                            bot_id = cb_data[len("confirm_resume:"):]
                            if self._resume_fn:
                                self._resume_fn(bot_id)
                            self.send(f"▶️ <b>{bot_id}</b> reprès.", chat_id=cb_chat_id)

                        elif cb_data.startswith("confirm_close:"):
                            bot_id = cb_data[len("confirm_close:"):]
                            if self._close_fn:
                                self._close_fn(bot_id)
                            self.send(
                                f"🚨 Tancant posició de <b>{bot_id}</b>... "
                                "Rebràs confirmació quan s'executi.",
                                chat_id=cb_chat_id,
                            )

                        elif cb_data == "confirm_restart":
                            self.send("🔄 Reiniciant el runner... Torna en uns segons.", chat_id=cb_chat_id)
                            time.sleep(1)
                            if self._restart_fn:
                                self._restart_fn()

                        elif cb_data.startswith("confirm_reset:"):
                            bot_id = cb_data[len("confirm_reset:"):]
                            if self._reset_fn:
                                self._reset_fn(bot_id)
                            self.send(
                                f"🗑 <b>{bot_id}</b> reiniciat a 10.000€. Reiniciant runner...",
                                chat_id=cb_chat_id,
                            )
                            time.sleep(1)
                            if self._restart_fn:
                                self._restart_fn()

                        elif cb_data == "confirm_reset_all":
                            if self._reset_fn:
                                self._reset_fn(None)
                            self.send(
                                "🗑 Tots els bots reiniciats a 10.000€. Reiniciant runner...",
                                chat_id=cb_chat_id,
                            )
                            time.sleep(1)
                            if self._restart_fn:
                                self._restart_fn()

                        elif cb_data == "cancel_action":
                            self.send("❌ Acció cancel·lada.", chat_id=cb_chat_id)

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                time.sleep(5)
            except Exception as exc:
                logger.error(f"Telegram listener error: {exc}")
                time.sleep(5)

    def _drain_pending_updates(self) -> None:
        """
        Consume all updates already in the Telegram queue without processing them.
        Called once on startup so stale callbacks (e.g. confirm_reset that triggered
        a restart) are not re-executed.
        """
        try:
            while True:
                r = requests.get(
                    f"{self.base_url}/getUpdates",
                    params={"offset": self._last_update_id + 1, "timeout": 0},
                    timeout=10,
                )
                updates = r.json().get("result", [])
                if not updates:
                    break
                for upd in updates:
                    self._last_update_id = upd["update_id"]
            if self._last_update_id > 0:
                logger.info(f"Telegram: drained pending updates up to id={self._last_update_id}")
        except Exception as exc:
            logger.warning(f"Telegram: could not drain pending updates: {exc}")

    def start_listener(self) -> None:
        """Start the polling loop in a background daemon thread."""
        self._drain_pending_updates()
        self._listener_thread = threading.Thread(
            target=self._poll_updates, daemon=True, name="telegram-listener"
        )
        self._listener_thread.start()
