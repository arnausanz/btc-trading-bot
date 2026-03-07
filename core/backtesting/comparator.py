# core/backtesting/comparator.py
import logging
from core.interfaces.base_bot import BaseBot
from core.backtesting.engine import BacktestEngine
from core.backtesting.metrics import BacktestMetrics
from core.config import TRAIN_UNTIL, TEST_FROM

logger = logging.getLogger(__name__)


class BotComparator:
    """
    Compara múltiples bots en backtest.

    WALK-FORWARD: si train_until i test_from estan configurats (per defecte sí),
    cada bot es backtesta en DOS períodes separats:
    - Train (in-sample):  fins a train_until — dades que els bots "coneixien"
    - Test (out-of-sample): des de test_from — dades mai vistes durant optimització

    Aquesta separació permet detectar si un bot ha sobreajustat (overfit) o
    si les seves mètriques de train es mantenen en dades noves.
    """

    def __init__(
        self,
        bots: list[BaseBot],
        symbol: str,
        timeframe: str,
        exchange_config_path: str = "config/exchanges/paper.yaml",
        train_until: str = TRAIN_UNTIL,
        test_from: str = TEST_FROM,
    ):
        self.bots = bots
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange_config_path = exchange_config_path
        self.train_until = train_until
        self.test_from = test_from

    def run(self) -> list[dict]:
        results = []
        for bot in self.bots:
            print(f"\n▶ {bot.bot_id}")

            # ─── Període de TRAIN ──────────────────────────────────────────────
            print(f"  [train] fins {self.train_until}")
            engine_train = BacktestEngine(bot=bot, exchange_config_path=self.exchange_config_path)
            train_metrics = engine_train.run(
                symbol=self.symbol,
                timeframe=self.timeframe,
                end_date=self.train_until,
            )
            train_summary = train_metrics.summary()

            # ─── Període de TEST ───────────────────────────────────────────────
            print(f"  [test]  des de {self.test_from}")
            engine_test = BacktestEngine(bot=bot, exchange_config_path=self.exchange_config_path)
            test_metrics = engine_test.run(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_date=self.test_from,
            )
            test_summary = test_metrics.summary()

            results.append({
                "bot_id": bot.bot_id,
                "train": train_summary,
                "test": test_summary,
                "train_trades": self._count_trades(train_metrics),
                "test_trades": self._count_trades(test_metrics),
            })

        results.sort(key=lambda x: x["test"]["sharpe_ratio"], reverse=True)
        self._print_summary(results)
        return results

    def _count_trades(self, metrics: BacktestMetrics) -> dict:
        signals = metrics.df["signal"].astype(str)
        return {
            "buy":  int(signals.str.contains("buy").sum()),
            "sell": int(signals.str.contains("sell").sum()),
            "hold": int(signals.str.contains("hold").sum()),
        }

    def _print_summary(self, results: list[dict]) -> None:
        W = 90
        print("\n" + "═" * W)
        print(f"  RANKING DE BOTS  (train fins {self.train_until} | test des de {self.test_from})")
        print("═" * W)

        # ─── Taula de mètriques ────────────────────────────────────────────────
        hdr = f"  {'Bot':<20} {'Prd':>5} {'Return%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'Calmar':>8} {'WinRate%':>10}"
        print(f"\n{hdr}")
        print("  " + "─" * (W - 2))

        for r in results:
            for period_key, label in [("train", "TRAIN"), ("test", " TEST")]:
                s = r[period_key]
                print(
                    f"  {r['bot_id']:<20} {label:>5} "
                    f"{s['total_return_pct']:>9.2f} "
                    f"{s['sharpe_ratio']:>8.3f} "
                    f"{s['max_drawdown_pct']:>8.2f} "
                    f"{s['calmar_ratio']:>8.3f} "
                    f"{s['win_rate_pct']:>10.2f}"
                )
            print("  " + "·" * (W - 2))

        # ─── Taula de trades ───────────────────────────────────────────────────
        print(f"\n  {'Bot':<20} {'Prd':>5} {'BUY':>8} {'SELL':>8} {'HOLD':>10} {'Capital Final':>15}")
        print("  " + "─" * (W - 2))
        for r in results:
            for period_key, label in [("train", "TRAIN"), ("test", " TEST")]:
                tc = r[f"{period_key}_trades"]
                cap = r[period_key]["final_capital"]
                print(
                    f"  {r['bot_id']:<20} {label:>5} "
                    f"{tc['buy']:>8} "
                    f"{tc['sell']:>8} "
                    f"{tc['hold']:>10} "
                    f"{cap:>13.2f} USDT"
                )
            print("  " + "·" * (W - 2))

        print("\n" + "═" * W)
