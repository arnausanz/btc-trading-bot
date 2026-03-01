# core/backtesting/comparator.py
import logging
from core.interfaces.base_bot import BaseBot
from core.backtesting.engine import BacktestEngine
from core.backtesting.metrics import BacktestMetrics

logger = logging.getLogger(__name__)


class BotComparator:
    def __init__(
        self,
        bots: list[BaseBot],
        symbol: str,
        timeframe: str,
        exchange_config_path: str = "config/exchanges/paper.yaml",
    ):
        self.bots = bots
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange_config_path = exchange_config_path

    def run(self) -> list[dict]:
        results = []
        for bot in self.bots:
            print(f"\n▶ Executant backtest: {bot.bot_id}")
            engine = BacktestEngine(bot=bot, exchange_config_path=self.exchange_config_path)
            metrics = engine.run(symbol=self.symbol, timeframe=self.timeframe)
            summary = metrics.summary()
            summary["bot_id"] = bot.bot_id
            summary["trade_counts"] = self._count_trades(metrics)
            results.append(summary)

        results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
        self._print_summary(results)
        return results

    def _count_trades(self, metrics: BacktestMetrics) -> dict:
        signals = metrics.df["signal"].astype(str)
        return {
            "buy":  signals.str.contains("buy").sum(),
            "sell": signals.str.contains("sell").sum(),
            "hold": signals.str.contains("hold").sum(),
        }

    def _print_summary(self, results: list[dict]) -> None:
        print("\n" + "═" * 80)
        print("  RANKING FINAL DE BOTS")
        print("═" * 80)

        print(f"\n  {'Bot':<20} {'Return%':>9} {'Sharpe':>8} {'Drawdown%':>11} {'Calmar':>8} {'WinRate%':>10}")
        print("  " + "─" * 70)
        for r in results:
            print(
                f"  {r['bot_id']:<20} "
                f"{r['total_return_pct']:>9.2f} "
                f"{r['sharpe_ratio']:>8.3f} "
                f"{r['max_drawdown_pct']:>11.2f} "
                f"{r['calmar_ratio']:>8.3f} "
                f"{r['win_rate_pct']:>10.2f}"
            )

        print(f"\n  {'Bot':<20} {'BUY':>8} {'SELL':>8} {'HOLD':>10} {'Capital Final':>15}")
        print("  " + "─" * 65)
        for r in results:
            tc = r["trade_counts"]
            print(
                f"  {r['bot_id']:<20} "
                f"{tc['buy']:>8} "
                f"{tc['sell']:>8} "
                f"{tc['hold']:>10} "
                f"{r['final_capital']:>13.2f} USDT"
            )

        print("\n" + "═" * 80)