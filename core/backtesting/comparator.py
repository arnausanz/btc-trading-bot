# core/backtesting/comparator.py
import logging
from core.interfaces.base_bot import BaseBot
from core.backtesting.engine import BacktestEngine
from core.backtesting.metrics import BacktestMetrics

logger = logging.getLogger(__name__)


class BotComparator:
    """
    Executa backtests de múltiples bots sobre el mateix període i capital
    i retorna un ranking objectiu basat en les mètriques.
    Tots els bots s'executen amb les mateixes condicions — comparació justa.
    """

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
        """
        Executa el backtest de cada bot i retorna els resultats ordenats per Sharpe Ratio.
        """
        results = []

        for bot in self.bots:
            logger.info(f"Executant backtest: {bot.bot_id}...")
            engine = BacktestEngine(
                bot=bot,
                exchange_config_path=self.exchange_config_path,
            )
            metrics = engine.run(symbol=self.symbol, timeframe=self.timeframe)
            summary = metrics.summary()
            summary["bot_id"] = bot.bot_id
            results.append(summary)

        # Ordenem per Sharpe Ratio descendent
        results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)

        self._print_ranking(results)
        return results

    def _print_ranking(self, results: list[dict]) -> None:
        logger.info("=== RANKING DE BOTS ===")
        logger.info(f"{'Bot':<20} {'Return%':>10} {'Sharpe':>10} {'Drawdown%':>12} {'Calmar':>10} {'WinRate%':>10}")
        logger.info("-" * 75)
        for r in results:
            logger.info(
                f"{r['bot_id']:<20} "
                f"{r['total_return_pct']:>10.2f} "
                f"{r['sharpe_ratio']:>10.3f} "
                f"{r['max_drawdown_pct']:>12.2f} "
                f"{r['calmar_ratio']:>10.3f} "
                f"{r['win_rate_pct']:>10.2f}"
            )