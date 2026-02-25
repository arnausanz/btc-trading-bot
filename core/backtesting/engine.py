# core/backtesting/engine.py
import logging
from core.interfaces.base_bot import BaseBot
from core.engine.runner import Runner
from core.backtesting.metrics import BacktestMetrics
from exchanges.paper import PaperExchange

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Executa un backtest complet d'un bot i retorna les mètriques de rendiment.
    Usa el Runner i el PaperExchange internament.
    """

    def __init__(
        self,
        bot: BaseBot,
        exchange_config_path: str = "config/exchanges/paper.yaml",
    ):
        self.bot = bot
        self.exchange_config_path = exchange_config_path

    def run(self, symbol: str, timeframe: str) -> BacktestMetrics:
        """
        Executa el backtest i retorna les mètriques.
        Crea un exchange net per cada execució — cap estat residual entre backtests.
        """
        exchange = PaperExchange(config_path=self.exchange_config_path)
        initial_capital = exchange.get_balance("USDT")

        runner = Runner(bot=self.bot, exchange=exchange)
        history = runner.run(symbol=symbol, timeframe=timeframe)

        metrics = BacktestMetrics(history=history, initial_capital=initial_capital)

        logger.info("=== BACKTEST RESULTS ===")
        for key, value in metrics.summary().items():
            logger.info(f"  {key}: {value}")

        return metrics