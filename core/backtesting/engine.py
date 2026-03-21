# core/backtesting/engine.py
import logging
from core.interfaces.base_bot import BaseBot
from core.engine.runner import Runner
from core.backtesting.metrics import BacktestMetrics
from exchanges.paper import PaperExchange

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Executes a complete backtest and logs results via the standard logger."""

    def __init__(self, bot: BaseBot, exchange_config_path: str = "config/exchanges/paper.yaml"):
        self.bot = bot
        self.exchange_config_path = exchange_config_path

    def run(
        self,
        symbol: str,
        timeframe: str,
        start_date: str | None = None,
        end_date: str | None = None,
        desc: str | None = None,
    ) -> BacktestMetrics:
        exchange = PaperExchange(config_path=self.exchange_config_path)
        initial_capital = exchange.get_balance("USDT")

        runner = Runner(bot=self.bot, exchange=exchange)
        history = runner.run(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            desc=desc,
        )

        metrics = BacktestMetrics(history=history, initial_capital=initial_capital, timeframe=timeframe)
        summary = metrics.summary()

        logger.info("=== BACKTEST RESULTS ===")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")

        return metrics
