# core/backtesting/engine.py
import logging
import mlflow
from core.interfaces.base_bot import BaseBot
from core.engine.runner import Runner
from core.backtesting.metrics import BacktestMetrics
from exchanges.paper import PaperExchange
from core.config import MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Executes a complete backtest and automatically logs to MLflow."""

    def __init__(self, bot: BaseBot, exchange_config_path: str = "config/exchanges/paper.yaml"):
        self.bot = bot
        self.exchange_config_path = exchange_config_path
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

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

        mlflow.set_experiment(self.bot.bot_id)
        with mlflow.start_run():
            mlflow.log_params({
                "symbol": symbol,
                "timeframe": timeframe,
                "initial_capital": initial_capital,
                "start_date": start_date or "all",
                "end_date": end_date or "all",
                **{k: v for k, v in self.bot.config.items() if isinstance(v, (str, int, float, bool))},
            })

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

            mlflow.log_metrics({
                "total_return_pct": summary["total_return_pct"],
                "sharpe_ratio": summary["sharpe_ratio"],
                "max_drawdown_pct": summary["max_drawdown_pct"],
                "calmar_ratio": summary["calmar_ratio"],
                "win_rate_pct": summary["win_rate_pct"],
                "final_capital": summary["final_capital"],
            })

            logger.info("=== BACKTEST RESULTS ===")
            for key, value in summary.items():
                logger.info(f"  {key}: {value}")

        return metrics