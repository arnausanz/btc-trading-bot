# core/backtesting/optimizer.py
import logging
import optuna
import yaml
from typing import Callable
from core.backtesting.engine import BacktestEngine
from core.interfaces.base_bot import BaseBot

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


class BotOptimizer:
    """
    Optimitza els hiperparàmetres d'un bot usant Optuna.
    Cada trial és un backtest complet amb una configuració diferent.
    El resultat de cada trial és el Sharpe Ratio — volem maximitzar-lo.
    """

    def __init__(
        self,
        bot_class,
        base_config_path: str,
        param_space: dict,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        n_trials: int = 50,
    ):
        self.bot_class = bot_class
        self.base_config_path = base_config_path
        self.param_space = param_space
        self.symbol = symbol
        self.timeframe = timeframe
        self.n_trials = n_trials

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Funció objectiu que Optuna minimitza/maximitza.
        Construeix una config amb els paràmetres del trial i executa el backtest.
        """
        # Carrega la config base
        with open(self.base_config_path) as f:
            config = yaml.safe_load(f)

        # Substitueix els paràmetres que Optuna proposa
        for param_name, param_def in self.param_space.items():
            if param_def["type"] == "int":
                config[param_name] = trial.suggest_int(
                    param_name, param_def["low"], param_def["high"]
                )
            elif param_def["type"] == "float":
                config[param_name] = trial.suggest_float(
                    param_name, param_def["low"], param_def["high"]
                )
            elif param_def["type"] == "categorical":
                config[param_name] = trial.suggest_categorical(
                    param_name, param_def["choices"]
                )

        if "ema_fast" in config and "ema_slow" in config:
            if config["ema_fast"] >= config["ema_slow"] - 10:
                raise optuna.TrialPruned()

        # Escriu la config temporal
        temp_config_path = f"/tmp/optuna_trial_{trial.number}.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f)

        try:
            bot = self.bot_class(config_path=temp_config_path)
            engine = BacktestEngine(bot=bot)
            metrics = engine.run(symbol=self.symbol, timeframe=self.timeframe)
            sharpe = metrics.summary()["sharpe_ratio"]
            return sharpe
        except Exception as e:
            logger.warning(f"Trial {trial.number} ha fallat: {e}")
            return -999.0

    def run(self) -> optuna.Study:
        """
        Executa l'optimització i retorna l'study amb tots els resultats.
        """
        study = optuna.create_study(direction="maximize")
        logging.getLogger("core.engine.runner").setLevel(logging.WARNING)
        logging.getLogger("exchanges.paper").setLevel(logging.WARNING)
        logging.getLogger("data.processing.technical").setLevel(logging.WARNING)
        study.optimize(self._objective, n_trials=self.n_trials)

        logger.info(f"=== OPTIMITZACIÓ COMPLETADA ===")
        logger.info(f"Millors paràmetres: {study.best_params}")
        logger.info(f"Millor Sharpe: {study.best_value:.4f}")

        return study