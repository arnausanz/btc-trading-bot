# core/backtesting/optimizer.py
import logging
import optuna
import yaml
from tqdm import tqdm
from core.backtesting.engine import BacktestEngine
from core.config import TRAIN_UNTIL

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

# Suprimim loggers sorollosos durant l'optimització
for _noisy in ("core.engine.runner", "exchanges.paper",
               "data.processing.technical", "data.observation.builder"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


class BotOptimizer:
    """
    Optimitza els hiperparàmetres d'un bot usant Optuna.
    Cada trial és un backtest complet amb una configuració diferent.
    El resultat de cada trial és el Sharpe Ratio — volem maximitzar-lo.

    WALK-FORWARD: l'optimització es fa EXCLUSIVAMENT sobre el període de train
    (fins a train_until). Mai veu les dades de test, evitant overfitting al futur.
    """

    def __init__(
        self,
        bot_class,
        base_config_path: str,
        param_space: dict,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        n_trials: int = 50,
        train_until: str = TRAIN_UNTIL,
    ):
        self.bot_class = bot_class
        self.base_config_path = base_config_path
        self.param_space = param_space
        self.symbol = symbol
        self.timeframe = timeframe
        self.n_trials = n_trials
        self.train_until = train_until

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Funció objectiu que Optuna minimitza/maximitza.
        Construeix una config amb els paràmetres del trial i executa el backtest
        limitat al període de train (end_date=train_until).
        """
        with open(self.base_config_path) as f:
            config = yaml.safe_load(f)

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

        temp_config_path = f"/tmp/optuna_trial_{trial.number}.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f)

        try:
            bot = self.bot_class(config_path=temp_config_path)
            engine = BacktestEngine(bot=bot)
            metrics = engine.run(
                symbol=self.symbol,
                timeframe=self.timeframe,
                end_date=self.train_until,
                desc="train",
            )
            return metrics.summary()["sharpe_ratio"]
        except Exception as e:
            logger.warning(f"Trial {trial.number} ha fallat: {e}")
            return -999.0

    def run(self) -> optuna.Study:
        """
        Executa l'optimització amb barra de progrés tqdm.
        Mostra trial actual, temps transcorregut, ETA i millor Sharpe fins ara.
        """
        bot_name = self.bot_class.__name__
        study = optuna.create_study(direction="maximize")

        with tqdm(
            total=self.n_trials,
            desc=f"Optimitzant {bot_name}",
            unit="trial",
            dynamic_ncols=True,
        ) as pbar:
            def _callback(study: optuna.Study, trial: optuna.Trial) -> None:
                best = study.best_value if study.best_value > -900 else float("nan")
                pbar.set_postfix({"best_sharpe": f"{best:.3f}"}, refresh=False)
                pbar.update(1)

            study.optimize(self._objective, n_trials=self.n_trials, callbacks=[_callback])

        tqdm.write(
            f"  ✓ {bot_name} — Sharpe: {study.best_value:.3f} | "
            f"Params: {study.best_params}"
        )
        return study
