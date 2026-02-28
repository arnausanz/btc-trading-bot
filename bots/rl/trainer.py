# bots/rl/trainer.py
import logging
import mlflow
import numpy as np
import yaml
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from bots.rl.environment import BtcTradingEnv
from data.processing.technical import compute_features

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


class ProgressCallback(BaseCallback):
    """Callback que mostra progrés i registra mètriques a MLflow."""

    def __init__(self, total_timesteps: int, log_every: int = 10000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.log_every = log_every

    def _on_step(self) -> bool:
        if self.n_calls % max(1, self.total_timesteps // 1000) == 0:
            pct = self.n_calls / self.total_timesteps * 100
            filled = int(pct / 5)
            bar = "█" * filled + "░" * (20 - filled)
            print(f"\r  [{bar}] {pct:5.1f}% ({self.n_calls}/{self.total_timesteps})", end="", flush=True)
        return True


class RLTrainer:
    """
    Entrena un agent RL (DQN o PPO) sobre el BtcTradingEnv.
    Usa 80% de les dades per entrenar i 20% per validar.
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def run(self) -> dict:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.config["experiment_name"])

        with mlflow.start_run():
            mlflow.log_params({
                "model_type": self.config["model_type"],
                "total_timesteps": self.config["model"]["total_timesteps"],
                "lookback": self.config["environment"]["lookback"],
                "timeframe": self.config["data"]["timeframe"],
            })

            # Carrega i divideix les dades
            df = compute_features(
                symbol=self.config["data"]["symbol"],
                timeframe=self.config["data"]["timeframe"],
            )

            # Només features numèriques, exclou close per evitar leakage directe
            feature_cols = [c for c in df.columns if c != "close"]
            df_features = df[["close"] + feature_cols]

            split = int(len(df_features) * self.config["data"]["train_pct"])
            df_train = df_features.iloc[:split]
            df_val = df_features.iloc[split:]

            logger.info(f"Train: {len(df_train)} files | Val: {len(df_val)} files")

            # Crea els environments
            env_config = self.config["environment"]
            train_env = BtcTradingEnv(df=df_train, **env_config)
            val_env = BtcTradingEnv(df=df_val, **env_config)

            # Crea el model
            model_type = self.config["model_type"].upper()
            model_class = DQN if model_type == "DQN" else PPO

            model = model_class(
                "MlpPolicy",
                train_env,
                learning_rate=self.config["model"]["learning_rate"],
                batch_size=self.config["model"]["batch_size"],
                verbose=0,
            )

            # Entrena
            logger.info(f"Entrenant {model_type}...")
            callback = ProgressCallback(
                total_timesteps=self.config["model"]["total_timesteps"]
            )
            model.learn(
                total_timesteps=self.config["model"]["total_timesteps"],
                callback=callback,
            )
            print()

            # Valida
            metrics = self._validate(model, val_env)
            mlflow.log_metrics(metrics)

            # Guarda
            model.save(self.config["output"]["model_path"])
            logger.info(f"Model guardat a {self.config['output']['model_path']}")

            return metrics

    def _validate(self, model, env: BtcTradingEnv) -> dict:
        """Executa l'agent sobre el set de validació i retorna mètriques."""
        obs, _ = env.reset()
        done = False
        portfolio_values = [env.initial_capital]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            portfolio_values.append(info["portfolio_value"])

        final_value = portfolio_values[-1]
        total_return = (final_value - env.initial_capital) / env.initial_capital * 100

        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = ((values - peak) / peak * 100).min()

        logger.info(f"Validació → Return: {total_return:.2f}% | Drawdown: {drawdown:.2f}% | Trades: {env.trades}")

        return {
            "val_return_pct": round(total_return, 2),
            "val_max_drawdown_pct": round(float(drawdown), 2),
            "val_trades": env.trades,
            "val_final_capital": round(final_value, 2),
        }