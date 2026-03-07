# core/backtesting/rl_optimizer.py
import copy
import logging
import mlflow
import optuna
import yaml
from core.interfaces.base_rl_agent import BaseRLAgent

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


class RLOptimizer:
    """
    Optimizes hyperparameters of RL agents with Optuna.
    Each trial = short training (probe_timesteps) + validation.
    Objective metric: val_return_pct | val_max_drawdown_pct.

    Uses probe_timesteps << total_timesteps to perform many trials quickly.
    The best config is re-trained afterward with full total_timesteps.
    """

    def __init__(self, optimization_config_path: str):
        with open(optimization_config_path) as f:
            self.opt_cfg = yaml.safe_load(f)

        with open(self.opt_cfg["base_config"]) as f:
            self.base_train_cfg = yaml.safe_load(f)

        self.model_type = self.opt_cfg["model_type"]
        self.n_trials = self.opt_cfg["n_trials"]
        self.metric = self.opt_cfg.get("metric", "val_return_pct")
        self.direction = self.opt_cfg.get("direction", "maximize")
        self.probe_timesteps = self.opt_cfg.get("probe_timesteps", 20000)
        self.search_space = self.opt_cfg["search_space"]

    def _sample_params(self, trial: optuna.Trial) -> dict:
        params = {}
        for name, spec in self.search_space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"],
                    log=spec.get("log", False)
                )
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
        return params

    def _build_config(self, params: dict) -> dict:
        config = copy.deepcopy(self.base_train_cfg)
        for name, value in params.items():
            # Environment parameters go to config["environment"]
            if name in ("lookback", "reward_type", "reward_scaling"):
                config["environment"][name] = value
            else:
                config["model"][name] = value
        return config

    def _objective(self, trial: optuna.Trial) -> float:
        from bots.rl.agents import SACAgent, PPOAgent
        from bots.rl.environment import BtcTradingEnvDiscrete, BtcTradingEnvContinuous
        from bots.rl.rewards import builtins  # noqa: registra builtins
        from data.processing.technical import compute_features

        _AGENT_REGISTRY: dict[str, type[BaseRLAgent]] = {
            "sac": SACAgent,
            "ppo": PPOAgent,
        }
        _ENV_REGISTRY = {
            "sac": BtcTradingEnvContinuous,
            "ppo": BtcTradingEnvDiscrete,
        }

        params = self._sample_params(trial)
        config = self._build_config(params)

        try:
            mlflow.end_run()
            df = compute_features(
                symbol=config["data"]["symbol"],
                timeframe=config["data"]["timeframe"],
            )
            split = int(len(df) * config["data"]["train_pct"])
            df_train = df.iloc[:split]
            df_val = df.iloc[split:]

            env_class = _ENV_REGISTRY[self.model_type]
            env_cfg = config["environment"]
            train_env = env_class(df=df_train, **env_cfg)
            val_env = env_class(df=df_val, **env_cfg)

            agent = _AGENT_REGISTRY[self.model_type].from_config(config)
            metrics = agent.train(
                train_env=train_env,
                val_env=val_env,
                total_timesteps=self.probe_timesteps,  # ← short for quick optimization
            )

            score = metrics.get(self.metric, -999.0)
            logger.info(
                f"  Trial {trial.number:03d} | "
                f"{self.metric}={score:.4f} | "
                f"params={params}"
            )
            return score

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.warning(f"  Trial {trial.number} failed: {e}")
            return -999.0
        finally:
            mlflow.end_run()

    def run(self) -> optuna.Study:
        logger.info(
            f"=== Optimizing {self.model_type} | "
            f"{self.n_trials} trials | "
            f"probe={self.probe_timesteps} steps | "
            f"objective: {self.metric} ==="
        )
        study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(self._objective, n_trials=self.n_trials)

        logger.info(f"  Best {self.metric}: {study.best_value:.4f}")
        logger.info(f"  Best parameters: {study.best_params}")
        return study

    def best_config(self, study: optuna.Study) -> dict:
        return self._build_config(study.best_params)

    def save_best_config(self, study: optuna.Study, output_path: str) -> None:
        best = self.best_config(study)
        best["experiment_name"] = f"{self.model_type}_optimized"
        with open(output_path, "w") as f:
            yaml.dump(best, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"  Optimized config saved to {output_path}")