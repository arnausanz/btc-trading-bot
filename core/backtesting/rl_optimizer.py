# core/backtesting/rl_optimizer.py
import copy
import logging
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

    Llegeix des del YAML unificat (config/models/*.yaml) -- un sol fitxer conte
    tota la informacio: features, training, optimization search space i bot config.

    Uses probe_timesteps << total_timesteps to perform many trials quickly.
    The best config is re-trained afterward with full total_timesteps.
    """

    def __init__(self, config_path: str):
        """
        Args:
            config_path: ruta al YAML unificat (config/models/{agent}.yaml)
        """
        self.config_path = config_path
        with open(config_path) as f:
            self.unified_cfg = yaml.safe_load(f)

        opt = self.unified_cfg["optimization"]
        self.model_type = self.unified_cfg["model_type"]
        self.n_trials = opt["n_trials"]
        self.metric = opt.get("metric", "val_return_pct")
        self.direction = opt.get("direction", "maximize")
        self.probe_timesteps = opt.get("probe_timesteps", 20000)
        self.search_space = opt["search_space"]

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
        """Aplica parametres del trial al config unificat."""
        config = copy.deepcopy(self.unified_cfg)
        for name, value in params.items():
            # Parametres d'entorn
            if name in ("lookback", "reward_type", "reward_scaling", "stop_atr_multiplier"):
                config["training"]["environment"][name] = value
                # Sincronitzem lookback a features.lookback tambe
                if name == "lookback":
                    config["features"]["lookback"] = value
            else:
                # Hiperparametres del model
                config["training"]["model"][name] = value
        return config

    def _objective(self, trial: optuna.Trial) -> float:
        from bots.rl.agents import SACAgent, PPOAgent, TD3Agent
        from bots.rl.environment import BtcTradingEnvDiscrete, BtcTradingEnvContinuous
        from bots.rl.environment_professional import (
            BtcTradingEnvProfessionalDiscrete,
            BtcTradingEnvProfessionalContinuous,
        )
        from bots.rl.rewards import builtins, professional  # noqa: registers rewards
        from bots.rl.rewards import advanced                # noqa: registers regime_adaptive

        _AGENT_REGISTRY: dict[str, type[BaseRLAgent]] = {
            "sac":              SACAgent,
            "ppo":              PPOAgent,
            "ppo_professional": PPOAgent,
            "sac_professional": SACAgent,
            "td3_professional": TD3Agent,
            "td3_multiframe":   TD3Agent,
        }
        _ENV_REGISTRY = {
            "sac":              BtcTradingEnvContinuous,
            "ppo":              BtcTradingEnvDiscrete,
            "ppo_professional": BtcTradingEnvProfessionalDiscrete,
            "sac_professional": BtcTradingEnvProfessionalContinuous,
            "td3_professional": BtcTradingEnvProfessionalContinuous,
            "td3_multiframe":   BtcTradingEnvProfessionalContinuous,
        }
        _MULTIFRAME_TYPES = {"td3_multiframe"}

        params = self._sample_params(trial)
        config = self._build_config(params)

        try:
            if self.model_type in _MULTIFRAME_TYPES:
                from data.processing.multiframe_builder import MultiFrameFeatureBuilder
                df = MultiFrameFeatureBuilder.from_config(config).build()
            else:
                from data.processing.feature_builder import FeatureBuilder
                df = FeatureBuilder.from_config(config).build()

            train_cfg = config["training"]
            split = int(len(df) * train_cfg["train_pct"])
            df_train = df.iloc[:split]
            df_val = df.iloc[split:]

            env_class = _ENV_REGISTRY[self.model_type]
            # Sincronitzem lookback de features a l'entorn
            env_cfg = {**train_cfg["environment"], "lookback": config["features"]["lookback"]}
            train_env = env_class(df=df_train, **env_cfg)
            val_env = env_class(df=df_val, **env_cfg)

            agent = _AGENT_REGISTRY[self.model_type].from_config(train_cfg)
            metrics = agent.train(
                train_env=train_env,
                val_env=val_env,
                total_timesteps=self.probe_timesteps,
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

    def save_best_config(self, study: optuna.Study, config_path: str | None = None) -> None:
        """
        Guarda best_params dins el YAML base (in-place).

        En lloc de crear un fitxer *_optimized.yaml separat, escriu la secció
        ``best_params`` directament al YAML base.  Al moment d'entrenar,
        ``apply_best_params`` aplica aquests valors automàticament.

        Args:
            config_path: ruta del YAML on escriure.  Per defecte: el YAML base
                         passat al constructor (self.config_path).
        """
        target = config_path or self.config_path
        # Llegir el YAML base actual (fresh read — no usem self.unified_cfg per
        # evitar sobreescriure comentaris o seccions afegides manualment)
        with open(target) as f:
            config = yaml.safe_load(f)

        config["best_params"] = study.best_params

        with open(target, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(
            f"  best_params saved to {target}: {study.best_params}"
        )
