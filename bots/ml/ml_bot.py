# bots/ml/ml_bot.py
"""
ML/DL model-agnostic bot.

The ``_MODEL_REGISTRY`` is built automatically at import time by scanning
``config/models/*.yaml`` (category: ML).  Each YAML must declare:

    module:     bots.ml.<module_name>
    class_name: <ClassName>

Adding a new model = create the YAML + the Python class.  No edits here.
"""
import importlib
import logging

import yaml

from core.config_utils import discover_configs
from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.interfaces.base_ml_model import BaseMLModel
from core.models import Signal, Action
from datetime import datetime, timezone
import pandas as pd

logger = logging.getLogger(__name__)


def _build_ml_registry() -> dict[str, type[BaseMLModel]]:
    """
    Auto-populates the model registry from config/models/*.yaml (category: ML).

    Each YAML must have ``module`` and ``class_name`` fields.
    If a YAML is missing these fields it is silently skipped (can still be loaded
    manually via MLBot with a direct config path).
    """
    registry: dict[str, type[BaseMLModel]] = {}
    for stem, path in discover_configs("ML").items():
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            mt       = cfg.get("model_type")
            mod_path = cfg.get("module")
            cls_name = cfg.get("class_name")
            if mt and mod_path and cls_name:
                mod = importlib.import_module(mod_path)
                registry[mt] = getattr(mod, cls_name)
        except Exception as exc:
            logger.warning("Could not register ML model from %s: %s", path, exc)
    return registry


_MODEL_REGISTRY: dict[str, type[BaseMLModel]] = _build_ml_registry()


class MLBot(BaseBot):
    """
    ML/DL model-agnostic bot.
    Llegeix configs unificades de config/models/*.yaml.
    Afegir un nou model = una línia a _MODEL_REGISTRY.

    Config esperada (config/models/{model}.yaml):
      category:    ML
      model_type:  xgboost
      module:      bots.ml.xgboost_model
      class_name:  XGBoostModel
      timeframes:  [1h]
      training:
        model_path: models/xgboost_v1.pkl
      bot:
        bot_id: ml_xgb_v1
        lookback: 200
        min_confidence: 0.4
        trade_size: 0.5
        prediction_threshold: 0.35
    """

    def __init__(self, config_path: str = "config/models/xgboost.yaml"):
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        # Construïm config plana combinant root + bot section + claus derivades.
        config = {
            **raw,
            **raw.get("bot", {}),
            "timeframe": raw.get("timeframe") or raw["timeframes"][0],
            "model_path": raw["training"]["model_path"],
        }
        super().__init__(bot_id=config["bot_id"], config=config)
        self._model = self._load_model()
        self._in_position = False

    def _load_model(self) -> BaseMLModel:
        model_type = self.config["model_type"]
        model_path = self.config["model_path"]

        if model_type not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: '{model_type}'. "
                f"Available: {list(_MODEL_REGISTRY.keys())}"
            )

        model = _MODEL_REGISTRY[model_type]()
        model.load(model_path)
        logger.info(f"Model loaded: {model_type} from {model_path}")
        return model

    def observation_schema(self) -> ObservationSchema:
        return ObservationSchema(
            features=self._model.feature_names,
            timeframes=[self.config["timeframe"]],
            lookback=max(self.config["lookback"], self._model.lookback),
        )

    def on_observation(self, observation: dict) -> Signal:
        timeframe = self.config["timeframe"]
        features: pd.DataFrame = observation[timeframe]["features"].copy()

        X_input = features[self._model.feature_names].iloc[-self._model.lookback:]

        prediction, confidence = self._model.predict(
            X_input,
            threshold=self.config["prediction_threshold"],
        )

        min_confidence = self.config["min_confidence"]
        size = self.config["trade_size"]

        if prediction == 1 and confidence >= min_confidence and not self._in_position:
            self._in_position = True
            return Signal(
                bot_id=self.bot_id,
                timestamp=datetime.now(timezone.utc),
                action=Action.BUY,
                size=size,
                confidence=confidence,
                reason=f"ML predicts upward. Confidence: {confidence:.2f}",
            )

        if prediction == 0 and self._in_position:
            bearish_conf = 1.0 - confidence
            self._in_position = False
            return Signal(
                bot_id=self.bot_id,
                timestamp=datetime.now(timezone.utc),
                action=Action.SELL,
                size=1.0,
                confidence=bearish_conf,
                reason=f"ML does not predict upward. Bearish confidence: {bearish_conf:.2f}",
            )

        return Signal(
            bot_id=self.bot_id,
            timestamp=datetime.now(timezone.utc),
            action=Action.HOLD,
            size=0.0,
            confidence=confidence,
            reason=f"ML: {'in position, waiting for sell signal' if self._in_position else f'insufficient confidence ({confidence:.2f})'}",
        )

    def on_start(self) -> None:
        self._in_position = False
        logger.info(f"MLBot initialized with model: {self.config['model_type']}")
