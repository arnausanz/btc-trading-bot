# bots/ml/ml_bot.py
import yaml
import logging
import pandas as pd
from datetime import datetime, timezone
from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.models import Signal, Action
from bots.ml.random_forest import RandomForestModel
from bots.ml.xgboost_model import XGBoostModel
from bots.ml.lightgbm_model import LightGBMModel
from bots.ml.catboost_model import CatBoostModel

logger = logging.getLogger(__name__)

_MODEL_REGISTRY = {
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "catboost": CatBoostModel,
}


class MLBot(BaseBot):
    """
    Bot que usa un model ML entrenat per prendre decisions.
    Agnòstic al tipus de model: suporta RF, XGBoost, LightGBM, CatBoost
    (i futurs models DL com GRU, TFT) via _MODEL_REGISTRY.
    Afegir un model nou = afegir una entrada al registry.
    """

    def __init__(self, config_path: str = "config/bots/ml_bot.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        super().__init__(bot_id=config["bot_id"], config=config)
        self._model = self._load_model()
        self._in_position = False

    def _load_model(self):
        model_type = self.config["model_type"]
        model_path = self.config["model_path"]

        if model_type not in _MODEL_REGISTRY:
            raise ValueError(
                f"Model desconegut: '{model_type}'. "
                f"Disponibles: {list(_MODEL_REGISTRY.keys())}"
            )

        model = _MODEL_REGISTRY[model_type]()
        model.load(model_path)
        logger.info(f"Model carregat: {model_type} des de {model_path}")
        return model

    def observation_schema(self) -> ObservationSchema:
        return ObservationSchema(
            features=self._model.feature_names,
            timeframes=[self.config["timeframe"]],
            lookback=self.config["lookback"],
        )

    def on_observation(self, observation: dict) -> Signal:
        timeframe = self.config["timeframe"]
        features: pd.DataFrame = observation[timeframe]["features"].copy()
        X = features.iloc[[-1]][self._model.feature_names]

        prediction, confidence = self._model.predict(
            X,
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
                reason=f"ML prediu pujada. Confiança: {confidence:.2f}",
            )

        if prediction == 0 and confidence >= min_confidence and self._in_position:
            self._in_position = False
            return Signal(
                bot_id=self.bot_id,
                timestamp=datetime.now(timezone.utc),
                action=Action.SELL,
                size=1.0,
                confidence=confidence,
                reason=f"ML prediu baixada. Confiança: {confidence:.2f}",
            )

        return Signal(
            bot_id=self.bot_id,
            timestamp=datetime.now(timezone.utc),
            action=Action.HOLD,
            size=0.0,
            confidence=confidence,
            reason=f"Confiança insuficient o sense senyal: {confidence:.2f}",
        )

    def on_start(self) -> None:
        self._in_position = False
        logger.info(f"MLBot iniciat amb model: {self.config['model_type']}")