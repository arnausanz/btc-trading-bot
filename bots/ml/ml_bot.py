# bots/ml/ml_bot.py
import yaml
import logging
import pandas as pd
from datetime import datetime, timezone
from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.models import Signal, Action
from bots.ml.random_forest import RandomForestModel
from bots.ml.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)


class MLBot(BaseBot):
    """
    Bot que usa un model ML entrenat per prendre decisions.
    Agnòstic al tipus de model — funciona amb Random Forest, XGBoost,
    o qualsevol model futur que implementi predict().
    Només actua quan la confiança del model supera min_confidence.
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

        if model_type == "random_forest":
            model = RandomForestModel()
        elif model_type == "xgboost":
            model = XGBoostModel()
        else:
            raise ValueError(f"Model desconegut: {model_type}")

        model.load(model_path)
        return model

    def observation_schema(self) -> ObservationSchema:
        return ObservationSchema(
            features=self._model.feature_names,  # usa les features del model
            timeframes=[self.config["timeframe"]],
            lookback=self.config["lookback"],
        )

    def on_observation(self, observation: dict) -> Signal:
        timeframe = self.config["timeframe"]
        features: pd.DataFrame = observation[timeframe]["features"].copy()

        # Agafa l'última fila com a input del model
        X = features.iloc[[-1]][self._model.feature_names]

        prediction, confidence = self._model.predict(
            X,
            threshold=self.config["prediction_threshold"]
        )

        min_confidence = self.config["min_confidence"]
        size = self.config["trade_size"]

        # Compra si el model prediu pujada amb suficient confiança
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

        # Ven si el model prediu baixada i estem en posició
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
            reason=f"Confiança insuficient o sense senyal. Confiança: {confidence:.2f}",
        )

    def on_start(self) -> None:
        self._in_position = False
        logger.info(f"MLBot iniciat amb model: {self.config['model_type']}")