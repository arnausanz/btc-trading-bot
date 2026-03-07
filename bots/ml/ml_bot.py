# bots/ml/ml_bot.py
import yaml
import logging
import pandas as pd
from datetime import datetime, timezone
from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.interfaces.base_ml_model import BaseMLModel
from core.models import Signal, Action
from bots.ml.random_forest import RandomForestModel
from bots.ml.xgboost_model import XGBoostModel
from bots.ml.lightgbm_model import LightGBMModel
from bots.ml.catboost_model import CatBoostModel
from bots.ml.gru_model import GRUModel
from bots.ml.patchtst_model import PatchTSTModel

logger = logging.getLogger(__name__)

_MODEL_REGISTRY: dict[str, type[BaseMLModel]] = {
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "catboost": CatBoostModel,
    "gru": GRUModel,
    "patchtst": PatchTSTModel,
}


class MLBot(BaseBot):
    """
    ML/DL model-agnostic bot.
    Adding a new model = adding an entry to _MODEL_REGISTRY.
    """

    def __init__(self, config_path: str = "config/bots/ml_bot.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
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

        # Clean and generic — works for any current and future model
        X_input = features[self._model.feature_names].iloc[-self._model.lookback:]

        prediction, confidence = self._model.predict(
            X_input,
            threshold=self.config["prediction_threshold"],
        )

        min_confidence = self.config["min_confidence"]
        size = self.config["trade_size"]

        # BUY: model predicts upward and sufficient confidence and not in position
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

        # SELL: model stops predicting upward (prediction==0 means prob < threshold)
        # NOTE: we don't apply min_confidence to SELL — would be impossible (prob<threshold < min_confidence)
        # Bearish confidence is 1 - prob (lower prob = more certain sell signal)
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