# tests/unit/test_interfaces.py
"""
Unit tests for abstract interface contracts.
Verify that all bot implementations satisfy the BaseBot interface.
"""
import pytest
from abc import ABC, abstractmethod
from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.models import Signal, Action
from datetime import datetime, timezone


class MinimalBot(BaseBot):
    """Minimal implementation of BaseBot for testing."""

    def observation_schema(self) -> ObservationSchema:
        """Declare data dependencies."""
        return ObservationSchema(
            features=["close"],
            timeframes=["1h"],
            lookback=10,
        )

    def on_observation(self, observation: dict) -> Signal:
        """Generate a signal."""
        return Signal(
            bot_id=self.bot_id,
            timestamp=datetime.now(timezone.utc),
            action=Action.HOLD,
            size=0.0,
            confidence=1.0,
            reason="Minimal bot test",
        )


class TestBaseBot:
    """Test that BaseBot interface is properly defined."""

    def test_basebot_is_abstract(self):
        """BaseBot should not be directly instantiable."""
        with pytest.raises(TypeError):
            BaseBot(bot_id="test", config={})

    def test_basebot_subclass_requires_observation_schema(self):
        """Subclass must implement observation_schema()."""
        class IncompleteBot(BaseBot):
            def on_observation(self, observation: dict) -> Signal:
                return Signal(
                    bot_id="test",
                    timestamp=datetime.now(timezone.utc),
                    action=Action.HOLD,
                    size=0.0,
                    confidence=1.0,
                    reason="test",
                )

        with pytest.raises(TypeError):
            IncompleteBot(bot_id="test", config={})

    def test_basebot_subclass_requires_on_observation(self):
        """Subclass must implement on_observation()."""
        class IncompleteBot(BaseBot):
            def observation_schema(self) -> ObservationSchema:
                return ObservationSchema(
                    features=["close"],
                    timeframes=["1h"],
                    lookback=10,
                )

        with pytest.raises(TypeError):
            IncompleteBot(bot_id="test", config={})


class TestBotInterface:
    """Test the bot interface with a minimal implementation."""

    def test_bot_stores_bot_id(self):
        """Bot should store the bot_id passed to constructor."""
        bot = MinimalBot(bot_id="test_bot", config={})
        assert bot.bot_id == "test_bot"

    def test_bot_stores_config(self):
        """Bot should store the config dict."""
        config = {"param1": "value1"}
        bot = MinimalBot(bot_id="test_bot", config=config)
        assert bot.config == config

    def test_observation_schema_returns_valid_schema(self):
        """observation_schema() should return an ObservationSchema."""
        bot = MinimalBot(bot_id="test_bot", config={})
        schema = bot.observation_schema()
        assert isinstance(schema, ObservationSchema)
        assert schema.features is not None
        assert schema.timeframes is not None
        assert schema.lookback is not None

    def test_observation_schema_declares_features(self):
        """ObservationSchema should declare required features."""
        bot = MinimalBot(bot_id="test_bot", config={})
        schema = bot.observation_schema()
        assert "close" in schema.features

    def test_observation_schema_declares_timeframes(self):
        """ObservationSchema should declare required timeframes."""
        bot = MinimalBot(bot_id="test_bot", config={})
        schema = bot.observation_schema()
        assert "1h" in schema.timeframes

    def test_observation_schema_declares_lookback(self):
        """ObservationSchema should declare lookback period."""
        bot = MinimalBot(bot_id="test_bot", config={})
        schema = bot.observation_schema()
        assert schema.lookback > 0

    def test_on_observation_returns_signal(self):
        """on_observation() should return a Signal."""
        bot = MinimalBot(bot_id="test_bot", config={})
        observation = {"1h": {"features": None}}
        signal = bot.on_observation(observation)
        assert isinstance(signal, Signal)

    def test_signal_has_valid_action(self):
        """Signal returned by on_observation() should have valid action."""
        bot = MinimalBot(bot_id="test_bot", config={})
        observation = {"1h": {"features": None}}
        signal = bot.on_observation(observation)
        assert signal.action in [Action.BUY, Action.SELL, Action.HOLD]

    def test_signal_has_valid_confidence(self):
        """Signal confidence should be between 0 and 1."""
        bot = MinimalBot(bot_id="test_bot", config={})
        observation = {"1h": {"features": None}}
        signal = bot.on_observation(observation)
        assert 0.0 <= signal.confidence <= 1.0

    def test_optional_on_start_hook(self):
        """on_start() is optional and should be callable."""
        bot = MinimalBot(bot_id="test_bot", config={})
        # Should not raise
        bot.on_start()

    def test_optional_on_stop_hook(self):
        """on_stop() is optional and should be callable."""
        bot = MinimalBot(bot_id="test_bot", config={})
        # Should not raise
        bot.on_stop()


class TestObservationSchema:
    """Test the ObservationSchema dataclass."""

    def test_schema_has_features(self):
        schema = ObservationSchema(
            features=["close", "rsi"],
            timeframes=["1h"],
            lookback=20,
        )
        assert schema.features == ["close", "rsi"]

    def test_schema_has_timeframes(self):
        schema = ObservationSchema(
            features=["close"],
            timeframes=["1h", "4h"],
            lookback=20,
        )
        assert schema.timeframes == ["1h", "4h"]

    def test_schema_has_lookback(self):
        schema = ObservationSchema(
            features=["close"],
            timeframes=["1h"],
            lookback=50,
        )
        assert schema.lookback == 50

    def test_schema_has_optional_extras(self):
        schema = ObservationSchema(
            features=["close"],
            timeframes=["1h"],
            lookback=20,
            extras={"sentiment": "external_api"},
        )
        assert schema.extras == {"sentiment": "external_api"}

    def test_schema_extras_default_to_empty(self):
        schema = ObservationSchema(
            features=["close"],
            timeframes=["1h"],
            lookback=20,
        )
        assert schema.extras == {}
