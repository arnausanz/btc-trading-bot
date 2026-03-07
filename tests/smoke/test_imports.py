# tests/smoke/test_imports.py
"""Verify all key modules are importable — catches missing dependencies early."""


def test_import_core_models():
    from core.models import Candle, Signal, Action, Order, OrderSide, OrderStatus, Trade


def test_import_core_interfaces():
    from core.interfaces.base_bot import BaseBot, ObservationSchema
    from core.interfaces.base_exchange import BaseExchange
    from core.interfaces.base_ml_model import BaseMLModel
    from core.interfaces.base_rl_agent import BaseRLAgent


def test_import_classical_bots():
    from bots.classical.dca_bot import DCABot
    from bots.classical.trend_bot import TrendBot
    from bots.classical.hold_bot import HoldBot
    from bots.classical.grid_bot import GridBot


def test_import_ml_bots():
    """Import ML bot modules (may be skipped if optional dependencies not installed)."""
    try:
        import bots.ml.random_forest
    except ImportError:
        pass  # Optional dependency

    try:
        import bots.ml.xgboost_model
    except ImportError:
        pass  # Optional dependency

    try:
        import bots.ml.lightgbm_model
    except ImportError:
        pass  # Optional dependency

    try:
        import bots.ml.catboost_model
    except ImportError:
        pass  # Optional dependency


def test_import_backtesting():
    from core.backtesting.metrics import BacktestMetrics
    from core.backtesting.engine import BacktestEngine
    from core.backtesting.optimizer import BotOptimizer
    from core.backtesting.comparator import BotComparator


def test_import_paper_exchange():
    from exchanges.paper import PaperExchange


def test_import_data_pipeline():
    from data.processing.technical import TechnicalIndicators
    from data.observation.builder import ObservationBuilder


def test_import_config():
    from core.config import TRAIN_UNTIL, TEST_FROM, MLFLOW_TRACKING_URI
    assert isinstance(TRAIN_UNTIL, str)
    assert isinstance(TEST_FROM, str)
