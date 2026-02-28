# tests/unit/test_optimizer.py
from bots.classical.dca_bot import DCABot
from core.backtesting.optimizer import BotOptimizer


def test_optimizer_finds_best_params():
    optimizer = BotOptimizer(
        bot_class=DCABot,
        base_config_path="config/bots/dca.yaml",
        param_space={
            "buy_every_n_ticks": {"type": "int", "low": 12, "high": 48},
            "buy_size": {"type": "float", "low": 0.05, "high": 0.3},
        },
        n_trials=5,  # poc per al test, suficient per validar que funciona
    )
    study = optimizer.run()
    assert study.best_value > -999.0
    assert "buy_every_n_ticks" in study.best_params
    assert "buy_size" in study.best_params