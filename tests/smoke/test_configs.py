# tests/smoke/test_configs.py
"""Verify all YAML config files are well-formed and contain required fields."""
import yaml
import os
import pytest

CONFIG_ROOT = "config"

# Map config path → required top-level keys
BOT_CONFIGS = {
    "config/bots/dca.yaml":    ["bot_id", "features", "timeframe", "lookback"],
    "config/bots/trend.yaml":  ["bot_id", "features", "timeframe", "lookback"],
    "config/bots/hold.yaml":   ["bot_id", "features", "timeframe", "lookback"],
    "config/bots/grid.yaml":   ["bot_id", "features", "timeframe", "lookback"],
    "config/bots/ml_bot.yaml": ["bot_id", "model_type", "model_path", "timeframe"],
    "config/exchanges/paper.yaml": ["initial_capital"],
    "config/demo.yaml": ["demo", "bots", "exchange"],
}


@pytest.mark.parametrize("path,required_keys", BOT_CONFIGS.items())
def test_config_parses_and_has_required_keys(path, required_keys):
    assert os.path.exists(path), f"Config file missing: {path}"
    with open(path) as f:
        data = yaml.safe_load(f)
    assert data is not None, f"Config file is empty: {path}"
    for key in required_keys:
        assert key in data, f"Missing required key '{key}' in {path}"


def test_all_training_configs_exist():
    training_dir = "config/training"
    assert os.path.isdir(training_dir)
    configs = os.listdir(training_dir)
    assert len(configs) > 0, "No training configs found"


def test_settings_yaml_parses():
    path = "config/settings.yaml"
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data is not None
