# tests/smoke/test_configs.py
"""Verify all YAML config files are well-formed and contain required fields."""
import yaml
import os
import pytest

CONFIG_ROOT = "config"

# Map config path → required top-level keys
# Classical bot configs live at config/models/{name}.yaml (same dir as ML models)
BOT_CONFIGS = {
    "config/models/dca.yaml":     ["bot_id", "features", "timeframe", "lookback"],
    "config/models/trend.yaml":   ["bot_id", "features", "timeframe", "lookback"],
    "config/models/hold.yaml":    ["bot_id", "timeframe"],
    "config/models/grid.yaml":    ["bot_id", "features", "timeframe", "lookback"],
    "config/models/xgboost.yaml": ["model_type", "bot", "features"],
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


def test_all_model_configs_exist():
    models_dir = "config/models"
    assert os.path.isdir(models_dir), f"Missing config directory: {models_dir}"
    configs = [f for f in os.listdir(models_dir) if f.endswith(".yaml")]
    assert len(configs) > 0, "No model configs found in config/models/"


def test_settings_yaml_parses():
    path = "config/settings.yaml"
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data is not None
