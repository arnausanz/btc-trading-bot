# core/config.py
import os
import yaml

# ─── Walk-forward split dates ──────────────────────────────────────────────────
# Read from config/settings.yaml; if file doesn't exist, use defaults.
_SETTINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config", "settings.yaml"
)


def _load_settings() -> dict:
    try:
        with open(_SETTINGS_PATH) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


_settings = _load_settings()
_bt = _settings.get("backtesting", {})

TRAIN_UNTIL: str = _bt.get("train_until", "2024-12-31")
TEST_FROM: str = _bt.get("test_from", "2025-01-01")
