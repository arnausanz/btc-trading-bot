# tests/conftest.py
"""
Pytest configuration and shared fixtures for the BTC trading bot test suite.

Test categories:
  smoke       - Fast sanity checks, no external dependencies (<1s each)
  unit        - Logic tests with mocked data, no real database
  integration - Full pipeline tests requiring PostgreSQL + OHLCV data
                Mark with @pytest.mark.integration and run with:
                  pytest tests/integration/ -m integration

To run only smoke + unit tests (recommended for CI/CD):
  pytest tests/smoke/ tests/unit/ -v

To run all tests including integration:
  pytest tests/ -v
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests that require a real database with OHLCV data"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests that take more than a few seconds"
    )
