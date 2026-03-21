# tests/integration/test_full_backtests.py
"""
Integration tests for full backtest runs.
Requires a live PostgreSQL database with OHLCV data for BTC/USDT 1h.

These tests skip automatically when the database is unavailable or empty,
so they are safe to run via `make test` both locally and on the Oracle VM.

Run only integration tests with:
    pytest tests/integration/ -v

Run everything (unit + smoke + integration):
    make test
"""
import math
import pytest

from core.config import TEST_FROM

# ─────────────────────────────────────────────────────────────────────────────
# Skip fixture: detect DB availability + data presence once per session
# ─────────────────────────────────────────────────────────────────────────────

MIN_CANDLES = 200   # minimum rows required to consider the DB "ready"
SYMBOL      = "BTC/USDT"
TIMEFRAME   = "1h"


def _db_has_data() -> tuple[bool, str]:
    """
    Returns (True, "") if PostgreSQL is reachable and has enough candles.
    Returns (False, reason) otherwise.
    """
    try:
        from core.db.session import SessionLocal
        from core.db.models import CandleDB
        session = SessionLocal()
        try:
            count = session.query(CandleDB).filter_by(
                symbol=SYMBOL, timeframe=TIMEFRAME
            ).count()
        finally:
            session.close()
    except Exception as exc:
        return False, f"DB connection failed: {exc}"

    if count < MIN_CANDLES:
        return False, (
            f"Not enough candles: {count} < {MIN_CANDLES} "
            f"for {SYMBOL} {TIMEFRAME}. Run: python scripts/download_data.py"
        )
    return True, ""


# Evaluated once at collection time — cheap
_DB_READY, _DB_SKIP_REASON = _db_has_data()

requires_db = pytest.mark.skipif(
    not _DB_READY,
    reason=_DB_SKIP_REASON or "PostgreSQL + OHLCV data not available",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_backtest(config_path: str, start_date: str | None = None, end_date: str | None = None):
    """Instantiate bot + engine and run a backtest. Returns BacktestMetrics."""
    import importlib
    import yaml
    from core.backtesting.engine import BacktestEngine

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mod       = importlib.import_module(cfg["module"])
    bot_cls   = getattr(mod, cfg["class_name"])
    bot       = bot_cls(config_path=config_path)

    engine    = BacktestEngine(bot=bot)
    return engine.run(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=start_date,
        end_date=end_date,
    )


def _assert_metrics_valid(metrics, *, bot_name: str) -> None:
    """Common assertions that must hold for any completed backtest."""
    summary = metrics.summary()

    assert summary["total_ticks"] > 0, \
        f"[{bot_name}] backtest produced 0 ticks"
    assert summary["initial_capital"] > 0, \
        f"[{bot_name}] initial capital must be positive"
    assert summary["final_capital"] >= 0, \
        f"[{bot_name}] final capital went negative"
    assert math.isfinite(summary["total_return_pct"]), \
        f"[{bot_name}] total_return_pct is not finite: {summary['total_return_pct']}"
    assert math.isfinite(summary["sharpe_ratio"]), \
        f"[{bot_name}] sharpe_ratio is not finite: {summary['sharpe_ratio']}"
    assert summary["max_drawdown_pct"] <= 0, \
        f"[{bot_name}] max_drawdown_pct should be ≤ 0, got {summary['max_drawdown_pct']}"
    assert 0.0 <= summary["win_rate_pct"] <= 100.0, \
        f"[{bot_name}] win_rate_pct out of range: {summary['win_rate_pct']}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@requires_db
def test_holdbot_completes_full_backtest():
    """
    HoldBot should buy once on tick 1 and hold.
    Final capital must be non-negative and all metrics finite.
    """
    metrics = _run_backtest("config/models/hold.yaml", start_date=TEST_FROM)
    _assert_metrics_valid(metrics, bot_name="HoldBot")

    summary = metrics.summary()
    # HoldBot never sells — after buying once, BTC balance > 0 and USDT ≈ 0
    # The portfolio_value can rise or fall with BTC price — just check it's finite
    assert math.isfinite(summary["final_capital"]), "HoldBot final capital not finite"


@pytest.mark.integration
@requires_db
def test_dcabot_completes_full_backtest():
    """
    DCABot should execute multiple buy orders spread over time.
    Expects at least 2 ticks to give it enough time to buy.
    """
    metrics = _run_backtest("config/models/dca.yaml", start_date=TEST_FROM)
    _assert_metrics_valid(metrics, bot_name="DCABot")

    summary = metrics.summary()
    # DCA accumulates over time — must have run at least some ticks
    assert summary["total_ticks"] >= 2, \
        f"DCABot only ran {summary['total_ticks']} ticks — not enough data"


@pytest.mark.integration
@requires_db
def test_trendbot_completes_full_backtest():
    """
    TrendBot uses EMA crossover + RSI filter.
    It may or may not trade in every period, but must not crash.
    """
    metrics = _run_backtest("config/models/trend.yaml", start_date=TEST_FROM)
    _assert_metrics_valid(metrics, bot_name="TrendBot")


@pytest.mark.integration
@requires_db
def test_backtest_produces_valid_metrics():
    """
    BacktestMetrics must produce a complete summary with no NaN / Inf values
    for any of the three classical bots.
    """
    for name, path in [
        ("HoldBot",  "config/models/hold.yaml"),
        ("DCABot",   "config/models/dca.yaml"),
        ("TrendBot", "config/models/trend.yaml"),
    ]:
        metrics = _run_backtest(path, start_date=TEST_FROM)
        summary = metrics.summary()

        for key, value in summary.items():
            if isinstance(value, float):
                assert math.isfinite(value), \
                    f"[{name}] metric '{key}' is not finite: {value}"

        assert set(summary.keys()) >= {
            "total_return_pct", "sharpe_ratio", "max_drawdown_pct",
            "calmar_ratio", "win_rate_pct", "initial_capital",
            "final_capital", "total_ticks",
        }, f"[{name}] summary dict is missing expected keys"


@pytest.mark.integration
@requires_db
def test_backtest_respects_date_filtering():
    """
    Running with start_date + end_date should produce fewer ticks than a full run.
    Also verifies that a tight 30-day window still produces a non-empty history.
    """
    import pandas as pd
    from core.config import TEST_FROM

    # Full run from TEST_FROM
    metrics_full = _run_backtest("config/models/hold.yaml", start_date=TEST_FROM)
    ticks_full   = metrics_full.summary()["total_ticks"]

    # Narrow 30-day window
    end_date_30d = (
        pd.Timestamp(TEST_FROM) + pd.Timedelta(days=30)
    ).strftime("%Y-%m-%d")

    metrics_30d = _run_backtest(
        "config/models/hold.yaml",
        start_date=TEST_FROM,
        end_date=end_date_30d,
    )
    ticks_30d = metrics_30d.summary()["total_ticks"]

    assert ticks_30d > 0, \
        f"30-day window produced 0 ticks (TEST_FROM={TEST_FROM}, end={end_date_30d})"
    assert ticks_30d < ticks_full, (
        f"Date filtering had no effect: "
        f"30-day window={ticks_30d} ticks, full run={ticks_full} ticks"
    )
    # 30 days × 24h = 720 hourly ticks at most
    assert ticks_30d <= 720 + 1, \
        f"30-day window produced {ticks_30d} ticks — suspiciously many for 30 days of 1h data"
