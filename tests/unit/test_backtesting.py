# tests/unit/test_backtesting.py
# Tests unitaris de BacktestMetrics. No necessiten DB ni connexió externa.
# Executa'ls amb: poetry run pytest tests/unit/test_backtesting.py -v
import numpy as np
import pandas as pd
import pytest

from core.backtesting.metrics import BacktestMetrics, _PERIODS_PER_YEAR


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_history(n: int = 100, initial: float = 10_000.0, seed: int = 42) -> list[dict]:
    """Historial sintètic bàsic (sense timestamps ni senyals). Seed fix = determinista."""
    rng = np.random.default_rng(seed)
    values = initial + np.cumsum(rng.normal(0, 10, n))
    return [{"portfolio_value": float(v), "order_status": "hold"} for v in values]


def make_history_with_timestamps(
    n: int = 100,
    initial: float = 10_000.0,
    timeframe: str = "1h",
    seed: int = 42,
) -> list[dict]:
    """Historial amb timestamps reals per als tests que verifiquen _duration_days."""
    rng = np.random.default_rng(seed)
    values = initial + np.cumsum(rng.normal(0, 10, n))
    freq_map = {"1h": "h", "4h": "4h", "1d": "D", "1w": "W"}
    freq = freq_map.get(timeframe, "h")
    timestamps = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    return [
        {
            "portfolio_value": float(v),
            "order_status": "hold",
            "signal": "hold",
            "timestamp": ts,
        }
        for v, ts in zip(values, timestamps)
    ]


# ── Tests bàsics (total_return, max_drawdown) ─────────────────────────────────

def test_total_return_positive():
    history = [
        {"portfolio_value": 10_000.0, "order_status": "hold"},
        {"portfolio_value": 11_000.0, "order_status": "hold"},
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.total_return() == pytest.approx(10.0)


def test_total_return_negative():
    history = [
        {"portfolio_value": 10_000.0, "order_status": "hold"},
        {"portfolio_value": 9_000.0, "order_status": "hold"},
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.total_return() == pytest.approx(-10.0)


def test_total_return_zero():
    history = [{"portfolio_value": 10_000.0, "order_status": "hold"}] * 10
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.total_return() == pytest.approx(0.0)


def test_max_drawdown_is_negative():
    history = [
        {"portfolio_value": 10_000.0, "order_status": "hold"},
        {"portfolio_value": 12_000.0, "order_status": "hold"},
        {"portfolio_value": 9_000.0, "order_status": "hold"},
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.max_drawdown() < 0


def test_max_drawdown_exact():
    """Pujada fins a 12.000, caiguda fins a 9.000 → drawdown = -25%."""
    history = [
        {"portfolio_value": 10_000.0, "order_status": "hold"},
        {"portfolio_value": 12_000.0, "order_status": "hold"},
        {"portfolio_value": 9_000.0, "order_status": "hold"},
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.max_drawdown() == pytest.approx(-25.0)


def test_max_drawdown_no_drawdown():
    """Portfolio sempre creixent → drawdown = 0%."""
    values = [10_000.0 * (1.01 ** i) for i in range(50)]
    history = [{"portfolio_value": v, "order_status": "hold"} for v in values]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.max_drawdown() == pytest.approx(0.0)


# ── Tests Sharpe Ratio ────────────────────────────────────────────────────────

def test_sharpe_ratio_flat_portfolio():
    """Portfolio sense canvis → desviació = 0 → Sharpe = 0."""
    history = [{"portfolio_value": 10_000.0, "order_status": "hold"}] * 50
    m = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    assert m.sharpe_ratio() == 0.0


def test_sharpe_positive_for_growing_portfolio():
    """Portfolio sempre creixent → Sharpe positiu."""
    values = [10_000.0 * (1.001 ** i) for i in range(200)]
    history = [{"portfolio_value": v, "order_status": "hold"} for v in values]
    m = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    assert m.sharpe_ratio() > 0


def test_sharpe_negative_for_declining_portfolio():
    """Portfolio sempre baixant → Sharpe negatiu."""
    values = [10_000.0 * (0.999 ** i) for i in range(200)]
    history = [{"portfolio_value": v, "order_status": "hold"} for v in values]
    m = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    assert m.sharpe_ratio() < 0


def test_sharpe_annualization_1h_vs_1d():
    """
    Sobre la mateixa seqüència de retorns, Sharpe(1h) / Sharpe(1d) = sqrt(24).
    Això verifica que l'anualització usa el factor correcte per a cada timeframe.
    """
    history = make_history(n=500, seed=42)
    m_1h = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    m_1d = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1d")

    s_1h = m_1h.sharpe_ratio()
    s_1d = m_1d.sharpe_ratio()

    assert s_1d != 0, "Sharpe 1d hauria de ser != 0 amb dades aleatòries"
    ratio = s_1h / s_1d
    assert ratio == pytest.approx(np.sqrt(24), rel=0.01)


def test_sharpe_annualization_1h_vs_4h():
    """Sharpe(1h) / Sharpe(4h) = sqrt(4) = 2."""
    history = make_history(n=500, seed=42)
    m_1h = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    m_4h = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="4h")

    s_1h = m_1h.sharpe_ratio()
    s_4h = m_4h.sharpe_ratio()

    assert s_4h != 0
    ratio = s_1h / s_4h
    assert ratio == pytest.approx(np.sqrt(4), rel=0.01)


def test_sharpe_periods_per_year_values():
    """El diccionari _PERIODS_PER_YEAR ha de tenir els valors correctes."""
    assert _PERIODS_PER_YEAR["1h"] == 365 * 24        # 8.760
    assert _PERIODS_PER_YEAR["4h"] == 365 * 6         # 2.190
    assert _PERIODS_PER_YEAR["1d"] == 365
    assert _PERIODS_PER_YEAR["1w"] == 52
    assert _PERIODS_PER_YEAR["1m"] == 365 * 24 * 60


def test_sharpe_unknown_timeframe_defaults_to_1h():
    """Un timeframe desconegut ha de fer servir el factor d'1h per defecte."""
    history = make_history(n=100, seed=42)
    m_unknown = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="7h")
    m_1h = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    assert m_unknown.sharpe_ratio() == pytest.approx(m_1h.sharpe_ratio())


# ── Tests Calmar Ratio ────────────────────────────────────────────────────────

def test_calmar_zero_drawdown():
    """Sense drawdown → Calmar = 0 (per convenció: indefinit)."""
    values = [10_000.0 * (1.001 ** i) for i in range(50)]
    history = [{"portfolio_value": v, "order_status": "hold"} for v in values]
    m = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    assert m.calmar_ratio() == 0.0


def test_duration_days_with_timestamps_1h():
    """360 ticks d'1h = 15 dies exactes."""
    history = make_history_with_timestamps(n=361, timeframe="1h")
    m = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    assert m._duration_days() == pytest.approx(360 / 24, abs=0.1)


def test_duration_days_with_timestamps_1d():
    """365 ticks d'1d = ~364 dies."""
    history = make_history_with_timestamps(n=365, timeframe="1d")
    m = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1d")
    assert m._duration_days() == pytest.approx(364, abs=1)


def test_duration_days_fallback_without_timestamps():
    """Sense timestamps, el fallback ha d'estimar dies consistentment."""
    history = make_history(n=8760)  # 1 any de ticks horaris
    m = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    # 8760 ticks / 24 períodes_per_dia = 365 dies
    assert m._duration_days() == pytest.approx(365.0, rel=0.01)


def test_duration_days_1h_vs_1d_same_n_ticks():
    """
    Amb el mateix nombre de ticks (sense timestamps), 1d dóna 24x més dies que 1h.
    100 ticks 1h → 100/24 ≈ 4.17 dies
    100 ticks 1d → 100/1  = 100 dies
    """
    history = make_history(n=100)
    m_1h = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1h")
    m_1d = BacktestMetrics(history=history, initial_capital=10_000.0, timeframe="1d")
    assert m_1d._duration_days() > m_1h._duration_days()
    assert m_1d._duration_days() / m_1h._duration_days() == pytest.approx(24, rel=0.01)


# ── Tests Win Rate ────────────────────────────────────────────────────────────

def test_win_rate_no_signal_column():
    """Historial sense columna 'signal' → win rate = 0."""
    m = BacktestMetrics(history=make_history(), initial_capital=10_000.0)
    assert m.win_rate() == 0.0


def test_win_rate_no_trades_only_hold():
    """Tots els senyals son HOLD → 0 round-trips → win rate = 0."""
    history = [
        {"portfolio_value": 10_000.0, "signal": "hold", "order_status": "hold"},
        {"portfolio_value": 10_500.0, "signal": "hold", "order_status": "hold"},
        {"portfolio_value": 11_000.0, "signal": "hold", "order_status": "hold"},
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.win_rate() == 0.0


def test_win_rate_open_position_not_counted():
    """BUY sense SELL posterior → posició oberta, no compta com a trade."""
    history = [
        {"portfolio_value": 10_000.0, "signal": "buy",  "order_status": "filled"},
        {"portfolio_value": 11_000.0, "signal": "hold", "order_status": "filled"},
        # Sense SELL → no hi ha round-trip tancat
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.win_rate() == 0.0


def test_win_rate_all_winning():
    """2 round-trips guanyadors → win rate = 100%."""
    history = [
        {"portfolio_value": 10_000.0, "signal": "buy",  "order_status": "filled"},
        {"portfolio_value": 10_500.0, "signal": "hold", "order_status": "hold"},
        {"portfolio_value": 11_000.0, "signal": "sell", "order_status": "filled"},  # +1.000 ✓
        {"portfolio_value": 11_000.0, "signal": "buy",  "order_status": "filled"},
        {"portfolio_value": 12_000.0, "signal": "sell", "order_status": "filled"},  # +1.000 ✓
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.win_rate() == pytest.approx(100.0)


def test_win_rate_all_losing():
    """2 round-trips perdedors → win rate = 0%."""
    history = [
        {"portfolio_value": 11_000.0, "signal": "buy",  "order_status": "filled"},
        {"portfolio_value": 10_000.0, "signal": "sell", "order_status": "filled"},  # -1.000 ✗
        {"portfolio_value": 10_000.0, "signal": "buy",  "order_status": "filled"},
        {"portfolio_value": 9_000.0,  "signal": "sell", "order_status": "filled"},  # -1.000 ✗
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.win_rate() == pytest.approx(0.0)


def test_win_rate_two_wins_one_loss():
    """2 guanyadors, 1 perdedor → win rate = 66.67%."""
    history = [
        {"portfolio_value": 10_000.0, "signal": "buy",  "order_status": "filled"},
        {"portfolio_value": 11_000.0, "signal": "sell", "order_status": "filled"},  # +1.000 ✓
        {"portfolio_value": 11_000.0, "signal": "buy",  "order_status": "filled"},
        {"portfolio_value": 10_000.0, "signal": "sell", "order_status": "filled"},  # -1.000 ✗
        {"portfolio_value": 10_000.0, "signal": "buy",  "order_status": "filled"},
        {"portfolio_value": 10_500.0, "signal": "sell", "order_status": "filled"},  # +500  ✓
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.win_rate() == pytest.approx(100 * 2 / 3, rel=0.01)


def test_win_rate_breakeven_not_winning():
    """Portfolio igual a l'entrada → NO es compta com a guanyador."""
    history = [
        {"portfolio_value": 10_000.0, "signal": "buy",  "order_status": "filled"},
        {"portfolio_value": 10_000.0, "signal": "sell", "order_status": "filled"},  # =0 → ✗
    ]
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.win_rate() == pytest.approx(0.0)


# ── Tests summary ─────────────────────────────────────────────────────────────

def test_summary_contains_all_keys():
    m = BacktestMetrics(history=make_history(), initial_capital=10_000.0)
    summary = m.summary()
    expected_keys = [
        "total_return_pct", "sharpe_ratio", "max_drawdown_pct",
        "calmar_ratio", "win_rate_pct", "initial_capital",
        "final_capital", "total_ticks",
    ]
    for key in expected_keys:
        assert key in summary, f"Clau absent al summary: {key}"


def test_summary_total_ticks():
    history = make_history(n=150)
    m = BacktestMetrics(history=history, initial_capital=10_000.0)
    assert m.summary()["total_ticks"] == 150


def test_summary_initial_capital_preserved():
    history = make_history(n=50)
    m = BacktestMetrics(history=history, initial_capital=12_345.0)
    assert m.summary()["initial_capital"] == 12_345.0
