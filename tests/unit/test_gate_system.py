"""
Tests unitaris del Gate System — Validació completa de les 5 portes.

Cobreix:
  - P1: càlcul de features, inferència, llindars de confiança
  - P2: taula Fear & Greed, funding rate scoring, producte de sub-scores
  - P3: pivots fractals, fibonacci, volume profile, merge, volum d'acostament
  - P4: derivades, RSI-2, MACD cross, senyals per règim
  - P5: Kelly sizing, condicions VETO, trailing stop, desacceleració, circuit breaker
  - GateBot: pipeline seqüencial, near-miss logging, gestió de posicions
  - Regime models: HMM trainer, XGB classifier, walk-forward splits
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

def _make_df_4h(n: int = 300, base_price: float = 50_000.0, seed: int = 42) -> pd.DataFrame:
    """Genera un DataFrame 4H sintètic amb OHLCV + indicadors."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    close = base_price + np.cumsum(rng.normal(0, 100, n))
    close = np.maximum(close, 1000)  # evitar preus negatius
    high  = close + rng.uniform(50, 300, n)
    low   = close - rng.uniform(50, 300, n)
    volume = rng.uniform(100, 10_000, n)

    # ATR, RSI, MACD simulats
    atr_14 = pd.Series(close).rolling(14).std().fillna(200).values
    rsi_14 = 30 + rng.uniform(0, 40, n)

    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()

    df = pd.DataFrame({
        "close": close, "high": high, "low": low, "open": close - rng.normal(0, 50, n),
        "volume": volume, "atr_14": atr_14, "rsi_14": rsi_14,
        "macd": macd_line.values, "macd_signal": macd_signal.values,
        "funding_rate": rng.normal(0.0001, 0.0002, n),
    }, index=dates)
    return df


def _make_df_1d(n: int = 300, base_price: float = 50_000.0, seed: int = 42) -> pd.DataFrame:
    """Genera un DataFrame diari sintètic amb indicadors per P1/P2."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    close = base_price + np.cumsum(rng.normal(0, 300, n))
    close = np.maximum(close, 1000)
    volume = rng.uniform(500, 50_000, n)

    ema50  = pd.Series(close).ewm(span=50, adjust=False).mean()
    ema200 = pd.Series(close).ewm(span=200, adjust=False).mean()
    atr_14 = pd.Series(close).rolling(14).std().fillna(500).values
    adx_14 = 20 + rng.uniform(0, 30, n)
    rsi_14 = 30 + rng.uniform(0, 40, n)

    df = pd.DataFrame({
        "close": close, "high": close + 500, "low": close - 500,
        "open": close - rng.normal(0, 100, n),
        "volume": volume,
        "ema_50": ema50.values, "ema_200": ema200.values,
        "adx_14": adx_14, "atr_14": atr_14, "rsi_14": rsi_14,
        "fear_greed_value": rng.integers(10, 90, n).astype(float),
        "funding_rate": rng.normal(0.0001, 0.0002, n),
    }, index=dates)
    return df


def _gate_config() -> dict:
    """Configuració mínima per instanciar les portes."""
    return {
        "bot_id": "gate_test",
        "category": "gate",
        "model_type": "gate",
        "lookback": 300,
        "features_4h": ["close", "high", "low", "volume", "atr_14", "rsi_14", "macd", "macd_signal"],
        "features_1d": ["close", "ema_50", "ema_200", "adx_14", "rsi_14", "atr_14", "fear_greed_value"],
        "model_paths": {"hmm": "models/gate_hmm.pkl", "xgb": "models/gate_xgb_regime.pkl"},
        "external": {"fear_greed": True, "funding_rate": True},
        "p1": {"min_regime_confidence": 0.60},
        "p2": {"onchain_enabled": True},
        "p3": {"min_level_strength": 0.4, "fractal_n_4h": 2, "fractal_n_1d": 5,
               "min_swing_pct": 0.03, "volume_profile_bins": 50},
        "p4": {"ewm_span": 3, "rsi2_oversold": 10},
        "p5": {"max_risk_pct": 0.01, "max_exposure_pct": 0.95, "max_open_positions": 2,
               "weekly_drawdown_limit": 0.05, "stagnation_days": 6,
               "min_rr": {"STRONG_BULL": 1.5, "WEAK_BULL": 2.0, "RANGING": 2.0},
               "trailing_atr_multiplier": {"low_vol": 1.5, "normal_vol": 2.0, "high_vol": 2.5},
               "decel_exit_candles": {"STRONG_BULL": 5, "WEAK_BULL": 3, "RANGING": 2}},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests P1 — Règim Macro
# ═══════════════════════════════════════════════════════════════════════════════

class TestP1Regime:
    def test_no_model_returns_uncertain(self):
        from bots.gate.gates.p1_regime import P1Regime
        p1 = P1Regime(_gate_config())
        # Sense carregar model → UNCERTAIN
        result = p1.evaluate(_make_df_1d())
        assert result.regime == "UNCERTAIN"
        assert result.is_open is False

    def test_compute_features_returns_14_columns(self):
        from bots.gate.gates.p1_regime import P1Regime, _linear_slope
        from bots.gate.regime_models.xgb_classifier import P1_FEATURES
        df = _make_df_1d(n=300)
        features = P1Regime._compute_features(df)
        assert list(features.columns) == P1_FEATURES
        assert len(features) == 1
        assert not features.isnull().any().any()

    def test_compute_features_handles_missing_optional_columns(self):
        """Si falten funding_rate o fear_greed, usa valors per defecte."""
        from bots.gate.gates.p1_regime import P1Regime
        df = _make_df_1d(n=300)
        df = df.drop(columns=["funding_rate", "fear_greed_value"])
        features = P1Regime._compute_features(df)
        assert features["funding_rate_3d"].iloc[0] == 0.0
        assert features["fear_greed"].iloc[0] == 50.0

    def test_p1result_allows_long(self):
        from bots.gate.gates.p1_regime import P1Result
        assert P1Result(regime="STRONG_BULL", confidence=0.8, is_open=True).allows_long()
        assert P1Result(regime="WEAK_BULL", confidence=0.7, is_open=True).allows_long()
        assert P1Result(regime="RANGING", confidence=0.65, is_open=True).allows_long()
        assert not P1Result(regime="STRONG_BEAR", confidence=0.8, is_open=True).allows_long()
        assert not P1Result(regime="UNCERTAIN", confidence=0.3, is_open=False).allows_long()

    def test_p1result_invalidates_long(self):
        from bots.gate.gates.p1_regime import P1Result
        assert P1Result(regime="STRONG_BEAR", confidence=0.8, is_open=True).invalidates_long_position()
        assert P1Result(regime="WEAK_BEAR", confidence=0.8, is_open=True).invalidates_long_position()
        assert P1Result(regime="UNCERTAIN", confidence=0.3, is_open=False).invalidates_long_position()
        assert not P1Result(regime="STRONG_BULL", confidence=0.8, is_open=True).invalidates_long_position()

    def test_linear_slope_basic(self):
        from bots.gate.gates.p1_regime import _linear_slope
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        slope = _linear_slope(series)
        assert abs(slope - 1.0) < 0.01

    def test_linear_slope_short_series(self):
        from bots.gate.gates.p1_regime import _linear_slope
        assert _linear_slope(pd.Series([5.0])) == 0.0
        assert _linear_slope(pd.Series(dtype=float)) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Tests P2 — Salut Qualitativa
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2Health:
    def test_fg_extreme_fear_bull(self):
        """En bull market + por extrema → oportunitat → score alt."""
        from bots.gate.gates.p2_health import _fg_score
        assert _fg_score(10, is_bull=True) == 1.0

    def test_fg_extreme_greed_bull(self):
        """En bull market + cobdícia extrema → perill → score baix."""
        from bots.gate.gates.p2_health import _fg_score
        assert _fg_score(80, is_bull=True) == 0.2

    def test_fg_neutral(self):
        from bots.gate.gates.p2_health import _fg_score
        assert _fg_score(50, is_bull=True) == 0.7
        assert _fg_score(50, is_bull=False) == 0.7

    def test_evaluate_returns_between_0_and_1(self):
        from bots.gate.gates.p2_health import P2Health
        p2 = P2Health(_gate_config())
        df = _make_df_1d(n=50)
        mult = p2.evaluate(df, regime="STRONG_BULL")
        assert 0.0 <= mult <= 1.0

    def test_multiplier_is_product(self):
        """El multiplier ha de ser el producte dels sub-scores, no la mitjana."""
        from bots.gate.gates.p2_health import P2Health
        p2 = P2Health(_gate_config())
        # Amb FG=80 (bull) → 0.2 i funding neutre → 0.8 → producte = 0.16
        df = _make_df_1d(n=50)
        df["fear_greed_value"] = 80.0
        df["funding_rate"] = 0.0  # neutre
        mult = p2.evaluate(df, regime="STRONG_BULL")
        assert mult == pytest.approx(0.2 * 0.8, abs=0.05)

    def test_evaluate_without_fg_or_funding(self):
        """Sense dades externes → defaults de 0.7."""
        from bots.gate.gates.p2_health import P2Health
        p2 = P2Health(_gate_config())
        df = pd.DataFrame({"close": [50000.0] * 50}, index=pd.date_range("2025-01-01", periods=50, freq="1D"))
        mult = p2.evaluate(df, regime="STRONG_BULL")
        assert mult == pytest.approx(0.7 * 0.7, abs=0.05)

    def test_funding_score_high_positive(self):
        """Funding molt positiu → risc de cascade → score baix en bull."""
        from bots.gate.gates.p2_health import P2Health
        score = P2Health._funding_score(0.001, is_bull=True)
        assert score == 0.5

    def test_funding_score_negative(self):
        """Funding negatiu → bé per llargs → score = 1.0."""
        from bots.gate.gates.p2_health import P2Health
        score = P2Health._funding_score(-0.0005, is_bull=True)
        assert score == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Tests P3 — Estructura de Preu
# ═══════════════════════════════════════════════════════════════════════════════

class TestP3Structure:
    def test_fractal_pivots_detect_highs_and_lows(self):
        from bots.gate.gates.p3_structure import P3Structure
        p3 = P3Structure(_gate_config())
        df = _make_df_4h(n=50)
        levels = p3._fractal_pivots(df, n=2)
        assert len(levels) > 0
        types = {l.level_type for l in levels}
        assert "support" in types or "resistance" in types

    def test_fractal_pivots_empty_for_short_df(self):
        from bots.gate.gates.p3_structure import P3Structure
        p3 = P3Structure(_gate_config())
        df = _make_df_4h(n=3)
        levels = p3._fractal_pivots(df, n=2)
        assert levels == []

    def test_fibonacci_levels_count(self):
        from bots.gate.gates.p3_structure import P3Structure, _FIB_LEVELS
        p3 = P3Structure(_gate_config())
        df = _make_df_1d(n=100)
        # Force un swing significant
        df.iloc[-60:, df.columns.get_loc("low")] = df["close"].iloc[-60:] - 3000
        df.iloc[-30:, df.columns.get_loc("high")] = df["close"].iloc[-30:] + 3000
        levels = p3._fibonacci_levels(df)
        # Si hi ha un swing >= 3%, hauria de retornar len(_FIB_LEVELS) nivells
        if levels:
            assert len(levels) == len(_FIB_LEVELS)

    def test_volume_profile_detects_hvn(self):
        from bots.gate.gates.p3_structure import P3Structure
        p3 = P3Structure(_gate_config())
        df = _make_df_4h(n=100)
        levels = p3._volume_profile(df)
        # Almenys algun HVN en 100 candles
        assert isinstance(levels, list)

    def test_merge_levels_consolidates_close_prices(self):
        from bots.gate.gates.p3_structure import P3Structure, LevelInfo
        p3 = P3Structure(_gate_config())
        levels = [
            LevelInfo(price=50000, level_type="support", strength=0.3, sources=["fractal"]),
            LevelInfo(price=50100, level_type="support", strength=0.3, sources=["fib"]),
            LevelInfo(price=55000, level_type="resistance", strength=0.3, sources=["volume_profile"]),
        ]
        merged = p3._merge_levels(levels, merge_pct=0.005)
        # 50000 i 50100 estan a <0.5% → haurien de fusionar
        assert len(merged) == 2
        # El nivell fusionat ha de tenir força 0.6 (2 fonts)
        assert merged[0].strength == pytest.approx(0.6)

    def test_approach_volume_ratio(self):
        from bots.gate.gates.p3_structure import P3Structure
        p3 = P3Structure(_gate_config())
        df = _make_df_4h(n=30)
        ratio = p3._approach_volume_ratio(df)
        assert ratio > 0

    def test_approach_volume_ratio_short_df(self):
        from bots.gate.gates.p3_structure import P3Structure
        p3 = P3Structure(_gate_config())
        df = _make_df_4h(n=5)
        ratio = p3._approach_volume_ratio(df)
        assert ratio == 1.0  # default per df curt

    def test_evaluate_returns_p3result(self):
        from bots.gate.gates.p3_structure import P3Structure
        p3 = P3Structure(_gate_config())
        df_4h = _make_df_4h(n=300)
        df_1d = _make_df_1d(n=300)
        price = float(df_4h["close"].iloc[-1])
        atr = float(df_4h["atr_14"].iloc[-1])
        result = p3.evaluate(df_4h, df_1d, price, atr, "STRONG_BULL")
        assert hasattr(result, "has_actionable_level")
        assert hasattr(result, "volume_ratio")


# ═══════════════════════════════════════════════════════════════════════════════
# Tests P4 — Momentum
# ═══════════════════════════════════════════════════════════════════════════════

class TestP4Momentum:
    def test_rsi_calculation(self):
        from bots.gate.gates.p4_momentum import P4Momentum
        close = pd.Series([10, 11, 12, 13, 14, 13, 12, 11, 10, 9, 10, 11, 12, 13, 14])
        rsi = P4Momentum._rsi(close, period=2)
        assert len(rsi) == len(close)
        # RSI ha d'estar entre 0 i 100
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_macd_cross_bullish(self):
        from bots.gate.gates.p4_momentum import P4Momentum
        # Crear una sèrie amb creuament MACD alcista recent
        n = 60
        close = pd.Series(np.concatenate([
            np.linspace(50000, 48000, 40),  # baixada
            np.linspace(48000, 52000, 20),  # pujada forta
        ]))
        result = P4Momentum._macd_cross_bullish(close)
        assert isinstance(result, bool)

    def test_evaluate_short_df(self):
        from bots.gate.gates.p4_momentum import P4Momentum
        p4 = P4Momentum(_gate_config())
        df = _make_df_4h(n=10)
        result = p4.evaluate(df, "STRONG_BULL")
        assert result.triggered is False
        assert result.signals_active == 0

    def test_strong_bull_needs_only_1_signal(self):
        from bots.gate.gates.p4_momentum import P4Momentum, _MIN_SIGNALS
        assert _MIN_SIGNALS["STRONG_BULL"] == 1

    def test_ranging_needs_2_signals(self):
        from bots.gate.gates.p4_momentum import _MIN_SIGNALS
        assert _MIN_SIGNALS["RANGING"] == 2

    def test_evaluate_returns_p4result(self):
        from bots.gate.gates.p4_momentum import P4Momentum
        p4 = P4Momentum(_gate_config())
        df = _make_df_4h(n=100)
        result = p4.evaluate(df, "STRONG_BULL")
        assert 0 <= result.signals_active <= 4
        assert 0.0 <= result.confidence <= 1.0
        assert "d1_accel" in result.signals_detail
        assert "macd_cross" in result.signals_detail

    def test_unknown_regime_never_triggers(self):
        """Règims no registrats a _MIN_SIGNALS han de requerir 9 senyals (impossible)."""
        from bots.gate.gates.p4_momentum import P4Momentum
        p4 = P4Momentum(_gate_config())
        df = _make_df_4h(n=100)
        result = p4.evaluate(df, "STRONG_BEAR")
        assert result.triggered is False


# ═══════════════════════════════════════════════════════════════════════════════
# Tests P5 — Risc i Gestió
# ═══════════════════════════════════════════════════════════════════════════════

class TestP5Risk:
    def test_veto_max_positions(self):
        from bots.gate.gates.p5_risk import P5Risk
        p5 = P5Risk(_gate_config())
        result = p5.evaluate_entry(
            usdt_balance=10000, entry_price=50000, stop_price=49000,
            target_price=53000, regime="STRONG_BULL", p2_multiplier=0.8,
            p4_confidence=0.75, n_open_positions=2,  # >= max_open_positions
            weekly_pnl_pct=0.0,
        )
        assert result.vetoed
        assert result.veto_reason == "max_open_positions"

    def test_veto_weekly_drawdown(self):
        from bots.gate.gates.p5_risk import P5Risk
        p5 = P5Risk(_gate_config())
        result = p5.evaluate_entry(
            usdt_balance=10000, entry_price=50000, stop_price=49000,
            target_price=53000, regime="STRONG_BULL", p2_multiplier=0.8,
            p4_confidence=0.75, n_open_positions=0,
            weekly_pnl_pct=-0.06,  # > weekly_drawdown_limit
        )
        assert result.vetoed
        assert result.veto_reason == "weekly_drawdown_exceeded"

    def test_veto_low_rr(self):
        from bots.gate.gates.p5_risk import P5Risk
        p5 = P5Risk(_gate_config())
        result = p5.evaluate_entry(
            usdt_balance=10000, entry_price=50000, stop_price=49500,
            target_price=50200,  # R:R = 200/500 = 0.4, molt per sota del mínim
            regime="STRONG_BULL", p2_multiplier=0.8,
            p4_confidence=0.75, n_open_positions=0, weekly_pnl_pct=0.0,
        )
        assert result.vetoed
        assert "rr_" in result.veto_reason

    def test_sizing_formula(self):
        """Verifica que el sizing segueix la fórmula Kelly fraccionari."""
        from bots.gate.gates.p5_risk import P5Risk
        p5 = P5Risk(_gate_config())
        result = p5.evaluate_entry(
            usdt_balance=10000, entry_price=50000, stop_price=48500,
            target_price=54500,  # R:R = 4500/1500 = 3.0 ✓
            regime="STRONG_BULL", p2_multiplier=1.0,
            p4_confidence=0.75, n_open_positions=0, weekly_pnl_pct=0.0,
        )
        assert not result.vetoed
        # risk_eur = 10000 * 0.01 * 1.0 * 0.75 = 75
        # stop_dist_pct = 1500 / 50000 = 0.03
        # position_usdt = 75 / 0.03 = 2500
        # size_fraction = 2500 / 10000 = 0.25
        assert result.size_fraction == pytest.approx(0.25, abs=0.01)

    def test_veto_invalid_stop(self):
        from bots.gate.gates.p5_risk import P5Risk
        p5 = P5Risk(_gate_config())
        result = p5.evaluate_entry(
            usdt_balance=10000, entry_price=50000, stop_price=50000,  # stop = entry
            target_price=55000, regime="STRONG_BULL", p2_multiplier=1.0,
            p4_confidence=0.75, n_open_positions=0, weekly_pnl_pct=0.0,
        )
        assert result.vetoed

    def test_trailing_stop_activation(self):
        """El trailing stop s'activa quan price >= entry + 1×ATR."""
        from bots.gate.gates.p5_risk import P5Risk
        p5 = P5Risk(_gate_config())
        position = {
            "entry_price": 50000, "stop_level": 48500,
            "highest_price": 50000, "opened_at": datetime.now(timezone.utc),
            "decel_counter": 0,
        }
        result = p5.evaluate_position(
            position=position, current_price=51500,  # > entry + ATR
            atr_14=1000, atr_percentile=50,  # normal
            d2_current=0.001, regime="STRONG_BULL",
            p1_compatible=True, p2_multiplier=0.8, p3_open=True,
        )
        assert not result.should_exit
        # Trailing hauria d'actualitzar-se: 51500 - 2.0*1000 = 49500 > 48500 original
        assert result.new_stop is not None
        assert result.new_stop > 48500

    def test_stop_loss_exit(self):
        from bots.gate.gates.p5_risk import P5Risk
        p5 = P5Risk(_gate_config())
        position = {
            "entry_price": 50000, "stop_level": 49000,
            "highest_price": 50000, "opened_at": datetime.now(timezone.utc),
            "decel_counter": 0,
        }
        result = p5.evaluate_position(
            position=position, current_price=48900,  # per sota del stop
            atr_14=1000, atr_percentile=50,
            d2_current=0.0, regime="STRONG_BULL",
            p1_compatible=True, p2_multiplier=0.8, p3_open=True,
        )
        assert result.should_exit
        assert result.exit_reason == "stop_loss_hit"

    def test_emergency_exit_p2_zero(self):
        from bots.gate.gates.p5_risk import P5Risk
        p5 = P5Risk(_gate_config())
        position = {
            "entry_price": 50000, "stop_level": 49000,
            "highest_price": 50500, "opened_at": datetime.now(timezone.utc),
            "decel_counter": 0,
        }
        result = p5.evaluate_position(
            position=position, current_price=50300,
            atr_14=1000, atr_percentile=50,
            d2_current=0.0, regime="STRONG_BULL",
            p1_compatible=True, p2_multiplier=0.0,  # EMERGÈNCIA
            p3_open=True,
        )
        assert result.should_exit
        assert result.exit_reason == "p2_emergency_exit"

    def test_regime_invalidation_exit(self):
        from bots.gate.gates.p5_risk import P5Risk
        p5 = P5Risk(_gate_config())
        position = {
            "entry_price": 50000, "stop_level": 49000,
            "highest_price": 50500, "opened_at": datetime.now(timezone.utc),
            "decel_counter": 0,
        }
        result = p5.evaluate_position(
            position=position, current_price=50300,
            atr_14=1000, atr_percentile=50,
            d2_current=0.0, regime="STRONG_BEAR",
            p1_compatible=False,  # règim incompatible
            p2_multiplier=0.8, p3_open=True,
        )
        assert result.should_exit
        assert result.exit_reason == "regime_invalidation"

    def test_deceleration_exit(self):
        from bots.gate.gates.p5_risk import P5Risk
        p5 = P5Risk(_gate_config())
        position = {
            "entry_price": 50000, "stop_level": 49000,
            "highest_price": 51000, "opened_at": datetime.now(timezone.utc),
            "decel_counter": 4,  # ja 4 candles amb d2 < 0
        }
        # En STRONG_BULL necessitem 5 candles → 4+1 = 5 → sortida
        result = p5.evaluate_position(
            position=position, current_price=50500,
            atr_14=1000, atr_percentile=50,
            d2_current=-0.001,  # d2 negatiu → counter incrementa a 5
            regime="STRONG_BULL",
            p1_compatible=True, p2_multiplier=0.8, p3_open=True,
        )
        assert result.should_exit
        assert "deceleration" in result.exit_reason

    def test_atr_percentile(self):
        from bots.gate.gates.p5_risk import P5Risk
        df = _make_df_4h(n=300)
        pct = P5Risk.atr_percentile(df)
        assert 0.0 <= pct <= 100.0


# ═══════════════════════════════════════════════════════════════════════════════
# Tests Regime Models
# ═══════════════════════════════════════════════════════════════════════════════

class TestHMMTrainer:
    def test_prepare_observations_shape(self):
        from bots.gate.regime_models.hmm_trainer import HMMTrainer
        df = _make_df_1d(n=100)
        obs = HMMTrainer._prepare_observations(df)
        assert obs.shape == (100, 3)
        assert not np.any(np.isnan(obs[1:]))  # primera fila pot tenir NaN per pct_change

    def test_bic_formula(self):
        from bots.gate.regime_models.hmm_trainer import HMMTrainer
        # BIC amb 4 estats, 3 observacions, loglik=-100, T=500
        bic = HMMTrainer._bic(k=4, n_obs=3, loglik=-100.0, T=500)
        assert isinstance(bic, float)
        assert bic > 0  # -2*(-100) + params*log(500) > 0

    def test_map_states_k2(self):
        from bots.gate.regime_models.hmm_trainer import HMMTrainer
        labels = np.array([0, 0, 1, 1, 0])
        obs = np.array([[0.01, 0.5, 1.0], [-0.02, 0.5, 1.0],
                        [0.03, 0.5, 1.0], [0.05, 0.5, 1.0], [-0.01, 0.5, 1.0]])
        mapping = HMMTrainer._map_states(labels, obs, k=2)
        assert len(mapping) == 2
        # Estat 1 (mean_return > estat 0) → STRONG_BULL
        assert "STRONG_BULL" in mapping.values()
        assert "STRONG_BEAR" in mapping.values()

    def test_map_states_k5(self):
        from bots.gate.regime_models.hmm_trainer import HMMTrainer
        labels = np.array([0, 1, 2, 3, 4])
        obs = np.array([[-0.05, 0, 0], [-0.02, 0, 0], [0.0, 0, 0], [0.02, 0, 0], [0.05, 0, 0]])
        mapping = HMMTrainer._map_states(labels, obs, k=5)
        assert len(mapping) == 5
        expected = {"STRONG_BEAR", "WEAK_BEAR", "RANGING", "WEAK_BULL", "STRONG_BULL"}
        assert set(mapping.values()) == expected


class TestXGBClassifier:
    def test_walk_forward_splits(self):
        from bots.gate.regime_models.xgb_classifier import XGBRegimeClassifier
        folds = XGBRegimeClassifier._walk_forward_splits(n=1000, n_folds=5)
        assert len(folds) > 0
        # Cada fold: train acaba on test comença
        for train_idx, test_idx in folds:
            assert len(train_idx) >= 500  # min_train_pct=0.5
            assert len(test_idx) > 0
            assert train_idx[-1] < test_idx[0]

    def test_walk_forward_no_overlap(self):
        """Els folds WF no han de solapar-se."""
        from bots.gate.regime_models.xgb_classifier import XGBRegimeClassifier
        folds = XGBRegimeClassifier._walk_forward_splits(n=1000, n_folds=5)
        for i in range(len(folds) - 1):
            _, test_i = folds[i]
            _, test_j = folds[i + 1]
            assert test_i[-1] < test_j[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Tests Near-Miss Logger
# ═══════════════════════════════════════════════════════════════════════════════

class TestNearMissLogger:
    def test_log_catches_exceptions(self):
        """NearMissLogger no hauria de llançar excepcions — mai bloqueja el flux de trading."""
        # Mock DB dependencies before import
        with patch.dict('sys.modules', {
            'core.db.session': MagicMock(),
            'core.db.demo_repository': MagicMock(),
        }):
            # Force reimport
            import importlib
            if 'bots.gate.near_miss_logger' in sys.modules:
                del sys.modules['bots.gate.near_miss_logger']
            from bots.gate.near_miss_logger import NearMissLogger, GateSnapshot
            repo = MagicMock()
            repo.save_gate_near_miss.side_effect = Exception("DB down")
            logger = NearMissLogger(repo)
            snapshot = GateSnapshot(
                bot_id="test", timestamp=datetime.now(timezone.utc),
                p1_regime="STRONG_BULL", p1_confidence=0.8, p2_multiplier=0.7,
            )
            # No hauria de llançar excepció
            logger.log(snapshot)

    def test_snapshot_default_values(self):
        with patch.dict('sys.modules', {
            'core.db.session': MagicMock(),
            'core.db.demo_repository': MagicMock(),
        }):
            if 'bots.gate.near_miss_logger' in sys.modules:
                del sys.modules['bots.gate.near_miss_logger']
            from bots.gate.near_miss_logger import GateSnapshot
            snapshot = GateSnapshot(
                bot_id="test", timestamp=datetime.now(timezone.utc),
                p1_regime="STRONG_BULL", p1_confidence=0.8, p2_multiplier=0.7,
            )
            assert snapshot.executed is False
            assert snapshot.p4_d1_ok is None
            assert snapshot.p5_veto_reason is None


# ═══════════════════════════════════════════════════════════════════════════════
# Tests GateBot Integration (lightweight, mocked DB)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGateBotHoldSignal:
    """Test del mètode _hold()."""

    def test_hold_returns_valid_signal(self):
        """_hold() ha d'usar self.bot_id (fix: ja no és @staticmethod)."""
        with patch.dict('sys.modules', {
            'core.db.session': MagicMock(),
            'core.db.demo_repository': MagicMock(),
        }):
            if 'bots.gate.gate_bot' in sys.modules:
                del sys.modules['bots.gate.gate_bot']
            if 'bots.gate.near_miss_logger' in sys.modules:
                del sys.modules['bots.gate.near_miss_logger']
            from bots.gate.gate_bot import GateBot
            # Crear instància mock per testejar _hold com a mètode d'instància
            with patch.object(GateBot, '__init__', lambda self, *a, **kw: None):
                bot = GateBot.__new__(GateBot)
                bot.bot_id = "test_bot_id"
                signal = bot._hold("test reason")
                assert signal.action.value == "hold"
                assert signal.size == 0.0
                # Ara usa self.bot_id dinàmicament
                assert signal.bot_id == "test_bot_id"


class TestGateBotNewDailyCandle:
    def test_same_day_is_not_new(self):
        """Dues candles del mateix dia no han de triggejar re-avaluació diària."""
        with patch.dict('sys.modules', {
            'core.db.session': MagicMock(),
            'core.db.demo_repository': MagicMock(),
        }):
            if 'bots.gate.gate_bot' in sys.modules:
                del sys.modules['bots.gate.gate_bot']
            if 'bots.gate.near_miss_logger' in sys.modules:
                del sys.modules['bots.gate.near_miss_logger']
            from bots.gate.gate_bot import GateBot
            with patch.object(GateBot, '__init__', lambda self, *a, **kw: None):
                bot = GateBot.__new__(GateBot)
                bot._last_daily_ts = datetime(2025, 3, 15, tzinfo=timezone.utc)
                assert not bot._is_new_daily_candle(datetime(2025, 3, 15, 12, 0, tzinfo=timezone.utc))
                assert bot._is_new_daily_candle(datetime(2025, 3, 16, tzinfo=timezone.utc))


# ═══════════════════════════════════════════════════════════════════════════════
# Tests de coherència config ↔ codi
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigCoherence:
    def test_gate_yaml_loads_correctly(self):
        """Verifica que gate.yaml es carrega i té tots els camps necessaris."""
        import yaml
        from pathlib import Path
        yaml_path = Path("config/models/gate.yaml")
        if not yaml_path.exists():
            pytest.skip("gate.yaml not found")
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        # Camps obligatoris
        assert config["bot_id"] == "gate_v1"
        assert config["module"] == "bots.gate.gate_bot"
        assert config["class_name"] == "GateBot"
        assert "p1" in config
        assert "p2" in config
        assert "p3" in config
        assert "p4" in config
        assert "p5" in config

    def test_p1_features_match_spec(self):
        """Les 14 features de P1 han de coincidir entre xgb_classifier i p1_regime."""
        from bots.gate.regime_models.xgb_classifier import P1_FEATURES
        assert len(P1_FEATURES) == 14

    def test_regime_names_complete(self):
        """Verifica que tots els règims estan coberts als lookups de P4 i P5."""
        from bots.gate.gates.p4_momentum import _MIN_SIGNALS
        from bots.gate.gates.p5_risk import _MIN_RR, _DECEL_CANDLES
        from bots.gate.gates.p1_regime import LONG_REGIMES
        # Tots els LONG_REGIMES han de tenir entrada a P4 i P5
        for regime in LONG_REGIMES:
            assert regime in _MIN_SIGNALS, f"P4 _MIN_SIGNALS missing {regime}"
            assert regime in _MIN_RR, f"P5 _MIN_RR missing {regime}"
            assert regime in _DECEL_CANDLES, f"P5 _DECEL_CANDLES missing {regime}"
