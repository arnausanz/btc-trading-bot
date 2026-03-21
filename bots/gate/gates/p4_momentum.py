# bots/gate/gates/p4_momentum.py
"""
Porta 4 — Momentum / Derivades

Trigger d'entrada. Mesura l'estat actual del momentum (no prediu).
Tots els càlculs són deterministes — res entrenable.

4 senyals calculats sobre el 4H:
  1. d1 > 0 AND d2 >= 0   → momentum positiu accelerant
  2. d1 < 0 AND d2 > 0    → momentum negatiu frenant (possible reversal)
  3. RSI-2 < 10           → sobrevenut extrem (Connors)
  4. MACD(12,26,9) creuant senyal en alça

Mínim de senyals adaptatiu al règim P1 (lookup table).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mínim de senyals necessaris per règim i direcció (long-only en v1)
# format: {regime: min_signals_to_trigger}
_MIN_SIGNALS: dict[str, int] = {
    "STRONG_BULL": 1,
    "WEAK_BULL":   2,
    "RANGING":     2,
    # BEAR/UNCERTAIN → no s'arriba a P4 en v1 (long-only)
}


@dataclass
class P4Result:
    """Output de la Porta 4."""
    triggered: bool
    confidence: float                     # senyals_actius / total_senyals [0–1]
    signals_active: int                   # nombre de senyals actius
    signals_total: int = 4
    signals_detail: dict = field(default_factory=dict)  # {senyal: bool}
    d2_value: float = 0.0                 # última derivada 2a (usada per P5 trailing exit)


class P4Momentum:
    """
    Porta 4: trigger de momentum.
    """

    def __init__(self, config: dict):
        p4 = config.get("p4", {})
        self.ewm_span     = int(p4.get("ewm_span", 3))         # EWM span per suavitzar preu
        self.rsi2_threshold = float(p4.get("rsi2_oversold", 10))  # RSI-2 llindar sobrevenut

    # ──────────────────────────────────────────────────────────────────────
    # Punt d'entrada públic
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(self, df_4h: pd.DataFrame, regime: str) -> P4Result:
        """
        Avalua els 4 senyals de momentum i decideix si s'activa el trigger.

        Args:
            df_4h: DataFrame 4H amb almenys 30 files (per MACD i EWM)
            regime: règim de P1 ('STRONG_BULL', 'WEAK_BULL', 'RANGING', ...)
        """
        if len(df_4h) < 30:
            logger.warning("P4: DataFrame massa curt (<30 files), retornem no-trigger")
            return P4Result(triggered=False, confidence=0.0, signals_active=0)

        close = df_4h["close"]

        # ── Derivades sobre EWM-smoothed ─────────────────────────────────
        smoothed  = close.ewm(span=self.ewm_span, adjust=False).mean()
        d1        = smoothed.pct_change()
        d2        = d1.diff()
        d1_last   = float(d1.iloc[-1])
        d2_last   = float(d2.iloc[-1])

        # Senyal 1: momentum positiu accelerant
        s1 = bool(d1_last > 0 and d2_last >= 0)
        # Senyal 2: momentum negatiu frenant (reversal potencial)
        s2 = bool(d1_last < 0 and d2_last > 0)

        # ── RSI-2 (Connors oversold) ──────────────────────────────────────
        rsi2  = self._rsi(close, period=2)
        rsi2_last = float(rsi2.iloc[-1]) if not rsi2.empty else 50.0
        s3 = bool(rsi2_last < self.rsi2_threshold)

        # ── MACD(12,26,9) creuant senyal en alça ─────────────────────────
        s4 = self._macd_cross_bullish(close)

        signals_detail = {"d1_accel": s1, "d1_reversal": s2, "rsi2_oversold": s3, "macd_cross": s4}
        n_active = sum([s1, s2, s3, s4])
        confidence = n_active / 4.0

        # ── Mínim de senyals per règim ────────────────────────────────────
        min_signals = _MIN_SIGNALS.get(regime, 9)  # 9 = impossible → no trigger
        triggered = n_active >= min_signals

        logger.debug(
            f"P4 [{regime}]: signals={n_active}/4 | "
            f"min={min_signals} | triggered={triggered} | "
            f"d1={d1_last:.4f} d2={d2_last:.6f} rsi2={rsi2_last:.1f}"
        )

        return P4Result(
            triggered=triggered,
            confidence=confidence,
            signals_active=n_active,
            signals_total=4,
            signals_detail=signals_detail,
            d2_value=d2_last,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Helpers de càlcul
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        """RSI estàndard Wilder (EWM amb alpha = 1/period)."""
        delta = series.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def _macd_cross_bullish(close: pd.Series) -> bool:
        """
        MACD(12,26,9): el MACD line creuant la signal line de baix cap a dalt
        a l'última candle (candle anterior MACD < signal, actual MACD > signal).
        """
        if len(close) < 35:
            return False
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()

        # Creuament: a candle -1 macd estava per sota de signal, ara per sobre
        prev_diff = float(macd.iloc[-2]) - float(signal.iloc[-2])
        curr_diff = float(macd.iloc[-1]) - float(signal.iloc[-1])
        return bool(prev_diff <= 0 and curr_diff > 0)
