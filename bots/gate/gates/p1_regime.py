# bots/gate/gates/p1_regime.py
"""
Porta 1 — Règim Macro

Carrega els models entrenats (HMM + XGBoost) i fa inferència diària.
El GateBot crida evaluate() 1 cop/dia quan tanca una nova candle diària.

Output: P1Result(regime, confidence, probabilities)
  - regime:       'STRONG_BULL' | 'WEAK_BULL' | 'RANGING' | 'WEAK_BEAR' | 'STRONG_BEAR' | 'UNCERTAIN'
  - confidence:   probabilitat de l'estat dominant [0–1]
  - probabilities: {regime_name: prob} per a tots els estats

Si confidence < min_regime_confidence → regime = 'UNCERTAIN' → porta tancada.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from bots.gate.regime_models.xgb_classifier import XGBRegimeClassifier, P1_FEATURES

logger = logging.getLogger(__name__)

# Règims compatibles amb entrades LONG (v1 long-only)
LONG_REGIMES = {"STRONG_BULL", "WEAK_BULL", "RANGING"}

# Règims que invaliden una posició llarga oberta → sortida immediata
INVALIDATE_LONG = {"STRONG_BEAR", "WEAK_BEAR", "UNCERTAIN"}


@dataclass
class P1Result:
    """Output de la Porta 1."""
    regime: str                                # RegimeState com a string
    confidence: float                          # [0–1]
    probabilities: dict[str, float] = field(default_factory=dict)
    is_open: bool = False                      # False = porta tancada (UNCERTAIN o conf baixa)

    def allows_long(self) -> bool:
        """V1 long-only: operem únicament en règims alcistes o laterals."""
        return self.is_open and self.regime in LONG_REGIMES

    def invalidates_long_position(self) -> bool:
        """Retorna True si el règim actual invalida una posició llarga oberta."""
        return self.regime in INVALIDATE_LONG


class P1Regime:
    """
    Porta 1: càrrega + inferència del model de règim.

    El model es carrega una vegada a on_start() i es re-utilitza
    en cada crida diària a evaluate().
    """

    def __init__(self, config: dict):
        p1 = config.get("p1", {})
        self.min_confidence = float(p1.get("min_regime_confidence", 0.60))
        model_paths = config.get("model_paths", {})
        self.xgb_path = Path(model_paths.get("xgb", "models/gate_xgb_regime.pkl"))
        self._classifier: XGBRegimeClassifier | None = None
        self._last_result: P1Result | None = None   # cached per reutilitzar entre candles

    def load_models(self) -> None:
        """Carrega els models des de disc. Cridat a GateBot.on_start()."""
        if not self.xgb_path.exists():
            logger.warning(
                f"P1: model no trobat a {self.xgb_path}. "
                "Executa scripts/train_gate_regime.py primer."
            )
            return
        self._classifier = XGBRegimeClassifier.load(self.xgb_path)
        logger.info(f"P1: model carregat des de {self.xgb_path}")

    def evaluate(self, df_1d: pd.DataFrame) -> P1Result:
        """
        Avalua el règim actual sobre l'última fila del df diari.
        Cridat 1 cop/dia quan tanca la candle diària.

        Args:
            df_1d: DataFrame diari amb totes les features necessàries per calcular P1_FEATURES.
                   Mínim 252 files (1 any) per atr_14_percentile.
        """
        if self._classifier is None:
            logger.warning("P1: cap model carregat → UNCERTAIN")
            return P1Result(regime="UNCERTAIN", confidence=0.0, is_open=False)

        # ── Calcular les 14 features de P1 ───────────────────────────────
        try:
            features = self._compute_features(df_1d)
        except Exception as e:
            logger.error(f"P1: error calculant features: {e}")
            return P1Result(regime="UNCERTAIN", confidence=0.0, is_open=False)

        # ── Inferència XGBoost ────────────────────────────────────────────
        proba = self._classifier.predict_proba(features)

        # L'estat dominant i la seva probabilitat
        best_regime = max(proba, key=proba.get)
        confidence  = proba[best_regime]

        # Si la confiança és baixa → UNCERTAIN
        if confidence < self.min_confidence:
            logger.debug(
                f"P1: max_prob={confidence:.2f} < {self.min_confidence} → UNCERTAIN"
            )
            result = P1Result(
                regime="UNCERTAIN",
                confidence=confidence,
                probabilities=proba,
                is_open=False,
            )
        else:
            logger.info(
                f"P1: regime={best_regime}  conf={confidence:.2f}  "
                f"probs={{{', '.join(f'{k}: {v:.2f}' for k, v in sorted(proba.items()))}}}"
            )
            result = P1Result(
                regime=best_regime,
                confidence=confidence,
                probabilities=proba,
                is_open=True,
            )

        self._last_result = result
        return result

    def get_cached(self) -> P1Result | None:
        """Retorna l'últim resultat evaluat (entre candles diàries)."""
        return self._last_result

    # ──────────────────────────────────────────────────────────────────────
    # Càlcul de les 14 features de P1
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Computa les 14 features de P1 sobre l'última fila del df diari.
        Retorna un DataFrame d'1 fila alineat amb P1_FEATURES.
        """
        close  = df["close"]
        volume = df["volume"]

        # EMA 50 i 200
        ema50  = close.ewm(span=50,  adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()

        # ema_ratio_50_200: posició relativa de EMA50 vs EMA200
        ema_ratio = (ema50 / ema200).iloc[-1]

        # ema_slope_50: pendent lineal de l'EMA50 en 10 dies (normalitzat per preu)
        ema50_slope  = _linear_slope(ema50.iloc[-10:]) / float(close.iloc[-1])
        # ema_slope_200: pendent lineal de l'EMA200 en 20 dies
        ema200_slope = _linear_slope(ema200.iloc[-20:]) / float(close.iloc[-1])

        # adx_14: directament del df (ja calculat pel FeatureBuilder)
        adx = float(df["adx_14"].iloc[-1]) if "adx_14" in df.columns else 25.0

        # atr_14_percentile: percentil de l'ATR actual vs últims 252 dies
        atr_series = df["atr_14"] if "atr_14" in df.columns else pd.Series([0.0])
        atr_current = float(atr_series.iloc[-1])
        atr_hist    = atr_series.iloc[-252:]
        atr_pct     = float((atr_hist < atr_current).mean() * 100.0)

        # funding_rate: rolling means (si disponible)
        if "funding_rate" in df.columns:
            fr          = df["funding_rate"]
            fr_3d       = float(fr.rolling(3, min_periods=1).mean().iloc[-1])
            fr_7d       = float(fr.rolling(7, min_periods=1).mean().iloc[-1])
        else:
            fr_3d = fr_7d = 0.0

        # volume_sma_ratio i volume_trend
        vol_ma       = volume.rolling(20, min_periods=1).mean()
        vol_sma_rat  = float((volume / vol_ma).iloc[-1])
        vol_trend    = _linear_slope(vol_ma.iloc[-20:])

        # fear_greed
        if "fear_greed_value" in df.columns:
            fg_series = df["fear_greed_value"]
        elif "fear_greed" in df.columns:
            fg_series = df["fear_greed"]
        else:
            fg_series = pd.Series([50.0] * len(df))
        fg_current   = float(fg_series.iloc[-1])
        fg_7d_change = float(fg_series.iloc[-1]) - float(fg_series.iloc[-min(8, len(fg_series))])

        # rsi_14_daily
        rsi14 = float(df["rsi_14"].iloc[-1]) if "rsi_14" in df.columns else 50.0

        # returns
        ret5d  = float(close.pct_change(5).iloc[-1])
        ret20d = float(close.pct_change(20).iloc[-1])

        row = {
            "ema_ratio_50_200":   ema_ratio,
            "ema_slope_50":       ema50_slope,
            "ema_slope_200":      ema200_slope,
            "adx_14":             adx,
            "atr_14_percentile":  atr_pct,
            "funding_rate_3d":    fr_3d,
            "funding_rate_7d":    fr_7d,
            "volume_sma_ratio":   vol_sma_rat,
            "volume_trend":       vol_trend,
            "fear_greed":         fg_current,
            "fear_greed_7d_change": fg_7d_change,
            "rsi_14_daily":       rsi14,
            "returns_5d":         ret5d,
            "returns_20d":        ret20d,
        }

        result_df = pd.DataFrame([row])[P1_FEATURES]
        result_df = result_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return result_df


def _linear_slope(series: pd.Series) -> float:
    """Pendent de regressió lineal sobre una sèrie. Normalitzat per longitud."""
    n = len(series)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = series.values.astype(float)
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return 0.0
    slope = np.polyfit(x[valid], y[valid], deg=1)[0]
    return float(slope)
