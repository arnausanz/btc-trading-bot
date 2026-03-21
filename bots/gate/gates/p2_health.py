# bots/gate/gates/p2_health.py
"""
Porta 2 — Salut Qualitativa

Modula la mida de la posició segons el context ambiental.
Output: position_multiplier [0.0–1.0]

El multiplier final és el PRODUCTE de tots els sub-scores.
Un sol component a 0 = veto total (multiplier=0).

V1: Fear & Greed (contrarian dins tendència) + On-chain flow proxy (funding rate).
News sentiment i LLM analysis queden per a v2.

Fonament:
  - FG contrarian: quan tothom té por (FG < 25) en bull market → oportunitat d'entrada.
  - Funding rate: positiu alt → mercat massa apalancat alcista → risc d'escorxament.
    Negatiu → mercat short-heavy → potencial short squeeze si tenim posició llarga.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ── Taula Fear & Greed → score ────────────────────────────────────────────────
# Contrarian: en bull market, por extrema = oportunitat, cobdícia extrema = perill.
# Format: (fg_min, fg_max): (bull_score, bear_score)
_FG_TABLE = [
    (0,   25,  1.0, 1.0),   # por extrema → bull opp / bear coherent
    (25,  40,  0.9, 0.8),
    (40,  60,  0.7, 0.7),   # neutre
    (60,  75,  0.5, 0.5),
    (75, 101,  0.2, 0.3),   # cobdícia extrema → bull perill / bear prudent
]

# ── Funding rate threshold (proxy d'on-chain flow) ────────────────────────────
# Un funding molt positiu indica posicionament apalancat en llarg → risc de cascade.
_FR_HIGH_THRESHOLD  = 0.0003   # > 0.03% per candle 8H → mercat apalancat bullish
_FR_NEG_THRESHOLD   = -0.0001  # < -0.01% → mercat posicionat short (bé per llargs)


def _fg_score(fg_value: float, is_bull: bool) -> float:
    """Mapeja el valor Fear & Greed → sub-score [0.0–1.0]."""
    for lo, hi, bull_s, bear_s in _FG_TABLE:
        if lo <= fg_value < hi:
            return bull_s if is_bull else bear_s
    return 0.7  # default


class P2Health:
    """
    Porta 2: salut qualitativa del context.

    En v1 usa:
      1. Fear & Greed (contrarian dins tendència)
      2. Funding rate com a proxy d'on-chain flow (sense API de pagament)
    """

    def __init__(self, config: dict):
        p2 = config.get("p2", {})
        self.onchain_enabled = bool(p2.get("onchain_enabled", True))

    def evaluate(self, df_1d: pd.DataFrame, regime: str) -> float:
        """
        Calcula el position_multiplier [0.0–1.0].

        Args:
            df_1d: DataFrame diari amb 'fear_greed_value' (o 'fear_greed') i 'funding_rate'
            regime: règim actual de P1 ('STRONG_BULL', 'WEAK_BULL', 'RANGING', ...)

        Returns:
            position_multiplier: 0.0 = veto total, 1.0 = mida màxima permesa
        """
        is_bull = regime in {"STRONG_BULL", "WEAK_BULL"}
        scores: list[float] = []

        # ── Sub-score 1: Fear & Greed ─────────────────────────────────────
        fg_col = "fear_greed_value" if "fear_greed_value" in df_1d.columns else "fear_greed"
        if fg_col in df_1d.columns:
            fg_value = float(df_1d[fg_col].iloc[-1])
            fg_s = _fg_score(fg_value, is_bull)
            scores.append(fg_s)
            logger.debug(f"P2: FG={fg_value:.0f}  score={fg_s:.2f}  (is_bull={is_bull})")
        else:
            logger.debug("P2: Fear & Greed no disponible, score=0.7")
            scores.append(0.7)

        # ── Sub-score 2: Funding rate (proxy on-chain) ───────────────────
        if self.onchain_enabled and "funding_rate" in df_1d.columns:
            fr_3d = float(df_1d["funding_rate"].rolling(3, min_periods=1).mean().iloc[-1])
            onchain_score = self._funding_score(fr_3d, is_bull)
            scores.append(onchain_score)
            logger.debug(f"P2: FR_3d={fr_3d:.5f}  onchain_score={onchain_score:.2f}")
        elif self.onchain_enabled:
            logger.debug("P2: funding_rate no disponible, onchain_score=0.7")
            scores.append(0.7)

        # ── Multiplier final = producte ───────────────────────────────────
        multiplier = 1.0
        for s in scores:
            multiplier *= s
        multiplier = min(1.0, max(0.0, multiplier))

        logger.info(f"P2: regime={regime}  multiplier={multiplier:.3f}  scores={[f'{s:.2f}' for s in scores]}")
        return multiplier

    @staticmethod
    def _funding_score(fr_3d: float, is_bull: bool) -> float:
        """
        Funding rate → sub-score.
        - Funding molt positiu + bull → mercat apalancat en llarg → reduir mida (0.5)
        - Funding negatiu → mercat posicionat short → bona senyal per llargs (1.0)
        - Neutre → 0.8
        """
        if fr_3d > _FR_HIGH_THRESHOLD:
            # Mercat molt apalancat en llarg → risc de liquidació en cascada
            return 0.5 if is_bull else 0.8
        elif fr_3d < _FR_NEG_THRESHOLD:
            # Mercat posicionat short → favorable per posicions llargues
            return 1.0
        else:
            return 0.8  # neutre
