# bots/gate/near_miss_logger.py
"""
NearMissLogger — Registre de candidats a trade (P1+P2+P3 passen)

S'enregistra CADA COP que P1, P2 i P3 passen simultàniament,
independentment de si el trade s'executa o no.

Motivació: el GateBot és conservador. Sense aquest log no sabem quantes
oportunitats perdem ni per quin motiu. Amb el log podem fer:

  -- Quina porta de P4 tanca més sovint?
  SELECT SUM(CASE WHEN NOT p4_d1_ok THEN 1 ELSE 0 END) ...
  FROM gate_near_misses WHERE NOT executed;

  -- P5 veta molt? Per quin motiu?
  SELECT p5_veto_reason, COUNT(*) FROM gate_near_misses
  WHERE p4_triggered AND NOT executed GROUP BY 1 ORDER BY 2 DESC;

  -- Candidats vs executats per setmana
  SELECT date_trunc('week', timestamp) AS week,
         COUNT(*) AS candidats, SUM(executed::int) AS executats
  FROM gate_near_misses GROUP BY 1 ORDER BY 1;
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.db.demo_repository import DemoRepository

logger = logging.getLogger(__name__)


@dataclass
class GateSnapshot:
    """
    Captura l'estat de totes les portes en un moment donat.
    Instanciada per GateBot quan P1+P2+P3 passen.
    """
    bot_id: str
    timestamp: datetime

    # P1
    p1_regime: str
    p1_confidence: float

    # P2
    p2_multiplier: float

    # P3
    p3_level_type: str | None = None
    p3_level_strength: float | None = None
    p3_risk_reward: float | None = None
    p3_volume_ratio: float | None = None

    # P4 (omplert per GateBot si P4 s'evalua)
    p4_d1_ok: bool | None = None
    p4_d2_ok: bool | None = None
    p4_rsi_ok: bool | None = None
    p4_macd_ok: bool | None = None
    p4_score: float | None = None
    p4_triggered: bool | None = None

    # P5
    p5_veto_reason: str | None = None
    p5_position_size: float | None = None

    # Resultat
    executed: bool = False


class NearMissLogger:
    """
    Persiste GateSnapshot a la taula gate_near_misses via DemoRepository.
    Es crida des de GateBot.on_observation() quan P3 passa.
    """

    def __init__(self, repo: DemoRepository):
        self._repo = repo

    def log(self, snapshot: GateSnapshot) -> None:
        """Guarda el snapshot. Errors no bloquegen el flux de trading."""
        try:
            self._repo.save_gate_near_miss({
                "timestamp":         snapshot.timestamp,
                "bot_id":            snapshot.bot_id,
                "p1_regime":         snapshot.p1_regime,
                "p1_confidence":     snapshot.p1_confidence,
                "p2_multiplier":     snapshot.p2_multiplier,
                "p3_level_type":     snapshot.p3_level_type,
                "p3_level_strength": snapshot.p3_level_strength,
                "p3_risk_reward":    snapshot.p3_risk_reward,
                "p3_volume_ratio":   snapshot.p3_volume_ratio,
                "p4_d1_ok":          snapshot.p4_d1_ok,
                "p4_d2_ok":          snapshot.p4_d2_ok,
                "p4_rsi_ok":         snapshot.p4_rsi_ok,
                "p4_macd_ok":        snapshot.p4_macd_ok,
                "p4_score":          snapshot.p4_score,
                "p4_triggered":      snapshot.p4_triggered,
                "p5_veto_reason":    snapshot.p5_veto_reason,
                "p5_position_size":  snapshot.p5_position_size,
                "executed":          snapshot.executed,
            })
        except Exception as e:
            # No bloquear el flux de trading per un error de logging
            logger.error(f"NearMissLogger: error guardant snapshot: {e}")
