# bots/gate/gates/p5_risk.py
"""
Porta 5 — Risc i Gestió de Posicions

Dues responsabilitats:
  A) Sizing d'entrada: Kelly fraccionari + condicions de VETO
  B) Gestió de posicions obertes (cada 4H): trailing stop, sortides condicionals

El % de risc NO és la mida de la posició. És el que perds si toca el stop.

  risc_eur      = capital × max_risk_pct × multiplier_P2 × confidence_P4
  dist_stop_pct = |entry - stop| / entry
  posicio_usdt  = risc_eur / dist_stop_pct
  size_fraction = min(posicio_usdt / usdt_balance, max_exposure_pct)  ← Signal.size
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

# Trailing stop multiplier per percentil d'ATR
_TRAILING_MULT = {
    "low":    1.5,   # percentil ATR < 30 (mercat calm)
    "normal": 2.0,   # percentil ATR 30–70
    "high":   2.5,   # percentil ATR > 70 (mercat volàtil)
}

# Candles 4H consecutives amb d2 < 0 per sortida per desacceleració
_DECEL_CANDLES: dict[str, int] = {
    "STRONG_BULL": 5,
    "WEAK_BULL":   3,
    "RANGING":     2,
}

# R:R mínim per règim (long-only v1 — no hi ha entrades en BEAR)
_MIN_RR: dict[str, float] = {
    "STRONG_BULL": 1.5,
    "WEAK_BULL":   2.0,
    "RANGING":     2.0,
}


@dataclass
class P5EntryResult:
    """Resultat del sizing d'entrada."""
    vetoed: bool
    veto_reason: str | None = None
    size_fraction: float = 0.0      # Signal.size [0.0–1.0]
    position_usdt: float = 0.0      # mida en USDT (per logging)


@dataclass
class P5TrailingResult:
    """Resultat de la gestió d'una posició oberta."""
    should_exit: bool
    exit_reason: str | None = None
    new_stop: float | None = None    # None = stop no s'ha mogut
    reduce_half: bool = False        # True = reduir posició al 50% (estancament)


class P5Risk:
    """
    Porta 5: sizing d'entrada + gestió de posicions obertes.
    """

    def __init__(self, config: dict):
        p5 = config.get("p5", {})
        self.max_risk_pct          = float(p5.get("max_risk_pct",          0.01))   # 1%
        self.max_exposure_pct      = float(p5.get("max_exposure_pct",      0.95))   # 95%
        self.max_open_positions    = int(p5.get("max_open_positions",       2))
        self.weekly_drawdown_limit = float(p5.get("weekly_drawdown_limit",  0.05))  # 5%
        self.stagnation_days       = float(p5.get("stagnation_days",        6.0))
        # R:R mínims configurables (sobreescriuen valors per defecte si presents)
        cfg_min_rr = p5.get("min_rr", {})
        self.min_rr = {**_MIN_RR, **{k: float(v) for k, v in cfg_min_rr.items()}}
        # Trailing multipliers configurables
        cfg_trail = p5.get("trailing_atr_multiplier", {})
        self.trailing_mult = {
            "low":    float(cfg_trail.get("low_vol",    1.5)),
            "normal": float(cfg_trail.get("normal_vol", 2.0)),
            "high":   float(cfg_trail.get("high_vol",   2.5)),
        }
        # Decel exit candles configurables
        cfg_decel = p5.get("decel_exit_candles", {})
        self.decel_candles = {**_DECEL_CANDLES, **{k: int(v) for k, v in cfg_decel.items()}}

    # ──────────────────────────────────────────────────────────────────────
    # A) Sizing d'entrada
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_entry(
        self,
        usdt_balance: float,
        entry_price: float,
        stop_price: float,
        target_price: float,
        regime: str,
        p2_multiplier: float,
        p4_confidence: float,
        n_open_positions: int,
        weekly_pnl_pct: float,   # PnL setmanal com a fracció (negatiu = pèrdua)
    ) -> P5EntryResult:
        """
        Calcula la mida d'entrada o veta el trade.

        Returns P5EntryResult amb vetoed=True + veto_reason si no s'ha de fer el trade.
        """
        # VETO 1: massa posicions obertes
        if n_open_positions >= self.max_open_positions:
            return P5EntryResult(vetoed=True, veto_reason="max_open_positions")

        # VETO 2: drawdown setmanal excedit
        if weekly_pnl_pct <= -self.weekly_drawdown_limit:
            return P5EntryResult(vetoed=True, veto_reason="weekly_drawdown_exceeded")

        # VETO 3: R:R mínim no assolit
        dist_stop   = abs(entry_price - stop_price)
        dist_target = abs(target_price - entry_price)
        if dist_stop <= 0:
            return P5EntryResult(vetoed=True, veto_reason="invalid_stop_distance")

        rr = dist_target / dist_stop
        min_rr = self.min_rr.get(regime, 2.0)
        if rr < min_rr:
            return P5EntryResult(
                vetoed=True,
                veto_reason=f"rr_{rr:.2f}_below_min_{min_rr:.2f}",
            )

        # ── Sizing Kelly fraccionari ──────────────────────────────────────
        stop_dist_pct = dist_stop / entry_price
        risk_eur      = usdt_balance * self.max_risk_pct * p2_multiplier * p4_confidence
        position_usdt = risk_eur / stop_dist_pct

        # Cap: max_exposure_pct del capital disponible
        max_usdt     = usdt_balance * self.max_exposure_pct
        position_usdt = min(position_usdt, max_usdt)

        size_fraction = position_usdt / usdt_balance
        size_fraction = min(max(size_fraction, 0.01), self.max_exposure_pct)

        logger.debug(
            f"P5 SIZING: rr={rr:.2f} risk={risk_eur:.1f} USDT "
            f"pos={position_usdt:.1f} USDT size={size_fraction:.3f}"
        )

        return P5EntryResult(
            vetoed=False,
            size_fraction=size_fraction,
            position_usdt=position_usdt,
        )

    # ──────────────────────────────────────────────────────────────────────
    # B) Gestió de posicions obertes (cada 4H)
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_position(
        self,
        position: dict,
        current_price: float,
        atr_14: float,
        atr_percentile: float,    # 0–100
        d2_current: float,
        regime: str,
        p1_compatible: bool,      # True si el règim actual és compatible amb la posició
        p2_multiplier: float,
        p3_open: bool,            # True si P3 segueix tenint un nivell accionable
        prev_close: float | None = None,  # tancament candle anterior (per circuit breaker)
    ) -> P5TrailingResult:
        """
        Avalua si cal sortir, reduir o actualitzar el stop d'una posició oberta.

        Args:
            position: dict amb entry_price, stop_level, highest_price, opened_at,
                      size_usdt, regime, decel_counter
            current_price: preu actual
            atr_14: ATR 14 del 4H
            atr_percentile: percentil ATR (0–100)
            d2_current: derivada 2a actual (negatiu = decelerant)
            regime: règim P1 actual
            p1_compatible: True si el règim suporta la posició existent (long-only)
            p2_multiplier: 0.0 → sortida d'emergència
            p3_open: si P3 segueix oberta (nivell vàlid a prop)
        """
        entry_price  = position["entry_price"]
        stop_level   = position["stop_level"]
        highest_price = position.get("highest_price", entry_price)
        opened_at    = position.get("opened_at", datetime.now(timezone.utc))
        decel_counter = position.get("decel_counter", 0)

        # ── Sortida D: emergència (P2 = 0) ───────────────────────────────
        if p2_multiplier <= 0.0:
            return P5TrailingResult(
                should_exit=True, exit_reason="p2_emergency_exit"
            )

        # ── Sortida C: invalidació de règim ──────────────────────────────
        if not p1_compatible:
            return P5TrailingResult(
                should_exit=True, exit_reason="regime_invalidation"
            )

        # ── Circuit breaker: moviment > 3×ATR en 1 candle ────────────────
        ref_price = prev_close if prev_close is not None else entry_price
        if abs(current_price - ref_price) > 3.0 * atr_14:
            if current_price < ref_price:
                return P5TrailingResult(
                    should_exit=True, exit_reason="circuit_breaker_3atr"
                )

        # ── Sortida per stop loss ─────────────────────────────────────────
        if current_price <= stop_level:
            return P5TrailingResult(
                should_exit=True, exit_reason="stop_loss_hit"
            )

        # ── Trailing stop: actualitzar si preu ha avançat ≥ 1×ATR ────────
        new_stop = stop_level
        if current_price >= entry_price + atr_14:
            # Multiplicador adaptatiu per volatilitat
            if atr_percentile < 30:
                mult = self.trailing_mult["low"]
            elif atr_percentile < 70:
                mult = self.trailing_mult["normal"]
            else:
                mult = self.trailing_mult["high"]

            new_highest = max(highest_price, current_price)
            trailing    = new_highest - mult * atr_14
            new_stop    = max(stop_level, trailing)  # el stop mai baixa

        # ── Sortida B: desacceleració (d2 negatiu N candles consecutives) ─
        n_decel = self.decel_candles.get(regime, 3)
        if d2_current < 0:
            decel_counter += 1
        else:
            decel_counter = 0

        if decel_counter >= n_decel:
            return P5TrailingResult(
                should_exit=True,
                exit_reason=f"deceleration_{decel_counter}x",
                new_stop=new_stop,
            )

        # ── Sortida E: estancament prolongat (reducció al 50%) ───────────
        days_open = (datetime.now(timezone.utc) - opened_at).total_seconds() / 86400
        in_loss   = current_price < entry_price
        if in_loss and not p3_open and days_open > self.stagnation_days:
            return P5TrailingResult(
                should_exit=False,
                exit_reason=None,
                new_stop=new_stop,
                reduce_half=True,
            )

        # ── Cap condició de sortida — actualitzar trailing ────────────────
        return P5TrailingResult(
            should_exit=False,
            new_stop=new_stop if new_stop != stop_level else None,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def atr_percentile(df_4h: pd.DataFrame, window: int = 252) -> float:
        """Percentil actual de l'ATR vs la finestra d'historial."""
        if "atr_14" not in df_4h.columns or len(df_4h) < 10:
            return 50.0
        atr_series = df_4h["atr_14"].dropna()
        current_atr = float(atr_series.iloc[-1])
        hist = atr_series.iloc[-min(window, len(atr_series)):]
        pct = float((hist < current_atr).mean() * 100.0)
        return pct
