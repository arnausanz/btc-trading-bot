# bots/gate/gates/p3_structure.py
"""
Porta 3 — Estructura de Preu

Identifica nivells on el preu podria reaccionar.
NO prediu moviment — mapeja el camp de batalla.

Mètodes (tots deterministes, sense ML):
  - Pivots fractals: swing highs/lows (N=2 al 4H, N=5 al diari)
  - Fibonacci retracement: 38.2%, 50%, 61.8%, 78.6% de l'últim swing significatiu
  - Volume Profile: High Volume Nodes (bins amb volum > 1.5× la mitjana) al 4H
  - Filtre de volum d'acostament: si l'acostament és flàix → porta tancada
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Nivells de Fibonacci estàndard per a retracements
_FIB_LEVELS = [0.382, 0.500, 0.618, 0.786]


@dataclass
class LevelInfo:
    """Un nivell de preu candidat amb metadades."""
    price: float
    level_type: str          # 'support' | 'resistance'
    strength: float          # [0.0, 1.0]
    sources: list[str] = field(default_factory=list)   # ['fractal', 'fib', 'volume_profile']
    touches: int = 0         # nombre de tocs anteriors sense trencar


@dataclass
class P3Result:
    """Output de la Porta 3."""
    has_actionable_level: bool
    best_level: LevelInfo | None = None
    stop_level: float | None = None
    target_level: float | None = None
    risk_reward: float | None = None
    volume_ratio: float = 1.0   # loguejat per NearMissLogger


class P3Structure:
    """
    Porta 3: estructura de preu.
    Tots els càlculs són deterministes — cap paràmetre entrenable.
    """

    def __init__(self, config: dict):
        p3 = config.get("p3", {})
        # N fractals per timeframe
        self.fractal_n_4h    = int(p3.get("fractal_n_4h", 2))
        self.fractal_n_1d    = int(p3.get("fractal_n_1d", 5))
        # Força mínima per considerar un nivell accionable
        self.min_strength    = float(p3.get("min_level_strength", 0.4))
        # Swing mínim (%) per calcular Fibonacci
        self.min_swing_pct   = float(p3.get("min_swing_pct", 0.03))
        # Volume Profile: bin count
        self.vp_bins         = int(p3.get("volume_profile_bins", 50))

    # ──────────────────────────────────────────────────────────────────────
    # Punt d'entrada públic
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        df_4h: pd.DataFrame,
        df_1d: pd.DataFrame,
        current_price: float,
        atr_14: float,
        regime: str,
    ) -> P3Result:
        """
        Avalua si hi ha un nivell accionable a prop del preu actual.

        Args:
            df_4h: DataFrame 4H (mínim fractal_n_4h*2+1 files)
            df_1d: DataFrame diari (mínim fractal_n_1d*2+1 files)
            current_price: preu actual
            atr_14: ATR 14 del 4H (per proximity threshold)
            regime: règim P1 ('STRONG_BULL', etc.)
        """
        # 1. Recollir candidats de les tres fonts
        candidates: list[LevelInfo] = []
        candidates.extend(self._fractal_pivots(df_4h, self.fractal_n_4h))
        candidates.extend(self._fractal_pivots(df_1d, self.fractal_n_1d))
        candidates.extend(self._fibonacci_levels(df_1d))
        candidates.extend(self._volume_profile(df_4h))

        if not candidates:
            return P3Result(has_actionable_level=False)

        # 2. Consolidar nivells propers (merge a ±0.5% del preu)
        merged = self._merge_levels(candidates, merge_pct=0.005)

        # 3. Afegir tocs anteriors i recalcular força
        for lvl in merged:
            lvl.touches = self._count_touches(df_4h, lvl.price, atr_14)
            lvl.strength = min(1.0, lvl.strength + lvl.touches * 0.1)

        # 4. Filtre per força mínima
        viable = [l for l in merged if l.strength >= self.min_strength]
        if not viable:
            return P3Result(has_actionable_level=False)

        # 5. Proximity threshold adaptatiu per força
        def proximity(lvl: LevelInfo) -> float:
            if lvl.strength >= 0.7:
                return 2.0 * atr_14
            elif lvl.strength >= 0.4:
                return 1.5 * atr_14
            else:
                return 1.0 * atr_14

        # 6. Filtrar per proximitat al preu actual
        close_levels = [
            l for l in viable
            if abs(current_price - l.price) <= proximity(l)
        ]
        if not close_levels:
            return P3Result(has_actionable_level=False)

        # 7. Filtre de volum d'acostament (últimes 3 candles 4H)
        volume_ratio = self._approach_volume_ratio(df_4h)
        if volume_ratio < 0.8:
            logger.debug(
                f"P3 CLOSED — approach volume ratio {volume_ratio:.2f} < 0.8 "
                f"(low conviction approach)"
            )
            return P3Result(
                has_actionable_level=False,
                volume_ratio=volume_ratio,
            )

        # 8. Seleccionar el millor nivell (millor R:R)
        best, stop, target, rr = self._best_level_with_rr(
            close_levels, current_price, viable, regime
        )
        if best is None or rr is None:
            return P3Result(
                has_actionable_level=False,
                volume_ratio=volume_ratio,
            )

        return P3Result(
            has_actionable_level=True,
            best_level=best,
            stop_level=stop,
            target_level=target,
            risk_reward=rr,
            volume_ratio=volume_ratio,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Pivots fractals
    # ──────────────────────────────────────────────────────────────────────

    def _fractal_pivots(self, df: pd.DataFrame, n: int) -> list[LevelInfo]:
        """
        Detecta swing highs/lows usant la condició fractal de Williams.
        Un pivot high a l'índex i: high[i] > max(high[i-n:i] + high[i+1:i+n+1])
        """
        levels: list[LevelInfo] = []
        if len(df) < 2 * n + 1:
            return levels

        highs = df["high"].values
        lows  = df["low"].values

        for i in range(n, len(df) - n):
            # Pivot High (resistència)
            if highs[i] == max(highs[i - n: i + n + 1]):
                levels.append(LevelInfo(
                    price=float(highs[i]),
                    level_type="resistance",
                    strength=0.3,
                    sources=["fractal"],
                ))
            # Pivot Low (suport)
            if lows[i] == min(lows[i - n: i + n + 1]):
                levels.append(LevelInfo(
                    price=float(lows[i]),
                    level_type="support",
                    strength=0.3,
                    sources=["fractal"],
                ))
        return levels

    # ──────────────────────────────────────────────────────────────────────
    # Fibonacci retracement
    # ──────────────────────────────────────────────────────────────────────

    def _fibonacci_levels(self, df: pd.DataFrame) -> list[LevelInfo]:
        """
        Calcula nivells de Fibonacci de l'últim swing significatiu (≥ min_swing_pct).
        Detecta el swing com el parell (mínim, màxim) de la finestra més recent.
        """
        levels: list[LevelInfo] = []
        if len(df) < 20:
            return levels

        # Busca el swing alcista més recent (low → high)
        window = df.tail(60)   # últims 60 dies ≈ 2 mesos
        swing_low  = float(window["low"].min())
        swing_high = float(window["high"].max())
        swing_pct  = (swing_high - swing_low) / swing_low

        if swing_pct < self.min_swing_pct:
            return levels

        # Retracement des de high → low (per llargs, busquem suports al retracement)
        for fib in _FIB_LEVELS:
            price = swing_high - fib * (swing_high - swing_low)
            levels.append(LevelInfo(
                price=float(price),
                level_type="support",
                strength=0.3,
                sources=["fib"],
            ))

        return levels

    # ──────────────────────────────────────────────────────────────────────
    # Volume Profile
    # ──────────────────────────────────────────────────────────────────────

    def _volume_profile(self, df_4h: pd.DataFrame) -> list[LevelInfo]:
        """
        High Volume Nodes: bins de preu amb volum > 1.5× la mitjana.
        El volum concentrat en un preu indica que el mercat ha acceptat aquell nivell —
        serà un magnet o un rebuig en futures visites.
        """
        levels: list[LevelInfo] = []
        if len(df_4h) < 20:
            return levels

        prices = df_4h["close"].values
        volumes = df_4h["volume"].values

        price_min, price_max = prices.min(), prices.max()
        if price_min == price_max:
            return levels

        bins = np.linspace(price_min, price_max, self.vp_bins + 1)

        # Vectoritzat: assignar cada preu al seu bin i sumar volums
        bin_indices = np.digitize(prices, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.vp_bins - 1)
        bin_volume = np.bincount(bin_indices, weights=volumes, minlength=self.vp_bins)
        bin_volume = bin_volume[:self.vp_bins]  # assegurar mida correcta

        vol_mean = bin_volume.mean()
        for i, vol in enumerate(bin_volume):
            if vol > 1.5 * vol_mean:
                # Preu central del bin
                bin_price = (bins[i] + bins[i + 1]) / 2
                # En relació al preu actual: support o resistance
                current = prices[-1]
                ltype = "support" if bin_price < current else "resistance"
                levels.append(LevelInfo(
                    price=float(bin_price),
                    level_type=ltype,
                    strength=0.3,
                    sources=["volume_profile"],
                ))

        return levels

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _merge_levels(self, levels: list[LevelInfo], merge_pct: float) -> list[LevelInfo]:
        """
        Consolida nivells que estan a menys de merge_pct entre ells.
        Força final = 0.3 × nfonts + contribució extra.
        """
        if not levels:
            return []

        sorted_lvls = sorted(levels, key=lambda l: l.price)
        merged: list[LevelInfo] = []
        current = sorted_lvls[0]

        for lvl in sorted_lvls[1:]:
            # Si el nivell és prou proper, fusionar
            if (lvl.price - current.price) / current.price < merge_pct:
                # Preu mig ponderat per força
                w1, w2 = current.strength, lvl.strength
                current.price = (current.price * w1 + lvl.price * w2) / (w1 + w2)
                all_sources = list(set(current.sources + lvl.sources))
                current.sources = all_sources
                # Força: 1 font = 0.3, 2 = 0.6, 3 = 0.9
                current.strength = min(1.0, len(all_sources) * 0.3)
                # Si els dos nivells coincideixen en tipus, mantenir; si no, 'mixed'
                if current.level_type != lvl.level_type:
                    current.level_type = "mixed"
            else:
                merged.append(current)
                current = lvl

        merged.append(current)
        return merged

    def _count_touches(self, df_4h: pd.DataFrame, price: float, atr: float) -> int:
        """Compta les candles que han tocat el nivell sense tancar al costat contrari."""
        tolerance = atr * 0.5
        touches = int(
            ((df_4h["low"].values <= price + tolerance) &
             (df_4h["high"].values >= price - tolerance)).sum()
        )
        return touches

    def _approach_volume_ratio(self, df_4h: pd.DataFrame) -> float:
        """
        Ràtio volum d'acostament: mean(últimes 3 candles) / vol_ma20.
        Si < 0.8 → acostament flàix → porta tancada.
        """
        if len(df_4h) < 20:
            return 1.0  # sense dades suficients, assumir normal
        vol_approach = float(df_4h["volume"].iloc[-3:].mean())
        vol_ma       = float(df_4h["volume"].rolling(20).mean().iloc[-1])
        if vol_ma <= 0:
            return 1.0
        return vol_approach / vol_ma

    def _best_level_with_rr(
        self,
        close_levels: list[LevelInfo],
        current_price: float,
        all_levels: list[LevelInfo],
        regime: str,
    ) -> tuple[LevelInfo | None, float | None, float | None, float | None]:
        """
        Per a cada nivell proper (entrada candidata), busca stop i target
        als nivells veïns i calcula el R:R. Retorna el millor.
        """
        # Nivells per sobre i per sota del preu (per target i stop)
        above = sorted([l for l in all_levels if l.price > current_price],
                       key=lambda l: l.price)
        below = sorted([l for l in all_levels if l.price < current_price],
                       key=lambda l: l.price, reverse=True)

        best_rr   = -1.0
        best_lvl  = None
        best_stop = None
        best_tgt  = None

        for lvl in close_levels:
            # Stop: nivell de suport per sota del preu (si llarg)
            stop = below[0].price if below else current_price * 0.97
            # Target: primer nivell de resistència per sobre
            target = above[0].price if above else current_price * 1.03

            dist_stop   = current_price - stop
            dist_target = target - current_price

            if dist_stop <= 0:
                continue

            rr = dist_target / dist_stop
            if rr > best_rr:
                best_rr   = rr
                best_lvl  = lvl
                best_stop = stop
                best_tgt  = target

        if best_lvl is None:
            return None, None, None, None

        return best_lvl, best_stop, best_tgt, best_rr
