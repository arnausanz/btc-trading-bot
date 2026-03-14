# bots/rl/rewards/advanced.py
"""
Advanced reward functions for C3-generation RL agents.

These rewards build on the `professional` reward and extend it with
regime-adaptive logic: the agent is rewarded or penalised differently
depending on the detected market regime (trending, ranging, or chaotic).

Functions defined here:
  regime_adaptive   — Detects trend/range/chaos and scales rewards accordingly.

Design principles:
  1. Regime detection via ADX (trend strength) + vol_ratio (volatility ratio).
     ADX  > 25   AND vol_ratio ∈ [0.8, 1.5]  → TRENDING regime
     ADX  < 20   AND vol_ratio < 0.8          → RANGING regime
     vol_ratio   > 1.5                         → CHAOTIC regime

  2. In TRENDING regime: reward trend-following entries, penalise premature exits.
  3. In RANGING  regime: reward patience (HOLD), penalise overtrading.
  4. In CHAOTIC  regime: strong penalty for new entries (same as professional but
     amplified); encourage flat exposure.

  5. All other reward components (drawdown, overtrading) remain identical to
     `professional` to ensure a fair apples-to-apples comparison.

Usage:
  Set reward_type: regime_adaptive in YAML  training.environment section.
  Requires BtcTradingEnvProfessionalContinuous (injects atr_pct, vol_ratio,
  adx, drawdown_pct, steps_since_trade, position_fraction, trade_happened).
"""
import numpy as np
from bots.rl.rewards.registry import register

# ── Shared constants ──────────────────────────────────────────────────────────
_ATR_REFERENCE      = 0.02   # 2 % of price = normal 12H BTC volatility baseline
_MIN_STEPS          = 3      # minimum steps between trades (overtrading guard)

# Drawdown penalty thresholds (same as professional)
_DD_MILD     = 0.05
_DD_MODERATE = 0.10
_DD_SEVERE   = 0.20

# Regime detection thresholds
_ADX_TREND_MIN  = 25.0   # ADX above this → trending
_ADX_RANGE_MAX  = 20.0   # ADX below this → ranging
_VOL_TREND_MAX  = 1.5    # vol_ratio upper bound for trending (not chaotic)
_VOL_RANGE_MAX  = 0.8    # vol_ratio upper bound for pure range
_VOL_CHAOS_MIN  = 1.5    # vol_ratio above this → chaotic


@register("regime_adaptive")
def regime_adaptive_reward(
    prev_value: float,
    curr_value: float,
    action,
    in_position: bool,
    history: list[float],
    scaling: float = 100.0,
    # ── Extended context (injected by BtcTradingEnvProfessional) ────────────
    atr_pct: float = _ATR_REFERENCE,
    vol_ratio: float = 1.0,
    drawdown_pct: float = 0.0,
    steps_since_trade: int = 999,
    position_fraction: float = 0.0,
    trade_happened: bool = False,
    # ── Additional context injected by MultiFrame environments ────────────
    adx: float = 20.0,     # ADX-14 (trend strength indicator 0–100)
    **kwargs,
) -> float:
    """
    Regime-adaptive reward that adjusts bonus/penalty weights based on
    the current market regime detected from ADX and vol_ratio.

    Regime logic:
      TRENDING  ADX > 25, vol_ratio ∈ [0.8, 1.5]
                → Amplify base PnL (trend-following pays more).
                → Bonus for holding profitable trend positions.
                → No extra patience bonus (standing still in a trend = missed money).

      RANGING   ADX < 20, vol_ratio < 0.8
                → Reduce base PnL weight (don't over-reward small oscillations).
                → Patience bonus amplified (waiting for breakout is correct behaviour).
                → Overtrading penalty amplified (whipsaw cost is highest here).

      CHAOTIC   vol_ratio > 1.5
                → Heavy penalty for opening new positions.
                → Patience bonus when already flat.
                → Same as professional chaos_penalty but stronger.

      NEUTRAL   Everything else.
                → Uses `professional`-equivalent weights.
    """
    if prev_value <= 0:
        return 0.0

    # ── Detect regime ─────────────────────────────────────────────────────────
    is_trending = (adx >= _ADX_TREND_MIN) and (_VOL_RANGE_MAX <= vol_ratio <= _VOL_TREND_MAX)
    is_ranging  = (adx <= _ADX_RANGE_MAX) and (vol_ratio < _VOL_RANGE_MAX)
    is_chaotic  = vol_ratio > _VOL_CHAOS_MIN

    # ── 1. Base PnL (ATR-scaled) ──────────────────────────────────────────────
    pnl_step = (curr_value - prev_value) / prev_value
    atr_scale = float(np.clip(_ATR_REFERENCE / max(atr_pct, 0.005), 0.2, 5.0))
    base = pnl_step * scaling * atr_scale

    # Regime amplification of base
    if is_trending:
        base *= 1.5   # trend-following: reward positive carry more
    elif is_ranging:
        base *= 0.7   # ranging: reduce noise-trading reward

    # ── 2. Drawdown penalty (same as professional) ────────────────────────────
    dd = abs(float(drawdown_pct))
    if dd > _DD_SEVERE:
        dd_penalty = scaling * (dd - _DD_MILD) ** 2 * 60.0
    elif dd > _DD_MODERATE:
        dd_penalty = scaling * (dd - _DD_MILD) ** 2 * 25.0
    elif dd > _DD_MILD:
        dd_penalty = scaling * (dd - _DD_MILD) * 6.0
    else:
        dd_penalty = 0.0

    # ── 3. Overtrading penalty ────────────────────────────────────────────────
    overtrading_penalty = 0.0
    if trade_happened and steps_since_trade < _MIN_STEPS:
        gap_ratio = 1.0 - steps_since_trade / _MIN_STEPS
        penalty_mult = 2.0 if is_ranging else 1.0   # amplify in ranging
        overtrading_penalty = scaling * gap_ratio * 0.15 * penalty_mult

    # ── 4. Regime-specific bonuses / penalties ────────────────────────────────
    regime_bonus = 0.0

    if is_chaotic:
        # Strongly penalise entering during chaos; reward being flat
        if trade_happened and position_fraction > 0.05:
            regime_bonus -= scaling * (vol_ratio - _VOL_CHAOS_MIN) * 0.40
        elif not in_position and not trade_happened:
            regime_bonus += scaling * 0.001   # flat is safe

    elif is_trending:
        # Bonus for holding a position during a trend
        if in_position and not trade_happened and pnl_step > 0:
            regime_bonus += scaling * 0.002   # "let the winner run"
        # Small penalty for exiting a profitable trend prematurely
        if trade_happened and position_fraction < 0.05 and pnl_step > 0:
            regime_bonus -= scaling * 0.003

    elif is_ranging:
        # Patience bonus: staying flat while vol is compressed
        if not in_position and not trade_happened:
            regime_bonus += scaling * 0.0015   # bigger than professional (0.0005)

    # ── 5. Total reward ───────────────────────────────────────────────────────
    reward = base - dd_penalty - overtrading_penalty + regime_bonus

    # Safety clamp
    return float(np.clip(reward, -10.0 * scaling, 10.0 * scaling))
