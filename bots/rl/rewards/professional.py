# bots/rl/rewards/professional.py
"""
Reward function that encodes professional trading principles from
docs/decisions/trading_policy_reference.md.

Extended signature: accepts extra context kwargs injected by
BtcTradingEnvProfessional.  When called from a base env (without context),
safe defaults kick in so it behaves like a slightly-enhanced simple reward.

Components
──────────
  base_pnl          ATR-scaled PnL: higher weight in low-vol, lower in high-vol.
  drawdown_penalty  Progressive quadratic penalty for deep drawdowns.
  overtrading       Penalty for opening a trade too soon after the last one.
  chaos_penalty     Penalty for entering when vol_ratio > 1.5 (chaos regime).
  patience_bonus    Tiny bonus for HOLDing when vol is compressed (< 0.8).
"""
import numpy as np
from bots.rl.rewards.registry import register

# ── Constants ────────────────────────────────────────────────────────────────
# ATR reference: 2 % of price is the "normal" 12H volatility baseline for BTC.
# Trades in lower-than-average vol get a slight reward boost; in higher-than-
# average vol they are dampened.
_ATR_REFERENCE = 0.02

# Minimum wait between trades to avoid overtrading (in steps).
# At 12H candles, 3 steps ≈ 1.5 days.
_MIN_STEPS_BETWEEN_TRADES = 3

# Drawdown thresholds
_DD_MILD_THRESHOLD     = 0.05   # 5 %  — linear penalty kicks in
_DD_MODERATE_THRESHOLD = 0.10   # 10 % — quadratic escalation
_DD_SEVERE_THRESHOLD   = 0.20   # 20 % — very aggressive quadratic


@register("professional")
def professional_risk_adjusted(
    prev_value: float,
    curr_value: float,
    action,
    in_position: bool,
    history: list[float],
    scaling: float = 100.0,
    # ── Extended context (injected by BtcTradingEnvProfessional) ────────────
    atr_pct: float = _ATR_REFERENCE,   # atr_14 / close; default = neutral
    vol_ratio: float = 1.0,            # atr_5 / atr_14; 1.0 = neutral
    drawdown_pct: float = 0.0,         # (portfolio - peak) / peak ≤ 0
    steps_since_trade: int = 999,      # steps since last trade
    position_fraction: float = 0.0,   # current fraction of capital in BTC
    trade_happened: bool = False,      # did a trade execute this step?
    **kwargs,                          # forward-compatibility
) -> float:
    """
    Risk-adjusted reward that encodes professional trading principles.

    Returns a float reward.  Positive means good, negative means bad.
    The magnitude is comparable to the `simple` reward (scaled by `scaling`).
    """
    if prev_value <= 0:
        return 0.0

    # ── 1. Base PnL (ATR-scaled) ─────────────────────────────────────────────
    # Normalize the step return by current volatility.
    # Low vol  → scale > 1 (each % return is more "impressive")
    # High vol → scale < 1 (high vol makes returns easy but risky)
    pnl_step = (curr_value - prev_value) / prev_value

    atr_scale = _ATR_REFERENCE / max(atr_pct, 0.005)   # prevent div-by-zero
    atr_scale = float(np.clip(atr_scale, 0.2, 5.0))

    base = pnl_step * scaling * atr_scale

    # ── 2. Drawdown penalty (progressive) ────────────────────────────────────
    dd = abs(float(drawdown_pct))   # positive magnitude

    if dd > _DD_SEVERE_THRESHOLD:
        # Quadratic, very aggressive above 20 %
        dd_penalty = scaling * (dd - _DD_MILD_THRESHOLD) ** 2 * 60.0
    elif dd > _DD_MODERATE_THRESHOLD:
        # Quadratic, strong above 10 %
        dd_penalty = scaling * (dd - _DD_MILD_THRESHOLD) ** 2 * 25.0
    elif dd > _DD_MILD_THRESHOLD:
        # Linear, mild 5–10 %
        dd_penalty = scaling * (dd - _DD_MILD_THRESHOLD) * 6.0
    else:
        dd_penalty = 0.0

    # ── 3. Overtrading penalty ────────────────────────────────────────────────
    # Penalise if a trade fires before _MIN_STEPS_BETWEEN_TRADES have elapsed.
    # Proportional to how early the trade is: 0 steps gap → full penalty.
    overtrading_penalty = 0.0
    if trade_happened and steps_since_trade < _MIN_STEPS_BETWEEN_TRADES:
        gap_ratio = 1.0 - steps_since_trade / _MIN_STEPS_BETWEEN_TRADES
        overtrading_penalty = scaling * gap_ratio * 0.15

    # ── 4. Chaos penalty ─────────────────────────────────────────────────────
    # Penalty for opening a NEW position when the market is in a chaotic
    # high-volatility regime (vol_ratio > 1.5).
    # "Don't trade the news" principle from the reference document.
    chaos_penalty = 0.0
    if trade_happened and vol_ratio > 1.5 and position_fraction > 0.05:
        chaos_penalty = scaling * (vol_ratio - 1.5) * 0.20

    # ── 5. Patience bonus ────────────────────────────────────────────────────
    # Very small bonus for HOLDing (action == 0 or no trade) when the market
    # is compressed (vol_ratio < 0.8).  Discourages FOMO entries and rewards
    # waiting for breakout confirmation.
    # MUST be tiny relative to expected trade profits to avoid HOLD-always bias.
    patience_bonus = 0.0
    if not trade_happened and not in_position and vol_ratio < 0.8:
        patience_bonus = scaling * 0.0005   # ≈ 0.05 % of `scaling`

    reward = (
        base
        - dd_penalty
        - overtrading_penalty
        - chaos_penalty
        + patience_bonus
    )

    # Safety clamp: prevent catastrophic reward spikes from NaN/inf inputs
    return float(np.clip(reward, -10.0 * scaling, 10.0 * scaling))
