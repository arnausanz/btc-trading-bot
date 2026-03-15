# bots/rl/constants.py
"""
Shared constants for all RL environments and reward functions.

Single source of truth — change here to affect the whole RL stack.
All values are documented with their role and the files that consume them.
"""

# ── Volatility reference ───────────────────────────────────────────────────────
# "Normal" 12H volatility for BTC expressed as a fraction of price.
# Used in: rewards/professional.py (atr_scale), environment_professional.py (ATR fallback)
ATR_REFERENCE: float = 0.02

# ── Observation normalisation ─────────────────────────────────────────────────
# Normalised observation features are hard-clipped to ±CLIP_OBS_BOUNDS to
# protect the neural network from float32 overflow and extreme outliers.
# Used in: environment.py (_get_observation), environment_professional.py (inherited)
CLIP_OBS_BOUNDS: float = 10.0

# ── Numerical stability ───────────────────────────────────────────────────────
# Minimum value added to denominators to avoid division by zero.
# Used in: environment.py (_get_observation), environment_professional.py
NUMERICAL_EPSILON: float = 1e-8

# ── Continuous action deadband ────────────────────────────────────────────────
# For SAC / TD3: if |target_fraction - current_fraction| < DEADBAND, the
# rebalance is skipped entirely. Prevents micro-trading fee bleed.
# Used in: environment_professional.py (BtcTradingEnvProfessionalContinuous)
DEADBAND: float = 0.05
