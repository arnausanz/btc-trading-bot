# bots/rl/environment_professional.py
"""
Professional trading environments for PPO (Discrete-5) and SAC (Continuous).

Key extensions over _BaseTradingEnv
────────────────────────────────────
  Position State   4 extra features appended to every observation:
                   pnl_pct, position_fraction, steps_in_position, drawdown_pct.
                   The agent can see WHERE it is in a trade, not just the market.

  ATR Stop Loss    Hard stop computed at entry: entry × (1 − k × ATR/price).
                   Forced SELL_FULL if price falls below stop.  The agent cannot
                   move the stop — preventing the classic "move the stop" error.

  Partial Trades   PPO variant: Discrete(5) — HOLD / BUY_FULL / BUY_PARTIAL /
                                              SELL_PARTIAL / SELL_FULL.
                   SAC variant: Continuous [0, 1] with deadband ±0.05 to
                   avoid micro-rebalancing that would eat commissions.

  Extended Reward  Overrides _compute_reward() to pass vol regime context to
                   the `professional` reward function (atr_pct, vol_ratio,
                   drawdown_pct, steps_since_trade, etc.).
"""

import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)

from bots.rl.environment import _BaseTradingEnv
# Import to trigger @register decorators for both builtin and professional reward fns
from bots.rl.rewards import builtins       # noqa: F401
from bots.rl.rewards import professional   # noqa: F401
from bots.rl.rewards.registry import get as get_reward
from bots.rl.constants import ATR_REFERENCE, NUMERICAL_EPSILON, DEADBAND


def _safe_col(row: pd.Series, col: str, default: float) -> float:
    """Safely fetch a column value from a row, handling missing columns and NaN."""
    if col not in row.index:
        return default
    val = row[col]
    if pd.isna(val):
        return default
    return float(val)


# ── Shared professional base ──────────────────────────────────────────────────

class _ProfessionalBase(_BaseTradingEnv):
    """
    Extends _BaseTradingEnv with:
      - Position state appended to observations  (+4 dims)
      - ATR-based stop loss (hard, forced by env)
      - Partial position tracking (position_fraction ∈ [0,1])
      - Extended _compute_reward() that passes market regime context

    NOT instantiated directly.  Use BtcTradingEnvProfessionalDiscrete
    or BtcTradingEnvProfessionalContinuous.
    """

    N_POS_FEATURES = 4   # pnl_pct, position_fraction, steps_in_position, drawdown_pct

    def __init__(
        self,
        stop_atr_multiplier: float = 2.0,
        **kwargs,
    ):
        """
        Args:
            stop_atr_multiplier:  How many ATR units below entry to place the stop.
                                  Default 2.0 gives room for normal 12H swing noise.
                                  Exposed to Optuna search_space (range 1.5–3.0).
            **kwargs:             Forwarded to _BaseTradingEnv (df, lookback, fees…).
        """
        # Parent sets self.n_features, self.initial_capital, self.lookback,
        # calls self._reset_state(), and sets the base observation_space.
        super().__init__(**kwargs)
        self.stop_atr_multiplier = stop_atr_multiplier

        # Override observation_space to include position state dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback * self.n_features + self.N_POS_FEATURES,),
            dtype=np.float32,
        )

    # ── State reset ───────────────────────────────────────────────────────────

    def _reset_state(self):
        super()._reset_state()
        # Position tracking
        self.entry_price: float = 0.0
        self.entry_atr: float   = 0.0
        self.stop_price: float  = 0.0
        self.position_fraction: float = 0.0
        self.steps_in_position: int   = 0
        self.steps_since_last_trade: int = 0
        self._trade_this_step: bool   = False
        self.peak_portfolio: float    = self.initial_capital

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_position_fraction(self) -> float:
        """Current fraction of portfolio value held in BTC [0, 1]."""
        price = self._get_price()
        total = self.usdt_balance + self.btc_balance * price
        if total < NUMERICAL_EPSILON:
            return 0.0
        return float(np.clip(self.btc_balance * price / total, 0.0, 1.0))

    def _get_position_state(self) -> np.ndarray:
        """4-dim position state vector, already in bounded ranges."""
        price = self._get_price()

        # 1. pnl_pct: unrealised return from entry price  [-1, +1]
        if self.in_position and self.entry_price > 0:
            pnl = (price - self.entry_price) / self.entry_price
        else:
            pnl = 0.0
        pnl = float(np.clip(pnl, -1.0, 1.0))

        # 2. position_fraction  [0, 1]
        pos_frac = self._get_position_fraction()

        # 3. steps_in_position normalised (ref = 14 steps = 7 days at 12H) [0, 3+]
        steps_norm = float(np.clip(self.steps_in_position / 14.0, 0.0, 3.0))

        # 4. drawdown_pct from session peak  [-1, 0]
        if self.portfolio_history:
            peak = max(self.portfolio_history)
        else:
            peak = self.initial_capital
        dd = (self.portfolio_value - peak) / max(peak, 1e-8)
        dd = float(np.clip(dd, -1.0, 0.0))

        return np.array([pnl, pos_frac, steps_norm, dd], dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        """Market observation + position state, concatenated."""
        market_obs = super()._get_observation()          # (lookback × n_features,)
        pos_state  = self._get_position_state()           # (N_POS_FEATURES,)
        return np.concatenate([market_obs, pos_state])

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, action) -> float:
        """
        Calls `professional_risk_adjusted` with the full regime context from
        the current environment state.  The context parameters are read from
        self.df.iloc[current_step] — so atr_14 and vol_ratio MUST be in the
        `features.select` list of the YAML config to be present in self.df.
        """
        row   = self.df.iloc[self.current_step]
        price = self._get_price()

        if "atr_14" not in row.index or pd.isna(row["atr_14"]):
            logger.warning(
                "atr_14 column missing or NaN — falling back to ATR_REFERENCE (%.4f). "
                "Add 'atr_14' to features.select in the training YAML to avoid "
                "miscalibrated stop losses.",
                ATR_REFERENCE,
            )
        atr_14    = _safe_col(row, "atr_14",    price * ATR_REFERENCE)
        atr_pct   = atr_14 / max(price, 1e-8)
        vol_ratio = _safe_col(row, "vol_ratio", 1.0)

        peak = max(self.portfolio_history) if self.portfolio_history else self.initial_capital
        drawdown_pct = (self.portfolio_value - peak) / max(peak, 1e-8)

        return self.reward_fn(
            prev_value        = self.prev_portfolio_value,
            curr_value        = self.portfolio_value,
            action            = action,
            in_position       = self.in_position,
            history           = self.portfolio_history,
            scaling           = self.reward_scaling,
            atr_pct           = float(np.clip(atr_pct, 1e-4, 1.0)),
            vol_ratio         = float(np.clip(vol_ratio, 0.1, 10.0)),
            drawdown_pct      = float(np.clip(drawdown_pct, -1.0, 0.0)),
            steps_since_trade = self.steps_since_last_trade,
            position_fraction = self._get_position_fraction(),
            trade_happened    = self._trade_this_step,
        )

    # ── Trade execution ───────────────────────────────────────────────────────

    def _execute_rebalance(self, target_fraction: float, price: float) -> None:
        """
        Rebalances portfolio so that `target_fraction` of its total value is in BTC.
        Deducts fee proportional to the fraction CHANGED, not the total value.
        Updates all position tracking state.
        """
        current_fraction = self._get_position_fraction()
        delta = abs(target_fraction - current_fraction)

        if delta < 1e-4:
            return   # nothing to do

        total_value = self.usdt_balance + self.btc_balance * price

        # Fee on the rebalanced portion only
        fee = delta * self.fee_rate * total_value
        total_value = max(total_value - fee, 0.0)

        self.btc_balance  = (total_value * target_fraction) / max(price, 1e-8)
        self.usdt_balance = total_value * (1.0 - target_fraction)
        self.trades      += 1
        self._trade_this_step     = True
        self.steps_since_last_trade = 0

        # Track entry state when moving from flat → position
        if current_fraction < 0.05 and target_fraction >= 0.05:
            self.entry_price = price
            row = self.df.iloc[self.current_step]
            self.entry_atr  = _safe_col(row, "atr_14", price * ATR_REFERENCE)
            self.stop_price = price * (
                1.0 - self.stop_atr_multiplier * self.entry_atr / max(price, 1e-8)
            )
            self.steps_in_position = 0

        # Clear entry state when closing fully
        if target_fraction < 0.05:
            self.entry_price = 0.0
            self.entry_atr   = 0.0
            self.stop_price  = 0.0

    def _check_and_apply_stop_loss(self, price: float) -> bool:
        """
        Checks if the ATR stop has been triggered.
        If so, forces SELL_FULL and returns True.
        """
        if self.in_position and self.stop_price > 0 and price <= self.stop_price:
            self._execute_rebalance(0.0, price)
            return True
        return False

    def _update_position_counters(self) -> None:
        """Updates per-step position duration counters."""
        pos_frac = self._get_position_fraction()
        self.in_position = pos_frac > 0.05

        if self.in_position:
            self.steps_in_position += 1
        else:
            self.steps_in_position = 0

        if not self._trade_this_step:
            self.steps_since_last_trade += 1

        # Track session peak for drawdown computation
        self.peak_portfolio = max(self.peak_portfolio, self.portfolio_value)

    def _build_step_return(self, action, done: bool):
        """Builds the (obs, reward, done, truncated, info) tuple."""
        reward = self._compute_reward(action)
        self.prev_portfolio_value = self.portfolio_value
        self.portfolio_history.append(self.portfolio_value)

        self.current_step += 1
        done = done or (self.current_step >= len(self.df) - 1)

        obs = self._get_observation() if not done else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        info = {
            "portfolio_value":   self.portfolio_value,
            "price":             self._get_price() if not done else 0.0,
            "trades":            self.trades,
            "position_fraction": self._get_position_fraction(),
            "steps_in_position": self.steps_in_position,
        }
        return obs, reward, done, False, info




# ── Discrete (5-action) variant — PPO ────────────────────────────────────────

class BtcTradingEnvProfessionalDiscrete(_ProfessionalBase):
    """
    Action space: Discrete(5)
      0  HOLD           — no change
      1  BUY_FULL       — target fraction → 1.0 (full position)
      2  BUY_PARTIAL    — target fraction → 0.5 (half position)
      3  SELL_PARTIAL   — halve current exposure (e.g. 1.0 → 0.5, 0.5 → 0.0)
      4  SELL_FULL      — target fraction → 0.0 (flat)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Discrete(5)

    def step(self, action: int):
        price = self._get_price()
        self._trade_this_step = False

        # ── ATR stop loss check (overrides agent action) ──────────────────
        self._check_and_apply_stop_loss(price)

        # ── Execute agent action (only if stop didn't already close us) ───
        if not self._trade_this_step:
            current_frac = self._get_position_fraction()

            if action == 1:   # BUY_FULL
                self._execute_rebalance(1.0, price)

            elif action == 2:   # BUY_PARTIAL
                # Idempotent: if already at 0.5+, no additional buy
                if current_frac < 0.45:
                    self._execute_rebalance(0.5, price)

            elif action == 3:   # SELL_PARTIAL
                # Halve current exposure; floor at 0 if already partial
                new_frac = max(0.0, current_frac - 0.5)
                if current_frac > 0.05:
                    self._execute_rebalance(new_frac, price)

            elif action == 4:   # SELL_FULL
                if current_frac > 0.05:
                    self._execute_rebalance(0.0, price)

            # action == 0: HOLD — no rebalance

        # ── Post-step bookkeeping ──────────────────────────────────────────
        self.portfolio_value = self._calculate_portfolio_value()
        self._update_position_counters()

        return self._build_step_return(action, done=False)


# ── Continuous variant — SAC ──────────────────────────────────────────────────

class BtcTradingEnvProfessionalContinuous(_ProfessionalBase):
    """
    Action space: Box([0, 1])
      The single action value represents the TARGET fraction of portfolio in BTC.
      0.0 = fully flat.  1.0 = fully long.  0.5 = half position.

    Deadband ±DEADBAND: if |target - current| < DEADBAND, no rebalance executes.
    This prevents micro-trading from eroding returns via continuous fee payments.
    """


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def _reset_state(self):
        super()._reset_state()
        self.position = 0.0   # mirrors position_fraction for SAC internal state

    def step(self, action: np.ndarray):
        price = self._get_price()
        self._trade_this_step = False

        target_fraction = float(np.clip(action[0], 0.0, 1.0))
        current_fraction = self._get_position_fraction()

        # ── ATR stop loss (takes precedence over agent action) ────────────
        self._check_and_apply_stop_loss(price)

        # ── Apply deadband and execute rebalance ──────────────────────────
        if not self._trade_this_step:
            if abs(target_fraction - current_fraction) >= DEADBAND:
                self._execute_rebalance(target_fraction, price)

        self.position = self._get_position_fraction()

        # ── Post-step bookkeeping ─────────────────────────────────────────
        self.portfolio_value = self._calculate_portfolio_value()
        self._update_position_counters()

        return self._build_step_return(target_fraction, done=False)
