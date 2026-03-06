# bots/rl/environment.py
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from bots.rl.rewards import builtins  # importa per registrar les builtins
from bots.rl.rewards.registry import get as get_reward


class _BaseTradingEnv(gym.Env):
    """
    Lògica comuna als dos entorns.
    No s'instancia directament — usar BtcTradingEnvDiscrete o BtcTradingEnvContinuous.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int = 50,
        initial_capital: float = 10_000.0,
        fee_rate: float = 0.001,
        reward_scaling: float = 100.0,
        reward_type: str = "simple",
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.lookback = lookback
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.reward_scaling = reward_scaling
        self.reward_fn = get_reward(reward_type)
        self.n_features = len(df.columns)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback * self.n_features,),
            dtype=np.float32,
        )
        self._reset_state()

    def _reset_state(self):
        self.current_step = self.lookback
        self.usdt_balance = self.initial_capital
        self.btc_balance = 0.0
        self.portfolio_value = self.initial_capital
        self.prev_portfolio_value = self.initial_capital
        self.in_position = False
        self.trades = 0
        self.portfolio_history = [self.initial_capital]

    def _get_observation(self) -> np.ndarray:
        window = self.df.iloc[self.current_step - self.lookback:self.current_step]
        obs = window.values.astype(np.float32)

        # Normalització robusta per evitar overflow i NaN
        col_mean = obs.mean(axis=0)
        col_std = obs.std(axis=0)
        col_std = np.where(col_std < 1e-8, 1.0, col_std)  # evita divisió per zero
        obs = (obs - col_mean) / col_std

        # Clip agressiu per evitar valors extrems que cauen fora de float32
        obs = np.clip(obs, -10.0, 10.0)

        # Verificació de seguretat
        if not np.isfinite(obs).all():
            obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

        return obs.flatten()

    def _get_price(self) -> float:
        return float(self.df.iloc[self.current_step]["close"])

    def _calculate_portfolio_value(self) -> float:
        return self.usdt_balance + self.btc_balance * self._get_price()

    def _compute_reward(self, action) -> float:
        return self.reward_fn(
            prev_value=self.prev_portfolio_value,
            curr_value=self.portfolio_value,
            action=action,
            in_position=self.in_position,
            history=self.portfolio_history,
            scaling=self.reward_scaling,
        )

    def _step_common(self, done: bool) -> tuple:
        obs = self._get_observation() if not done else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        return obs, done, {
            "portfolio_value": self.portfolio_value,
            "price": self._get_price() if not done else 0.0,
            "trades": self.trades,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}

    def render(self):
        print(
            f"Step {self.current_step} | "
            f"Price: {self._get_price():.0f} | "
            f"Portfolio: {self.portfolio_value:.2f} | "
            f"Trades: {self.trades}"
        )


class BtcTradingEnvDiscrete(_BaseTradingEnv):
    """
    Entorn amb accions discretes: 0=HOLD, 1=BUY, 2=SELL.
    Compatible amb PPO i DQN.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Discrete(3)

    def step(self, action: int):
        price = self._get_price()

        if action == 1 and not self.in_position:
            self.btc_balance = (self.usdt_balance * (1 - self.fee_rate)) / price
            self.usdt_balance = 0.0
            self.in_position = True
            self.trades += 1

        elif action == 2 and self.in_position:
            self.usdt_balance = self.btc_balance * price * (1 - self.fee_rate)
            self.btc_balance = 0.0
            self.in_position = False
            self.trades += 1

        self.portfolio_value = self._calculate_portfolio_value()
        reward = self._compute_reward(action)
        self.prev_portfolio_value = self.portfolio_value
        self.portfolio_history.append(self.portfolio_value)
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        obs, done, info = self._step_common(done)
        return obs, reward, done, False, info


class BtcTradingEnvContinuous(_BaseTradingEnv):
    """
    Entorn amb accions contínues: [0, 1] → fracció del capital en BTC.
    0 = 100% USDT (flat), 1 = 100% BTC (full long).

    Nota: sense shorts per ara — SAC aprèn long/flat primer.
    Shorts es poden afegir més endavant amb marge simulat.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.position = 0.0

    def _reset_state(self):
        super()._reset_state()
        self.position = 0.0

    def step(self, action: np.ndarray):
        price = self._get_price()

        # Posició target: fracció del portfolio en BTC [0, 1]
        target_position = float(np.clip(action[0], 0.0, 1.0))

        # Valor total actual
        total_value = self.usdt_balance + self.btc_balance * price

        # Fee proporcional al canvi de posició
        position_delta = abs(target_position - self.position)
        fee = position_delta * self.fee_rate * total_value

        # Aplica fee
        total_value = max(total_value - fee, 0.0)

        # Rebalancea
        self.btc_balance = (total_value * target_position) / price if price > 0 else 0.0
        self.usdt_balance = total_value * (1 - target_position)

        if position_delta > 0.01:
            self.trades += 1

        self.position = target_position
        self.in_position = target_position > 0.1
        self.portfolio_value = max(
            self.usdt_balance + self.btc_balance * price, 0.0
        )

        reward = self._compute_reward(target_position)
        self.prev_portfolio_value = self.portfolio_value
        self.portfolio_history.append(self.portfolio_value)
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        obs, done, info = self._step_common(done)
        return obs, reward, done, False, info