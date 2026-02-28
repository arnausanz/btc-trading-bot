# bots/rl/environment.py
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from bots.rl.rewards import builtins  # importa per registrar les builtins
from bots.rl.rewards.registry import get as get_reward


class BtcTradingEnv(gym.Env):
    """
    Environment Gymnasium per a trading de BTC.
    La reward function és configurable via nom — qualsevol funció
    registrada al RewardRegistry pot ser usada sense canviar codi.
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

        self.action_space = spaces.Discrete(3)
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
        col_std = obs.std(axis=0)
        col_std[col_std == 0] = 1.0
        obs = (obs - obs.mean(axis=0)) / col_std
        return obs.flatten()

    def _get_price(self) -> float:
        return float(self.df.iloc[self.current_step]["close"])

    def _calculate_portfolio_value(self) -> float:
        return self.usdt_balance + self.btc_balance * self._get_price()

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

        reward = self.reward_fn(
            prev_value=self.prev_portfolio_value,
            curr_value=self.portfolio_value,
            action=action,
            in_position=self.in_position,
            history=self.portfolio_history,
            scaling=self.reward_scaling,
        )

        self.prev_portfolio_value = self.portfolio_value
        self.portfolio_history.append(self.portfolio_value)
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        obs = self._get_observation() if not done else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )

        return obs, reward, done, False, {
            "portfolio_value": self.portfolio_value,
            "price": price,
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