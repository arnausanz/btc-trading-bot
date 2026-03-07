# core/backtesting/metrics.py
import numpy as np
import pandas as pd

# Annualization factors per timeframe (periods per year)
# Crypto operates 24/7/365 — no market days used
_PERIODS_PER_YEAR: dict[str, int] = {
    "1m":  365 * 24 * 60,   # 525.600
    "5m":  365 * 24 * 12,   # 105.120
    "15m": 365 * 24 * 4,    #  52.560
    "30m": 365 * 24 * 2,    #  26.280
    "1h":  365 * 24,        #   8.760
    "2h":  365 * 12,        #   4.380
    "4h":  365 * 6,         #   2.190
    "6h":  365 * 4,         #   1.460
    "8h":  365 * 3,         #   1.095
    "12h": 365 * 2,         #     730
    "1d":  365,             #     365
    "3d":  121,
    "1w":  52,
}


class BacktestMetrics:
    """
    Computes performance metrics from a backtest history.
    All metrics are comparable across bots if they run
    on the same period, initial capital, and timeframe.

    The timeframe is required to correctly annualize Sharpe and Calmar ratios.
    """

    def __init__(self, history: list[dict], initial_capital: float, timeframe: str = "1h"):
        self.df = pd.DataFrame(history)
        self.initial_capital = initial_capital
        self.timeframe = timeframe
        self._periods_per_year = _PERIODS_PER_YEAR.get(timeframe, 365 * 24)

    def total_return(self) -> float:
        """Total return in percentage."""
        final = self.df["portfolio_value"].iloc[-1]
        return (final - self.initial_capital) / self.initial_capital * 100

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Sharpe Ratio correctly annualized for the given timeframe.
        Uses sqrt(periods_per_year) to annualize periodic returns.

        For hourly data (1h): sqrt(8.760)
        For daily data  (1d): sqrt(365)

        >1 is acceptable, >2 is good, >3 is excellent.
        """
        returns = self.df["portfolio_value"].pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        rf_per_period = risk_free_rate / self._periods_per_year
        excess = returns - rf_per_period
        return float(np.sqrt(self._periods_per_year) * excess.mean() / excess.std())

    def max_drawdown(self) -> float:
        """
        Maximum drawdown from a peak to a subsequent trough.
        In percentage. The smaller (more negative) the worse.
        """
        values = self.df["portfolio_value"]
        peak = values.cummax()
        drawdown = (values - peak) / peak * 100
        return float(drawdown.min())

    def _duration_days(self) -> float:
        """
        Actual duration of the backtest in days.
        Uses timestamps from history if available.
        Fallback: estimates days from number of ticks and timeframe.
        """
        if "timestamp" in self.df.columns and len(self.df) >= 2:
            try:
                t0 = pd.Timestamp(self.df["timestamp"].iloc[0])
                t1 = pd.Timestamp(self.df["timestamp"].iloc[-1])
                days = (t1 - t0).total_seconds() / 86400
                if days > 0:
                    return days
            except Exception:
                pass
        # Fallback: ticks / (periods per day)
        periods_per_day = self._periods_per_year / 365
        return len(self.df) / periods_per_day

    def calmar_ratio(self) -> float:
        """
        Annualized return divided by max drawdown.
        Penalizes strategies with large drawdowns.
        """
        dd = abs(self.max_drawdown())
        if dd == 0:
            return 0.0
        duration_days = self._duration_days()
        if duration_days == 0:
            return 0.0
        annualized_return = self.total_return() * (365 / duration_days)
        return annualized_return / dd

    def win_rate(self) -> float:
        """
        Actual Win Rate: percentage of closed round-trips (buy→sell) with profit.
        Does not count open positions without closure.
        Returns 0.0 if no signals or no closed trades exist.
        """
        if "signal" not in self.df.columns:
            return 0.0

        trades_won = 0
        trades_total = 0
        entry_value = None

        for _, row in self.df.iterrows():
            signal_str = str(row.get("signal", "")).lower()
            portfolio = row["portfolio_value"]

            if "buy" in signal_str and entry_value is None:
                entry_value = portfolio
            elif "sell" in signal_str and entry_value is not None:
                trades_total += 1
                if portfolio > entry_value:
                    trades_won += 1
                entry_value = None

        if trades_total == 0:
            return 0.0
        return float(trades_won / trades_total * 100)

    def summary(self) -> dict:
        """Returns all metrics in a single dictionary."""
        return {
            "total_return_pct": round(self.total_return(), 2),
            "sharpe_ratio": round(self.sharpe_ratio(), 3),
            "max_drawdown_pct": round(self.max_drawdown(), 2),
            "calmar_ratio": round(self.calmar_ratio(), 3),
            "win_rate_pct": round(self.win_rate(), 2),
            "initial_capital": self.initial_capital,
            "final_capital": round(self.df["portfolio_value"].iloc[-1], 2),
            "total_ticks": len(self.df),
        }
