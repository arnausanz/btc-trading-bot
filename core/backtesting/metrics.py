# core/backtesting/metrics.py
import numpy as np
import pandas as pd


class BacktestMetrics:
    """
    Calcula mètriques de rendiment a partir de l'historial d'un backtest.
    Totes les mètriques són comparables entre bots si s'executen
    sobre el mateix període i capital inicial.
    """

    def __init__(self, history: list[dict], initial_capital: float):
        self.df = pd.DataFrame(history)
        self.initial_capital = initial_capital

    def total_return(self) -> float:
        """Retorn total en percentatge."""
        final = self.df["portfolio_value"].iloc[-1]
        return (final - self.initial_capital) / self.initial_capital * 100

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Sharpe Ratio anualitzat. Mesura retorn ajustat per risc.
        >1 és acceptable, >2 és bo, >3 és excel·lent.
        """
        returns = self.df["portfolio_value"].pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        excess = returns - risk_free_rate / 365
        return float(np.sqrt(365) * excess.mean() / excess.std())

    def max_drawdown(self) -> float:
        """
        Màxima caiguda des d'un màxim fins a un mínim posterior.
        En percentatge. Com més petit (negatiu) pitjor.
        """
        values = self.df["portfolio_value"]
        peak = values.cummax()
        drawdown = (values - peak) / peak * 100
        return float(drawdown.min())

    def calmar_ratio(self) -> float:
        """
        Retorn anualitzat dividit pel max drawdown.
        Penalitza estratègies amb drawdowns grans.
        """
        dd = abs(self.max_drawdown())
        if dd == 0:
            return 0.0
        days = len(self.df)
        annualized_return = self.total_return() * (365 / days)
        return annualized_return / dd

    def win_rate(self) -> float:
        """
        Percentatge de trades tancats amb benefici.
        Només compta ordres filled.
        """
        filled = self.df[self.df["order_status"].astype(str).str.contains("filled")]
        if len(filled) == 0:
            return 0.0
        returns = filled["portfolio_value"].pct_change().dropna()
        wins = (returns > 0).sum()
        return float(wins / len(returns) * 100)

    def summary(self) -> dict:
        """Retorna totes les mètriques en un sol diccionari."""
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