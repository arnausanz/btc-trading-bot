# core/backtesting/metrics.py
import numpy as np
import pandas as pd

# Factors d'anualització per timeframe (períodes per any)
# Crypto opera 24/7/365 — no s'usen dies de mercat
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
    Calcula mètriques de rendiment a partir de l'historial d'un backtest.
    Totes les mètriques són comparables entre bots si s'executen
    sobre el mateix període, capital inicial i timeframe.

    El timeframe és obligatori per anualitzar correctament el Sharpe i el Calmar.
    """

    def __init__(self, history: list[dict], initial_capital: float, timeframe: str = "1h"):
        self.df = pd.DataFrame(history)
        self.initial_capital = initial_capital
        self.timeframe = timeframe
        self._periods_per_year = _PERIODS_PER_YEAR.get(timeframe, 365 * 24)

    def total_return(self) -> float:
        """Retorn total en percentatge."""
        final = self.df["portfolio_value"].iloc[-1]
        return (final - self.initial_capital) / self.initial_capital * 100

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Sharpe Ratio anualitzat correctament per al timeframe donat.
        Usa sqrt(periods_per_year) per anualitzar els retorns periòdics.

        Per a dades horàries (1h): sqrt(8.760)
        Per a dades diàries  (1d): sqrt(365)

        >1 és acceptable, >2 és bo, >3 és excel·lent.
        """
        returns = self.df["portfolio_value"].pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        rf_per_period = risk_free_rate / self._periods_per_year
        excess = returns - rf_per_period
        return float(np.sqrt(self._periods_per_year) * excess.mean() / excess.std())

    def max_drawdown(self) -> float:
        """
        Màxima caiguda des d'un màxim fins a un mínim posterior.
        En percentatge. Com més petit (negatiu) pitjor.
        """
        values = self.df["portfolio_value"]
        peak = values.cummax()
        drawdown = (values - peak) / peak * 100
        return float(drawdown.min())

    def _duration_days(self) -> float:
        """
        Durada real del backtest en dies.
        Usa timestamps del historial si estan disponibles.
        Fallback: estima dies a partir del nombre de ticks i el timeframe.
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
        # Fallback: ticks / (períodes per dia)
        periods_per_day = self._periods_per_year / 365
        return len(self.df) / periods_per_day

    def calmar_ratio(self) -> float:
        """
        Retorn anualitzat dividit pel max drawdown.
        Penalitza estratègies amb drawdowns grans.
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
        Win Rate real: percentatge de round-trips tancats (buy→sell) amb profit.
        No compta posicions obertes sense tancar.
        Retorna 0.0 si no hi ha senyals o no hi ha cap trade tancat.
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
