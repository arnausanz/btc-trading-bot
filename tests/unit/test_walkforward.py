# tests/unit/test_walkforward.py
"""
Tests unitaris del walk-forward backtesting.
Comproven que la separació train/test funciona correctament
sense necessitat de connexió a la DB (tot mockejat).

Executa'ls amb: python3 -m pytest tests/unit/test_walkforward.py -v
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from core.engine.runner import _to_utc_timestamp, Runner
from core.interfaces.base_bot import ObservationSchema
from core.config import TRAIN_UNTIL, TEST_FROM


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_df(n: int = 500, start: str = "2022-01-01", freq: str = "h") -> pd.DataFrame:
    """DataFrame sintètic de candeles amb index UTC. Seed fix = determinista."""
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(42)
    close = 30_000 + np.cumsum(rng.normal(0, 100, n))
    return pd.DataFrame({"close": close, "volume": 1.0}, index=idx)


def make_mock_bot(lookback: int = 10) -> MagicMock:
    """Bot mock que retorna HOLD a cada tick."""
    schema = ObservationSchema(
        features=["close"],
        timeframes=["1h"],
        lookback=lookback,
    )
    signal = MagicMock()
    signal.action = "hold"

    bot = MagicMock()
    bot.observation_schema.return_value = schema
    bot.on_observation.return_value = signal
    return bot


def make_mock_exchange() -> MagicMock:
    """Exchange mock amb capital fix."""
    order = MagicMock()
    order.status = "filled"
    exchange = MagicMock()
    exchange.get_portfolio.return_value = {"USDT": 10_000.0}
    exchange.get_portfolio_value.return_value = 10_000.0
    exchange.send_order.return_value = order
    return exchange


def make_runner_with_df(df: pd.DataFrame, lookback: int = 10):
    """Retorna (runner, bot, exchange) amb ObservationBuilder mockejat."""
    bot = make_mock_bot(lookback=lookback)
    exchange = make_mock_exchange()
    runner = Runner(bot=bot, exchange=exchange)

    # Substituïm el builder per un mock que retorna el df sintètic
    runner.builder = MagicMock()
    runner.builder.get_dataframe.return_value = df
    runner.builder.build.return_value = {"1h": {"features": df.iloc[0:lookback], "current_price": 30_000.0, "timestamp": df.index[0]}}
    return runner, bot, exchange


# ── Tests _to_utc_timestamp ────────────────────────────────────────────────────

class TestToUtcTimestamp:
    def test_string_date_tz_naive_gets_utc(self):
        ts = _to_utc_timestamp("2025-01-01")
        assert ts.tzinfo is not None
        assert ts.year == 2025
        assert ts.month == 1
        assert ts.day == 1

    def test_string_with_tz_keeps_tz(self):
        ts = _to_utc_timestamp("2025-06-15T12:00:00+00:00")
        assert ts.year == 2025
        assert ts.hour == 12

    def test_end_of_year(self):
        ts = _to_utc_timestamp("2024-12-31")
        assert ts.year == 2024
        assert ts.month == 12
        assert ts.day == 31


# ── Tests Runner amb dates ─────────────────────────────────────────────────────

class TestRunnerDateFiltering:
    """
    Verifica que Runner filtra correctament els ticks per data.
    El DataFrame és de 500 hores des de 2022-01-01 (≈ 20 dies).
    Lookback = 10.
    """

    LOOKBACK = 10

    def test_no_dates_returns_full_history(self):
        """Sense dates, s'iteraran tots els ticks disponibles (n - lookback)."""
        df = make_df(n=200)
        runner, _, _ = make_runner_with_df(df, lookback=self.LOOKBACK)

        history = runner.run(symbol="BTC/USDT", timeframe="1h")

        expected_ticks = len(df) - self.LOOKBACK
        assert len(history) == expected_ticks

    def test_end_date_reduces_history(self):
        """end_date a la meitat del dataset → history ~meitat llarga."""
        df = make_df(n=500)  # 500 hores des de 2022-01-01
        runner, _, _ = make_runner_with_df(df, lookback=self.LOOKBACK)

        # Posem end_date a ~250 hores del inici
        midpoint_ts = df.index[250]
        end_date = midpoint_ts.strftime("%Y-%m-%dT%H:%M:%S")

        history = runner.run(symbol="BTC/USDT", timeframe="1h", end_date=end_date)

        # Ha d'haver processsat fins al punt de tall (~250 - lookback ticks)
        assert len(history) > 0
        assert len(history) < len(df) - self.LOOKBACK
        # L'últim timestamp ha de ser <= end_date
        last_ts = pd.Timestamp(history[-1]["timestamp"])
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        assert last_ts <= _to_utc_timestamp(end_date)

    def test_start_date_reduces_history(self):
        """start_date a ~250 del df → history és ~250 ticks (menys el lookback)."""
        df = make_df(n=500)
        runner, _, _ = make_runner_with_df(df, lookback=self.LOOKBACK)

        # start_date apunta a ~250 hores del inici
        mid_ts = df.index[250]
        start_date = mid_ts.strftime("%Y-%m-%dT%H:%M:%S")

        history = runner.run(symbol="BTC/USDT", timeframe="1h", start_date=start_date)

        assert len(history) > 0
        assert len(history) < len(df) - self.LOOKBACK
        # El primer timestamp ha de ser >= start_date
        first_ts = pd.Timestamp(history[0]["timestamp"])
        if first_ts.tzinfo is None:
            first_ts = first_ts.tz_localize("UTC")
        assert first_ts >= _to_utc_timestamp(start_date)

    def test_start_and_end_date_define_window(self):
        """Amb start i end, la finestra és exactament entre les dues dates."""
        df = make_df(n=500)
        runner, _, _ = make_runner_with_df(df, lookback=self.LOOKBACK)

        start_ts = df.index[150]
        end_ts = df.index[350]
        start_date = start_ts.strftime("%Y-%m-%dT%H:%M:%S")
        end_date = end_ts.strftime("%Y-%m-%dT%H:%M:%S")

        history = runner.run(symbol="BTC/USDT", timeframe="1h", start_date=start_date, end_date=end_date)

        assert len(history) > 0
        # Tots els timestamps han d'estar dins la finestra
        for tick in history:
            ts = pd.Timestamp(tick["timestamp"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            assert ts >= _to_utc_timestamp(start_date)
            assert ts <= _to_utc_timestamp(end_date)

    def test_end_date_before_data_returns_empty(self):
        """end_date anterior a totes les dades → retorna llista buida."""
        df = make_df(n=100, start="2024-01-01")
        runner, _, _ = make_runner_with_df(df, lookback=self.LOOKBACK)

        history = runner.run(symbol="BTC/USDT", timeframe="1h", end_date="2020-01-01")
        assert history == []

    def test_start_date_after_data_returns_empty(self):
        """start_date posterior a totes les dades → retorna llista buida."""
        df = make_df(n=100, start="2022-01-01")
        runner, _, _ = make_runner_with_df(df, lookback=self.LOOKBACK)

        history = runner.run(symbol="BTC/USDT", timeframe="1h", start_date="2030-01-01")
        assert history == []

    def test_lookback_preserved_with_start_date(self):
        """
        El lookback s'ha de poder satisfer fins i tot quan start_date apunta
        a una posició propera a l'inici del df.
        Aquí start_date apunta a la posició 5 (< lookback=10) → iter_start
        ha de recaure a 'schema.lookback' (10).
        """
        df = make_df(n=200)
        runner, _, _ = make_runner_with_df(df, lookback=self.LOOKBACK)

        # start_date molt al principi del df (posició ~5)
        early_ts = df.index[5]
        start_date = early_ts.strftime("%Y-%m-%dT%H:%M:%S")

        history = runner.run(symbol="BTC/USDT", timeframe="1h", start_date=start_date)
        # Ha de funcionar sense IndexError — si arriba aquí el test passa
        assert len(history) > 0

    def test_train_test_split_totals(self):
        """
        Train (end_date) + Test (start_date del tick següent) ≈ Total (sense dates).
        No han de solapar-se.
        """
        df = make_df(n=300)
        lookback = self.LOOKBACK

        # Fem servir la posició 150 com a tall
        split_idx = 150
        split_ts = df.index[split_idx]
        # end_date del train = split_ts
        # start_date del test = split_ts + 1 hora
        train_end = split_ts.strftime("%Y-%m-%dT%H:%M:%S")
        test_start = (split_ts + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")

        runner_train, _, _ = make_runner_with_df(df, lookback=lookback)
        runner_test, _, _ = make_runner_with_df(df, lookback=lookback)

        history_train = runner_train.run("BTC/USDT", "1h", end_date=train_end)
        history_test = runner_test.run("BTC/USDT", "1h", start_date=test_start)

        # Cap timestamp de train pot aparèixer al test
        train_timestamps = {tick["timestamp"] for tick in history_train}
        test_timestamps = {tick["timestamp"] for tick in history_test}
        overlap = train_timestamps & test_timestamps
        assert len(overlap) == 0, f"Solapament de {len(overlap)} ticks entre train i test!"


# ── Tests DatasetBuilder amb train_until ──────────────────────────────────────

class TestDatasetBuilderTrainUntil:
    """
    Verifica que DatasetBuilder filtra correctament les dades al período de train.
    Mockeja compute_features per no necessitar DB.
    """

    def _make_full_df(self, n: int = 1000, start: str = "2019-01-01") -> pd.DataFrame:
        """DataFrame de features sintètic."""
        idx = pd.date_range(start=start, periods=n, freq="h", tz="UTC")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "close":        30_000 + np.cumsum(rng.normal(0, 100, n)),
            "open":         30_000 + np.cumsum(rng.normal(0, 50, n)),
            "high":         31_000 + np.cumsum(rng.normal(0, 100, n)),
            "low":          29_000 + np.cumsum(rng.normal(0, 100, n)),
            "volume":       rng.uniform(1, 100, n),
            "rsi_14":       rng.uniform(30, 70, n),
            "ema_20":       30_000 + np.cumsum(rng.normal(0, 50, n)),
            "macd":         rng.normal(0, 50, n),
            "macd_signal":  rng.normal(0, 50, n),
            "macd_hist":    rng.normal(0, 20, n),
            "bb_middle_20": 30_000 + np.cumsum(rng.normal(0, 80, n)),
            "bb_upper_20":  31_000 + np.cumsum(rng.normal(0, 80, n)),
            "bb_lower_20":  29_000 + np.cumsum(rng.normal(0, 80, n)),
            "atr_14":       rng.uniform(50, 500, n),
            "ema_9":        30_000 + np.cumsum(rng.normal(0, 40, n)),
            "ema_50":       30_000 + np.cumsum(rng.normal(0, 30, n)),
            "ema_200":      30_000 + np.cumsum(rng.normal(0, 20, n)),
        }, index=idx)
        return df

    def test_train_until_filters_rows(self):
        """X ha de contenir únicament files <= train_until."""
        from data.processing.dataset import DatasetBuilder

        full_df = self._make_full_df(n=1000, start="2019-01-01")
        train_until = "2019-02-28"  # ~59 dies * 24 = ~1416 hores, però tenim 1000

        with patch("data.processing.dataset.compute_features", return_value=full_df):
            builder = DatasetBuilder(
                symbol="BTC/USDT",
                timeframes=["1h"],
                forward_window=24,
                threshold_pct=0.005,
                train_until=train_until,
            )
            X, y = builder.build()

        cutoff = pd.Timestamp(train_until, tz="UTC")
        assert (X.index <= cutoff).all(), "Hi ha files posteriors a train_until!"

    def test_train_until_reduces_dataset(self):
        """Amb train_until, el dataset ha de ser més petit que sense."""
        from data.processing.dataset import DatasetBuilder

        full_df = self._make_full_df(n=1000)
        train_until = "2019-01-21"  # ~480 hores (~meitat dels primers 1000h)

        with patch("data.processing.dataset.compute_features", return_value=full_df):
            builder_full = DatasetBuilder(
                symbol="BTC/USDT", timeframes=["1h"],
                forward_window=24, threshold_pct=0.005, train_until=None,
            )
            builder_train = DatasetBuilder(
                symbol="BTC/USDT", timeframes=["1h"],
                forward_window=24, threshold_pct=0.005, train_until=train_until,
            )
            X_full, _ = builder_full.build()
            X_train, _ = builder_train.build()

        assert len(X_train) < len(X_full)

    def test_no_train_until_uses_all_data(self):
        """Sense train_until (None), s'usa tot el dataset."""
        from data.processing.dataset import DatasetBuilder

        full_df = self._make_full_df(n=200)

        with patch("data.processing.dataset.compute_features", return_value=full_df):
            builder = DatasetBuilder(
                symbol="BTC/USDT", timeframes=["1h"],
                forward_window=24, threshold_pct=0.005, train_until=None,
            )
            X, y = builder.build()

        # Amb forward_window=24, l'últim índex del df és el que tindríem sense filtrar
        assert len(X) == len(full_df) - 24

    def test_from_config_reads_train_until(self):
        """from_config() ha de llegir train_until de config['data']['train_until']."""
        from data.processing.dataset import DatasetBuilder

        config = {
            "data": {
                "symbol": "BTC/USDT",
                "timeframes": ["1h"],
                "forward_window": 24,
                "threshold_pct": 0.005,
                "train_until": "2023-12-31",
            }
        }
        builder = DatasetBuilder.from_config(config)
        assert builder.train_until == "2023-12-31"

    def test_from_config_fallback_to_global(self):
        """Si no hi ha train_until a config, s'usa el TRAIN_UNTIL global."""
        from data.processing.dataset import DatasetBuilder

        config = {
            "data": {
                "symbol": "BTC/USDT",
                "timeframes": ["1h"],
                "forward_window": 24,
                "threshold_pct": 0.005,
                # sense train_until
            }
        }
        builder = DatasetBuilder.from_config(config)
        assert builder.train_until == TRAIN_UNTIL


# ── Tests core/config.py ──────────────────────────────────────────────────────

class TestCoreConfig:
    def test_train_until_is_string(self):
        assert isinstance(TRAIN_UNTIL, str)
        assert len(TRAIN_UNTIL) == 10  # format YYYY-MM-DD

    def test_test_from_is_string(self):
        assert isinstance(TEST_FROM, str)
        assert len(TEST_FROM) == 10

    def test_train_until_before_test_from(self):
        """El periode de train ha d'acabar ABANS que comenci el de test."""
        assert pd.Timestamp(TRAIN_UNTIL) < pd.Timestamp(TEST_FROM)

    def test_split_is_contiguous(self):
        """train_until i test_from han de ser dies consecutius."""
        train_end = pd.Timestamp(TRAIN_UNTIL)
        test_start = pd.Timestamp(TEST_FROM)
        gap = (test_start - train_end).days
        assert gap == 1, f"Gap entre train i test és {gap} dies (hauria de ser 1)"
