# data/processing/external.py
"""
Loaders for external feature sources.
Each function returns a UTC-indexed DataFrame ready for merge_asof alignment.

Column naming convention:
  fear_greed   → fear_greed_value (0-100), fear_greed_class (0-4 ordinal)
  funding_rate → funding_rate
  open_interest (timeframe=1h) → oi_btc_1h, oi_usdt_1h
  open_interest (timeframe=5m) → oi_btc_5m, oi_usdt_5m
  blockchain metric "hash-rate" → hash_rate
  blockchain metric "n-unique-addresses" → n_unique_addresses
  blockchain metric "transaction-fees" → transaction_fees
"""
import logging
import pandas as pd
from core.db.session import SessionLocal
from core.db.models import FearGreedDB, FundingRateDB, OpenInterestDB, BlockchainMetricDB

logger = logging.getLogger(__name__)

# Ordinal encoding for the fear & greed classification string
_CLASS_ENCODING: dict[str, int] = {
    "Extreme Fear": 0,
    "Fear": 1,
    "Neutral": 2,
    "Greed": 3,
    "Extreme Greed": 4,
}


def load_fear_greed() -> pd.DataFrame:
    """
    Loads Fear & Greed Index from DB.

    Returns:
        UTC-indexed DataFrame with columns:
          - fear_greed_value  : int 0-100
          - fear_greed_class  : float ordinal 0-4 (Extreme Fear → Extreme Greed)
    """
    session = SessionLocal()
    try:
        rows = session.query(FearGreedDB).order_by(FearGreedDB.timestamp).all()
        data = [
            {
                "timestamp": r.timestamp,
                "fear_greed_value": float(r.value),
                "fear_greed_class": float(_CLASS_ENCODING.get(r.classification, 2)),
            }
            for r in rows
        ]
        df = pd.DataFrame(data)
        if df.empty:
            logger.warning("External: fear_greed table is empty")
            return pd.DataFrame(columns=["fear_greed_value", "fear_greed_class"])
        df = df.set_index("timestamp")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        logger.info(f"External: loaded {len(df)} fear & greed rows")
        return df
    finally:
        session.close()


def load_funding_rate(symbol: str = "BTC/USDT:USDT") -> pd.DataFrame:
    """
    Loads funding rate (every 8h) from DB.

    Returns:
        UTC-indexed DataFrame with column:
          - funding_rate : float (e.g. 0.0001)
    """
    session = SessionLocal()
    try:
        rows = (
            session.query(FundingRateDB)
            .filter_by(symbol=symbol)
            .order_by(FundingRateDB.timestamp)
            .all()
        )
        data = [{"timestamp": r.timestamp, "funding_rate": r.rate} for r in rows]
        df = pd.DataFrame(data)
        if df.empty:
            logger.warning(f"External: funding_rates table empty for symbol={symbol}")
            return pd.DataFrame(columns=["funding_rate"])
        df = df.set_index("timestamp")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        logger.info(f"External: loaded {len(df)} funding rate rows for {symbol}")
        return df
    finally:
        session.close()


def load_open_interest(symbol: str = "BTC/USDT:USDT", timeframe: str = "1h") -> pd.DataFrame:
    """
    Loads open interest from DB for a given symbol and timeframe.

    Returns:
        UTC-indexed DataFrame with columns:
          - oi_btc_{timeframe}  : float, OI in BTC
          - oi_usdt_{timeframe} : float, OI in USDT
    """
    session = SessionLocal()
    try:
        rows = (
            session.query(OpenInterestDB)
            .filter_by(symbol=symbol, timeframe=timeframe)
            .order_by(OpenInterestDB.timestamp)
            .all()
        )
        tf_tag = timeframe  # e.g. "1h", "5m"
        data = [
            {
                "timestamp": r.timestamp,
                f"oi_btc_{tf_tag}": r.open_interest_btc,
                f"oi_usdt_{tf_tag}": r.open_interest_usdt,
            }
            for r in rows
        ]
        df = pd.DataFrame(data)
        if df.empty:
            logger.warning(f"External: open_interest table empty for {symbol} {timeframe}")
            return pd.DataFrame(columns=[f"oi_btc_{tf_tag}", f"oi_usdt_{tf_tag}"])
        df = df.set_index("timestamp")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        logger.info(f"External: loaded {len(df)} OI rows for {symbol} {timeframe}")
        return df
    finally:
        session.close()


def load_blockchain_metric(metric: str) -> pd.DataFrame:
    """
    Loads a single blockchain.com chart metric from DB.

    Args:
        metric: metric name as stored in DB (e.g. "hash-rate", "n-unique-addresses")

    Returns:
        UTC-indexed DataFrame with one column named as metric with hyphens → underscores
        (e.g. "hash-rate" → column "hash_rate")
    """
    session = SessionLocal()
    try:
        rows = (
            session.query(BlockchainMetricDB)
            .filter_by(metric=metric)
            .order_by(BlockchainMetricDB.timestamp)
            .all()
        )
        col_name = metric.replace("-", "_")
        data = [{"timestamp": r.timestamp, col_name: r.value} for r in rows]
        df = pd.DataFrame(data)
        if df.empty:
            logger.warning(f"External: blockchain_metrics table empty for metric='{metric}'")
            return pd.DataFrame(columns=[col_name])
        df = df.set_index("timestamp")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        logger.info(f"External: loaded {len(df)} rows for blockchain metric '{metric}'")
        return df
    finally:
        session.close()
