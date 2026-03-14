# scripts/download_data.py
import logging
import sys
from datetime import datetime, timezone

sys.path.append(".")  # per poder importar des de l'arrel del projecte

from data.sources.ohlcv import OHLCVFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

if __name__ == "__main__":
    fetcher = OHLCVFetcher(exchange_id="binance")

    since = datetime(2019, 1, 1, tzinfo=timezone.utc)  # des de 2019: cobreix bear recovery, COVID, 2021 ATH, 2022 bear, 2023-24 recovery

    for timeframe in ["1h", "4h", "12h", "1d"]:
        fetcher.fetch_and_store(
            symbol="BTC/USDT",
            timeframe=timeframe,
            since=since,
        )