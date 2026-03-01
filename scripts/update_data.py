# scripts/update_data.py
import logging
import sys

sys.path.append(".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)

from data.sources.ohlcv import OHLCVFetcher

if __name__ == "__main__":
    fetcher = OHLCVFetcher(exchange_id="binance")

    for timeframe in ["1h", "4h", "1d"]:
        new = fetcher.update(symbol="BTC/USDT", timeframe=timeframe)
        logger.info(f"  {timeframe}: {new} candles noves afegides")