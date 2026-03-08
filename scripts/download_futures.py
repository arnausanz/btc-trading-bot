# scripts/download_futures.py
"""
Descàrrega inicial de Funding Rate i Open Interest de Binance USDT-M.
Executar una sola vegada per poblar la BD.

  - Funding Rate: historial complet des de setembre 2019 (~cada 8h)
  - Open Interest: darrers 30 dies a 1h (límit de l'API de Binance)

Ús:
    python scripts/download_futures.py
"""
import logging
import sys

sys.path.append(".")

from data.sources.futures import FuturesFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    fetcher = FuturesFetcher()

    logger.info("=== Descàrrega inicial: Funding Rate ===")
    saved_fr = fetcher.fetch_and_store_funding_rates()
    logger.info(f"Funding rates: {saved_fr} registres guardats")

    logger.info("=== Descàrrega inicial: Open Interest (darrers 30 dies a 1h) ===")
    saved_oi = fetcher.fetch_and_store_open_interest()
    logger.info(f"Open interest: {saved_oi} registres guardats")

    logger.info(f"Download completat — FR: {saved_fr}, OI: {saved_oi}")
