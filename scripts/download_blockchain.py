# scripts/download_blockchain.py
"""
Descàrrega inicial de tot l'historial de mètriques Bitcoin de Blockchain.com.
Executar una sola vegada per poblar la BD.

Mètriques descarregades:
  - hash-rate           → potència de mineria (TH/s), des de ~2009
  - n-unique-addresses  → adreces actives úniques per dia, des de ~2009
  - transaction-fees    → comissions diàries totals (en BTC), des de ~2009

Ús:
    python scripts/download_blockchain.py
"""
import logging
import sys

sys.path.append(".")

from data.sources.blockchain import BlockchainFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    fetcher = BlockchainFetcher()
    results = fetcher.fetch_all()

    for metric, count in results.items():
        logger.info(f"  {metric}: {count} registres guardats")

    total = sum(results.values())
    logger.info(f"Download completat — {total} registres totals")
