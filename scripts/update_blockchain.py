# scripts/update_blockchain.py
"""
Actualitzaci\u00f3 incremental de les m\u00e8triques Bitcoin de Blockchain.com.
Pensat per executar-se via cron 1 cop al dia (les dades s\u00f3n di\u00e0ries).

Comportament per a cada m\u00e8trica:
  - Si la BD \u00e9s buida \u2192 descarrega tot l'hist\u00f2ric
  - Si ja hi ha dades \u2192 descarrega els darrers 5 dies (buffer per correccions de l'API)
  - La deduplicaci\u00f3 gestiona els solapaments autom\u00e0ticament

\u00da\u00fas:
    python scripts/update_blockchain.py

Cron exemple (cada dia a les 00:10 UTC):
    10 0 * * * cd /path/to/bot && .venv/bin/python scripts/update_blockchain.py
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
    results = fetcher.update_all()

    for metric, count in results.items():
        logger.info(f"  {metric}: {count} nous registres")

    total = sum(results.values())
    logger.info(f"Update completat — {total} nous registres totals")
