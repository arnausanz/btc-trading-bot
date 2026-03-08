# scripts/update_fear_greed.py
"""
Actualitzaci\u00f3 incremental del Fear & Greed Index.
Descarrega nom\u00e9s els darrers registres des de l\u2019\u00faltim emmagatzemat.
Pensat per executar-se via cron 1 cop al dia (les dades s\u00f3n di\u00e0ries).

Ús:
    python scripts/update_fear_greed.py

Cron exemple (cada dia a les 00:05 UTC):
    5 0 * * * cd /path/to/bot && .venv/bin/python scripts/update_fear_greed.py
"""
import logging
import sys

sys.path.append(".")

from data.sources.fear_greed import FearGreedFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    fetcher = FearGreedFetcher()
    new = fetcher.update()
    logger.info(f"Fear & Greed update: {new} new records added")
