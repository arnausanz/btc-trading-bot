# scripts/update_binance_vision.py
"""
Actualitzaci\u00f3 incremental de l'Open Interest de Binance Vision S3.
Descarrega els dies que falten des de l'\u00faltim registre fins a ahir.

Vision publica les dades amb ~1 dia de lag, de manera que:
  - Cron diari a les 08:00 UTC garanteix que el fitxer d'ahir ja \u00e9s disponible.
  - Executar m\u00e9s d'un cop al dia \u00e9s segur (idempotent, salta dies ja complets).

\u00da\u00fas:
    python scripts/update_binance_vision.py

Cron exemple (cada dia a les 08:00 UTC):
    0 8 * * * cd /path/to/bot && .venv/bin/python scripts/update_binance_vision.py
"""
import logging
import sys

sys.path.append(".")

from data.sources.binance_vision import BinanceVisionFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    fetcher = BinanceVisionFetcher()
    saved = fetcher.update()
    logger.info(f"Vision OI update completat: {saved} nous registres")
