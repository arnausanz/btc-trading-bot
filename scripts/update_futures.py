# scripts/update_futures.py
"""
Actualitzaci\u00f3 incremental de Funding Rate i Open Interest de Binance USDT-M.
Pensat per executar-se via cron cada hora (la freq\u00fc\u00e8ncia de l'OI a 1h).

Comportament:
  - Funding Rate: descarrega des de l'\u00faltim registre emmagatzemat fins ara
  - Open Interest: descarrega des de l'\u00faltim registre (m\u00e0xim 30 dies enrere)
  - Si la BD \u00e9s buida per a qualsevol font, descarrega l'hist\u00f2ric complet disponible

\u00da\u00fas:
    python scripts/update_futures.py

Cron exemple (cada hora a :05):
    5 * * * * cd /path/to/bot && .venv/bin/python scripts/update_futures.py
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
    results = fetcher.update()
    logger.info(
        f"Update completat — "
        f"Funding rates: {results['funding_rates']} nous, "
        f"Open interest: {results['open_interest']} nous"
    )
