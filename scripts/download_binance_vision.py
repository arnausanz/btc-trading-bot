# scripts/download_binance_vision.py
"""
Descàrrega inicial de tot l'historial d'Open Interest de Binance Vision S3.
Descarrega ~1.600 fitxers ZIP diaris (des de 2021-12-01 fins ahir).
Cadascun conté 288 registres de 5 minuts → ~460.000 registres totals.

Temps estimat: 5-15 minuts (depenent de la connexió).
Espai a la BD: ~40-50 MB.

Ús:
    python scripts/download_binance_vision.py

Per a un rang específic de dates:
    python scripts/download_binance_vision.py --start 2023-01-01 --end 2023-12-31
"""
import argparse
import logging
import sys
from datetime import date

sys.path.append(".")

from data.sources.binance_vision import BinanceVisionFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Descàrrega OI histèric de Binance Vision"
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=None,
        help="Data d'inici (YYYY-MM-DD). Default: 2021-12-01",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=None,
        help="Data de fi (YYYY-MM-DD). Default: ahir",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    fetcher = BinanceVisionFetcher()

    kwargs = {}
    if args.start:
        kwargs["start"] = args.start
    if args.end:
        kwargs["end"] = args.end

    saved = fetcher.fetch_and_store(**kwargs)
    logger.info(f"Download completat: {saved} nous registres guardats")
