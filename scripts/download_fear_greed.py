# scripts/download_fear_greed.py
"""
Descàrrega inicial de tot l'historial del Fear & Greed Index (~2018 fins avui).
Executar una sola vegada per poblar la BD.

Ús:
    python scripts/download_fear_greed.py
"""
import logging
import sys

sys.path.append(".")

from data.sources.fear_greed import FearGreedFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if __name__ == "__main__":
    fetcher = FearGreedFetcher()
    saved = fetcher.fetch_and_store(limit=0)  # limit=0 → tot l'historial disponible
    print(f"Download completed: {saved} Fear & Greed records saved")
