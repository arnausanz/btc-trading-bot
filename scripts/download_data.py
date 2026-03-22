# scripts/download_data.py
import glob
import logging
import sys
from datetime import datetime, timezone

import yaml

sys.path.append(".")  # per poder importar des de l'arrel del projecte

from data.sources.ohlcv import OHLCVFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Timeframes que sempre es descarreguen independentment del que declarin els YAML.
# - 1h, 4h, 12h: usats per ML/RL bots i el Gate System (senyals de swing).
# - 1d:          usat pel Gate System (detecció de règim diari).
#   Nota: cap YAML de bot declara `timeframe: 1d` individualment, per tant sense
#   aquesta constant _collect_required_timeframes() el saltaria silenciosament.
BASE_TIMEFRAMES: set[str] = {"1h", "4h", "12h", "1d"}


def _collect_required_timeframes() -> set[str]:
    """
    Reads all config/models/*.yaml and collects every timeframe referenced.

    Checks both the top-level `timeframe` key and the `aux_timeframes` list,
    so adding a new agent with a new timeframe automatically includes it in
    the download without any manual edits to this script.

    Always starts from BASE_TIMEFRAMES so the Gate System and update_data.py
    are guaranteed to have a matching historical baseline.
    """
    tfs: set[str] = set(BASE_TIMEFRAMES)
    for path in glob.glob("config/models/*.yaml"):
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
        if not cfg:
            continue
        if tf := cfg.get("timeframe"):
            tfs.add(tf)
        tfs.update(cfg.get("aux_timeframes", []))
    return tfs


if __name__ == "__main__":
    fetcher = OHLCVFetcher(exchange_id="binance")

    since = datetime(2019, 1, 1, tzinfo=timezone.utc)  # des de 2019: cobreix bear recovery, COVID, 2021 ATH, 2022 bear, 2023-24 recovery

    timeframes = _collect_required_timeframes()
    if not timeframes:
        timeframes = BASE_TIMEFRAMES
        logging.warning("No config/models/*.yaml found — using base timeframes: %s", timeframes)

    for timeframe in sorted(timeframes):
        fetcher.fetch_and_store(
            symbol="BTC/USDT",
            timeframe=timeframe,
            since=since,
        )
