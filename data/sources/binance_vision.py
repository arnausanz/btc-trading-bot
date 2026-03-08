# data/sources/binance_vision.py
"""
Descàrrega i emmagatzematge d'Open Interest histèric de Binance USDT-M
via Binance Vision S3 (bucket públic, sense API key, sense registre).

Font: https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/

Cobertura:  des de 2021-12-01 fins ahir (actualitzat diàriament amb 1 dia de lag)
Granularitat: 5 minuts (288 registres/dia)
Format:     ZIP → CSV amb columnes:
              create_time, symbol, sum_open_interest, sum_open_interest_value,
              count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
              count_long_short_ratio, sum_taker_long_short_vol_ratio

S'emmagatzema a la taula `open_interest` amb timeframe='5m', compatible
amb els registres ja existents de la REST API (timeframe='1h').
"""
import io
import logging
import zipfile
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests
from sqlalchemy import func
from sqlalchemy.orm import Session

from core.models import OpenInterestEntry
from core.db.models import OpenInterestDB
from core.db.session import SessionLocal

logger = logging.getLogger(__name__)

_SYMBOL = "BTC/USDT:USDT"     # format ccxt estàndard
_BINANCE_SYMBOL = "BTCUSDT"   # format Binance Vision
_OI_TIMEFRAME = "5m"          # granularitat dels fitxers de Vision

_HISTORY_START = date(2021, 12, 1)   # primer fitxer disponible al bucket
_RECORDS_PER_DAY = 288               # 24h × 12 intervals de 5min = 288

_BASE_URL = (
    "https://data.binance.vision/data/futures/um/daily/metrics"
    f"/{_BINANCE_SYMBOL}/{_BINANCE_SYMBOL}-metrics"
)
_REQUEST_TIMEOUT = 30  # segons per crida HTTP


class BinanceVisionFetcher:
    """
    Descarrega OI histèric de Binance Vision S3 i el persiteix a la BD.

    Característiques:
    - Gratuït i sense autenticació (S3 bucket públic de Binance)
    - Granularitat 5 minuts, des de 2021-12-01
    - Idempotent: reintenta dies incomplets, salta dies complets
    - Eficient: per a cada dia, carrega com a màxim 288 timestamps de la BD
    """

    # ---------------------------------------------------------------
    # Descàrrega d'un dia
    # ---------------------------------------------------------------

    def _download_day(self, target_date: date) -> pd.DataFrame | None:
        """
        Descarrega el fitxer ZIP d'un dia concret i retorna el DataFrame.
        Retorna None si el fitxer no existeix (data molt recent o futura).
        """
        url = f"{_BASE_URL}-{target_date}.zip"
        try:
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
            if resp.status_code == 404:
                return None  # fitxer no disponible per aquesta data
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"  Error HTTP {target_date}: {e}")
            return None

        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    return pd.read_csv(f)
        except Exception as e:
            logger.warning(f"  Error parsejant ZIP {target_date}: {e}")
            return None

    # ---------------------------------------------------------------
    # Parsing de les dades
    # ---------------------------------------------------------------

    def _parse_oi(self, df: pd.DataFrame, day: date) -> list[OpenInterestEntry]:
        """
        Converteix el DataFrame del CSV en objectes OpenInterestEntry (Pydantic).
        Descarta files invàlides i en registra un warning.
        """
        entries: list[OpenInterestEntry] = []
        for _, row in df.iterrows():
            try:
                # create_time format: "YYYY-MM-DD HH:MM:SS" (UTC implícit)
                ts = datetime.strptime(
                    str(row["create_time"]), "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=timezone.utc)

                entry = OpenInterestEntry(
                    symbol=_SYMBOL,
                    timeframe=_OI_TIMEFRAME,
                    timestamp=ts,
                    open_interest_btc=float(row["sum_open_interest"]),
                    open_interest_usdt=float(row["sum_open_interest_value"]),
                )
                entries.append(entry)
            except Exception as e:
                logger.warning(f"  Fila descartada ({day}): {e} — row={dict(row)}")

        return entries

    # ---------------------------------------------------------------
    # Comprovació de completesa per dia
    # ---------------------------------------------------------------

    def _needs_download(self, session: Session, day: date) -> bool:
        """
        Retorna True si el dia NO té dades completes a la BD.
        Evita descàrregues innecessàries quan ja s'han processat dies anteriors.
        """
        day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)

        count = (
            session.query(func.count(OpenInterestDB.id))
            .filter(
                OpenInterestDB.symbol == _SYMBOL,
                OpenInterestDB.timeframe == _OI_TIMEFRAME,
                OpenInterestDB.timestamp >= day_start,
                OpenInterestDB.timestamp < day_end,
            )
            .scalar()
        ) or 0

        return count < _RECORDS_PER_DAY

    # ---------------------------------------------------------------
    # Emmagatzematge eficient per dia
    # ---------------------------------------------------------------

    def _save(self, entries: list[OpenInterestEntry], day: date) -> int:
        """
        Guarda els registres d'un dia a la BD, ignorant duplicats.
        Carrega timestamp existents NOMÉS per al dia en qüestió (≤288 files)
        per evitar carregar tot l'historial en cada iteració.
        """
        if not entries:
            return 0

        day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)

        session: Session = SessionLocal()
        try:
            existing_ts = {
                row[0]
                for row in session.query(OpenInterestDB.timestamp)
                .filter(
                    OpenInterestDB.symbol == _SYMBOL,
                    OpenInterestDB.timeframe == _OI_TIMEFRAME,
                    OpenInterestDB.timestamp >= day_start,
                    OpenInterestDB.timestamp < day_end,
                )
                .all()
            }

            to_insert = [
                OpenInterestDB(
                    symbol=e.symbol,
                    timeframe=e.timeframe,
                    timestamp=e.timestamp,
                    open_interest_btc=e.open_interest_btc,
                    open_interest_usdt=e.open_interest_usdt,
                )
                for e in entries
                if e.timestamp not in existing_ts
            ]

            if to_insert:
                session.bulk_save_objects(to_insert)
                session.commit()

            return len(to_insert)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ---------------------------------------------------------------
    # Descàrrega d'un rang de dates (amb skip intel·ligent)
    # ---------------------------------------------------------------

    def fetch_and_store(
        self,
        start: date = _HISTORY_START,
        end: date | None = None,
    ) -> int:
        """
        Descarrega i emmagatzema els fitxers de Vision entre start i end.
        - Salta dies que ja té dades completes (idempotent i eficient).
        - Reinicia dies parcialment descarregats.
        - Retorna el nombre de nous registres guardats.

        end per defecte: ahir (Vision publica amb 1 dia de lag).
        """
        # Vision publica les dades amb ~1 dia de lag; no demanem avui
        end = end or (datetime.now(timezone.utc).date() - timedelta(days=1))

        total_saved = 0
        total_days = (end - start).days + 1
        skipped = 0
        downloaded = 0

        logger.info(
            f"Binance Vision OI: {start} → {end} "
            f"({total_days} dies, granularitat {_OI_TIMEFRAME})"
        )

        # Sessió compartida per als checks de completesa (evita obrir N sessions)
        check_session: Session = SessionLocal()
        try:
            current = start
            while current <= end:
                if not self._needs_download(check_session, current):
                    skipped += 1
                    current += timedelta(days=1)
                    continue

                df = self._download_day(current)
                if df is not None and not df.empty:
                    entries = self._parse_oi(df, current)
                    saved = self._save(entries, current)
                    total_saved += saved
                    downloaded += 1

                    # Log cada 50 dies o si s'han guardat nous registres
                    if downloaded % 50 == 0 or saved > 0:
                        logger.info(
                            f"  {current}: {saved}/{len(df)} nous "
                            f"({downloaded} descarregats, {skipped} omesos)"
                        )
                else:
                    logger.debug(f"  {current}: fitxer no disponible, omès")

                current += timedelta(days=1)
        finally:
            check_session.close()

        logger.info(
            f"Binance Vision completat: {total_saved} nous registres "
            f"({downloaded} dies descarregats, {skipped} omesos)"
        )
        return total_saved

    # ---------------------------------------------------------------
    # Timestamps i actualització incremental
    # ---------------------------------------------------------------

    def get_last_timestamp(self) -> datetime | None:
        """Retorna el timestamp del darrer OI de Vision emmagatzemat."""
        session = SessionLocal()
        try:
            return (
                session.query(func.max(OpenInterestDB.timestamp))
                .filter(
                    OpenInterestDB.symbol == _SYMBOL,
                    OpenInterestDB.timeframe == _OI_TIMEFRAME,
                )
                .scalar()
            )
        finally:
            session.close()

    def update(self) -> int:
        """
        Actualització incremental: descarrega els dies que falten des de
        l'últim registre emmagatzemat fins a ahir.
        Si la BD és buida, descarrega tot l'historial des de 2021-12-01.
        Idempotent — es pot executar múltiples vegades sense duplicats.
        """
        last = self.get_last_timestamp()

        if last is None:
            logger.info("No hi ha dades de Vision, descàrrega completa des de 2021-12-01")
            return self.fetch_and_store()

        # Comencem des de l'últim dia emmagatzemat (per si estava incomplet)
        start = last.date()
        logger.info(f"Vision OI update des de {start}")
        return self.fetch_and_store(start=start)
