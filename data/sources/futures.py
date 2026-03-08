# data/sources/futures.py
"""
Descàrrega i emmagatzematge de Funding Rate i Open Interest
del contracte perpetu BTC/USDT:USDT de Binance USDT-M via ccxt.

Granularitat:
  - Funding rate: cada 8h (natural de Binance), historial complet des de set. 2019
  - Open interest: 1h per màxima granularitat; Binance conserva els darrers 30 dies

Notes sobre limitacions de l'API de Binance:
  - Funding rate: historial complet disponible (des de 2019-09-13)
  - Open interest: màxim 30 dies enrere, 500 registres per crida
"""
import logging
import ccxt
from datetime import datetime, timedelta, timezone
from sqlalchemy import func
from sqlalchemy.orm import Session
from core.models import FundingRateEntry, OpenInterestEntry
from core.db.models import FundingRateDB, OpenInterestDB
from core.db.session import SessionLocal

logger = logging.getLogger(__name__)

_SYMBOL = "BTC/USDT:USDT"           # Binance USDT-M perpetual
_OI_TIMEFRAME = "1h"                 # granularitat de l'open interest
_FUNDING_SINCE = datetime(2019, 9, 13, tzinfo=timezone.utc)  # llançament dels perps USDT-M
_OI_MAX_HISTORY_DAYS = 29            # Binance conserva menys de 30 dies d'historial d'OI


class FuturesFetcher:
    """
    Descarrega Funding Rate i Open Interest del contracte BTC/USDT:USDT
    de Binance USDT-M via ccxt i els persiteix a la BD.
    """

    def __init__(self):
        self.exchange = ccxt.binanceusdm({"enableRateLimit": True})
        logger.info("FuturesFetcher inicialitzat (binanceusdm)")

    # ---------------------------------------------------------------
    # Funding Rate
    # ---------------------------------------------------------------

    def fetch_and_store_funding_rates(
        self,
        symbol: str = _SYMBOL,
        since: datetime = _FUNDING_SINCE,
        until: datetime | None = None,
    ) -> int:
        """
        Descarrega tota la història de funding rates entre since i until.
        Pagina automàticament (1000 registres per crida).
        Retorna el nombre de registres nous guardats.
        """
        until = until or datetime.now(timezone.utc)
        since_ms = int(since.timestamp() * 1000)
        until_ms = int(until.timestamp() * 1000)

        total_saved = 0
        current_since = since_ms

        logger.info(
            f"Descàrrega funding rates {symbol} "
            f"de {since.date()} a {until.date()}"
        )

        while current_since < until_ms:
            try:
                raw = self.exchange.fetch_funding_rate_history(
                    symbol=symbol,
                    since=current_since,
                    limit=1000,
                )
            except Exception as e:
                logger.error(f"Error fetch_funding_rate_history: {e}")
                break

            if not raw:
                break

            entries: list[FundingRateEntry] = []
            for row in raw:
                if row["timestamp"] >= until_ms:
                    break
                try:
                    entry = FundingRateEntry(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(
                            row["timestamp"] / 1000, tz=timezone.utc
                        ),
                        rate=float(row["fundingRate"]),
                    )
                    entries.append(entry)
                except Exception as e:
                    logger.warning(f"Funding rate descartada: {e} — raw={row}")

            saved = self._save_funding_rates(entries)
            total_saved += saved

            last_ts = raw[-1]["timestamp"]
            logger.info(
                f"  {saved} funding rates guardades fins "
                f"{datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)}"
            )

            # Si hem rebut menys de 1000 registres → ja hem acabat
            if len(raw) < 1000:
                break
            current_since = last_ts + 1  # +1ms per evitar solapament

        logger.info(f"Funding rates completat: {total_saved} registres nous")
        return total_saved

    def _save_funding_rates(self, entries: list[FundingRateEntry]) -> int:
        """Guarda un lot de funding rates, ignorant duplicats (idempotent)."""
        if not entries:
            return 0

        session: Session = SessionLocal()
        try:
            # Carrega tots els timestamps existents per aquest symbol d'un sol cop
            existing_ts = {
                row[0]
                for row in session.query(FundingRateDB.timestamp)
                .filter(FundingRateDB.symbol == entries[0].symbol)
                .all()
            }

            to_insert = [
                FundingRateDB(
                    symbol=e.symbol,
                    timestamp=e.timestamp,
                    rate=e.rate,
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

    def get_last_funding_rate_timestamp(self, symbol: str = _SYMBOL) -> datetime | None:
        """Retorna el timestamp del darrer funding rate emmagatzemat."""
        session = SessionLocal()
        try:
            return (
                session.query(func.max(FundingRateDB.timestamp))
                .filter(FundingRateDB.symbol == symbol)
                .scalar()
            )
        finally:
            session.close()

    # ---------------------------------------------------------------
    # Open Interest
    # ---------------------------------------------------------------

    def fetch_and_store_open_interest(
        self,
        symbol: str = _SYMBOL,
        timeframe: str = _OI_TIMEFRAME,
        since: datetime | None = None,
    ) -> int:
        """
        Descarrega l'open interest des de since fins ara.
        Binance conserva un màxim de 30 dies d'historial; si since és anterior
        a 30 dies es trunca automàticament.
        Pagina fins a obtenir tots els registres disponibles (500 per crida).
        Retorna el nombre de registres nous guardats.
        """
        # Límit de 30 dies de Binance; no té sentit demanar més
        cutoff = datetime.now(timezone.utc) - timedelta(days=_OI_MAX_HISTORY_DAYS)
        since = max(since, cutoff) if since else cutoff
        since_ms = int(since.timestamp() * 1000)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        total_saved = 0
        current_since = since_ms

        logger.info(
            f"Descàrrega open interest {symbol} {timeframe} "
            f"des de {since.date()}"
        )

        while current_since < now_ms:
            try:
                raw = self.exchange.fetch_open_interest_history(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=500,
                )
            except Exception as e:
                logger.error(f"Error fetch_open_interest_history: {e}")
                break

            if not raw:
                break

            entries: list[OpenInterestEntry] = []
            for row in raw:
                try:
                    entry = OpenInterestEntry(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=datetime.fromtimestamp(
                            row["timestamp"] / 1000, tz=timezone.utc
                        ),
                        open_interest_btc=float(row["openInterestAmount"]),
                        open_interest_usdt=float(row["openInterestValue"]),
                    )
                    entries.append(entry)
                except Exception as e:
                    logger.warning(f"Open interest descartada: {e} — raw={row}")

            saved = self._save_open_interest(entries)
            total_saved += saved

            last_ts = raw[-1]["timestamp"]
            logger.info(
                f"  {saved} OI guardades fins "
                f"{datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)}"
            )

            if len(raw) < 500:
                break
            current_since = last_ts + 1

        logger.info(f"Open interest completat: {total_saved} registres nous")
        return total_saved

    def _save_open_interest(self, entries: list[OpenInterestEntry]) -> int:
        """Guarda un lot d'open interest, ignorant duplicats (idempotent)."""
        if not entries:
            return 0

        session: Session = SessionLocal()
        try:
            # Clau composta (symbol, timeframe, timestamp)
            existing_keys = {
                (row[0], row[1])
                for row in session.query(
                    OpenInterestDB.timeframe, OpenInterestDB.timestamp
                )
                .filter(
                    OpenInterestDB.symbol == entries[0].symbol,
                    OpenInterestDB.timeframe == entries[0].timeframe,
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
                if (e.timeframe, e.timestamp) not in existing_keys
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

    def get_last_open_interest_timestamp(
        self,
        symbol: str = _SYMBOL,
        timeframe: str = _OI_TIMEFRAME,
    ) -> datetime | None:
        """Retorna el timestamp del darrer open interest emmagatzemat."""
        session = SessionLocal()
        try:
            return (
                session.query(func.max(OpenInterestDB.timestamp))
                .filter(
                    OpenInterestDB.symbol == symbol,
                    OpenInterestDB.timeframe == timeframe,
                )
                .scalar()
            )
        finally:
            session.close()

    # ---------------------------------------------------------------
    # Update (incremental, per cron)
    # ---------------------------------------------------------------

    def update(
        self,
        symbol: str = _SYMBOL,
        oi_timeframe: str = _OI_TIMEFRAME,
    ) -> dict[str, int]:
        """
        Actualització incremental de funding rates i open interest.
        Retorna {'funding_rates': N, 'open_interest': N} amb els nous registres.
        Idempotent — es pot executar múltiples vegades sense duplicats.
        """
        # --- Funding rates ---
        last_fr = self.get_last_funding_rate_timestamp(symbol)
        if last_fr is None:
            logger.info(f"No hi ha funding rates per {symbol}, descàrrega completa")
            since_fr = _FUNDING_SINCE
        else:
            logger.info(f"Update funding rates {symbol} des de {last_fr}")
            since_fr = last_fr  # fetch_and_store omitirà duplicats via _save

        new_fr = self.fetch_and_store_funding_rates(
            symbol=symbol, since=since_fr
        )

        # --- Open interest ---
        last_oi = self.get_last_open_interest_timestamp(symbol, oi_timeframe)
        if last_oi is None:
            logger.info(f"No hi ha OI per {symbol} {oi_timeframe}, descàrrega completa (últims 30 dies)")
            since_oi = None  # fetch_and_store farà el cutoff automàticament
        else:
            logger.info(f"Update OI {symbol} {oi_timeframe} des de {last_oi}")
            since_oi = last_oi

        new_oi = self.fetch_and_store_open_interest(
            symbol=symbol, timeframe=oi_timeframe, since=since_oi
        )

        return {"funding_rates": new_fr, "open_interest": new_oi}
