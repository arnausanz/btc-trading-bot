# data/sources/fear_greed.py
import logging
import requests
from datetime import datetime, timezone
from sqlalchemy import func
from sqlalchemy.orm import Session
from core.models import FearGreedEntry
from core.db.models import FearGreedDB
from core.db.session import SessionLocal

logger = logging.getLogger(__name__)

_API_URL = "https://api.alternative.me/fng/"


class FearGreedFetcher:
    """
    Downloads Fear & Greed Index data from alternative.me API
    and persists it to the DB. Diari, des de febrer 2018.
    """

    def fetch_and_store(self, limit: int = 0) -> int:
        """
        Downloads Fear & Greed records from the API and saves them to the DB.
        limit=0 fetches all available history (~2018 fins avui).
        Returns the number of new records saved.
        """
        params: dict = {"format": "json", "limit": limit if limit > 0 else 0}
        logger.info(f"Fetching Fear & Greed (limit={'all' if limit == 0 else limit})")

        response = requests.get(_API_URL, params=params, timeout=30)
        response.raise_for_status()

        payload = response.json()
        api_error = payload.get("metadata", {}).get("error")
        if api_error:
            raise RuntimeError(f"Fear & Greed API error: {api_error}")

        raw_entries = payload.get("data", [])
        if not raw_entries:
            logger.warning("API returned no Fear & Greed data")
            return 0

        entries: list[FearGreedEntry] = []
        for raw in raw_entries:
            try:
                entry = FearGreedEntry(
                    timestamp=datetime.fromtimestamp(int(raw["timestamp"]), tz=timezone.utc),
                    value=int(raw["value"]),
                    classification=raw["value_classification"],
                )
                entries.append(entry)
            except Exception as e:
                logger.warning(f"Invalid Fear & Greed entry discarded: {e} — raw={raw}")

        saved = self._save(entries)
        logger.info(f"Fear & Greed: {saved} new records saved ({len(entries)} fetched)")
        return saved

    def _save(self, entries: list[FearGreedEntry]) -> int:
        """Saves entries to DB, ignoring duplicates (idempotent)."""
        if not entries:
            return 0

        session: Session = SessionLocal()
        try:
            # Carrega tots els timestamps existents en un set per evitar N queries
            existing_ts = {
                row[0]
                for row in session.query(FearGreedDB.timestamp).all()
            }

            to_insert = [
                FearGreedDB(
                    timestamp=e.timestamp,
                    value=e.value,
                    classification=e.classification,
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

    def get_last_timestamp(self) -> datetime | None:
        """Returns the most recent timestamp stored in the DB."""
        session = SessionLocal()
        try:
            return session.query(func.max(FearGreedDB.timestamp)).scalar()
        finally:
            session.close()

    def update(self) -> int:
        """
        Downloads only new data since the last stored record.
        Si la BD és buida, descarrega tot l'historial.
        Idempotent — es pot executar múltiples vegades sense duplicats.
        """
        last = self.get_last_timestamp()

        if last is None:
            logger.info("No Fear & Greed data in DB, downloading full history")
            return self.fetch_and_store(limit=0)

        logger.info(f"Updating Fear & Greed from {last.date()}")
        # Agafem els darrers 2 dies per cobrir possibles correccions de l'API
        return self.fetch_and_store(limit=2)
